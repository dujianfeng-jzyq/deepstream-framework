"""模型规格（models/<id>/model.yaml）的读取与训练相关的校验。

部署契约（类别、输入尺寸、输出布局）由 dsf_deploy.models.check_spec 检查，与打包时同一套规则；
这里补充训练才需要的部分：任务是否已支持、数据来源与划分参数。
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from dsf_deploy import models

from . import backends, paths, tasks
from .errors import TrainError

# train init 生成的骨架。类别由命令行给出；其余是各任务在 Ultralytics 上的常用起点。
SKELETON = """# {model_id}：{display_name}。模型规格，人工维护；训练、发布与打包都从这里读取。
# 发布事实（产物、来源、指标）由发布工具写进 release.json，不手改。见 docs/training.md。
schema: 1
id: {model_id}
display_name: {display_name}
task: {task}

# 顺序即类别号，名称同时是标注中的类别名与 OSD 标签。用 train update 修改（docs/training.md §15），
# 它同时改写来源与共用来源的模型；手工新增或改名的类别无法 prepare。
classes:
{classes}
# 部署契约：网络输入尺寸与输出布局。输出布局由 nvinfer 配置使用的解析插件定义，按项目中的插件填写。
input: {{width: 640, height: 640}}
output_layout: xyxy-confidence-class_id

# 数据集：训练根目录 sources/ 下的来源（train import 的 --source），按需追加。
data:
  sources: []
  ignore_classes: []
  split:
    val: 0.15
    test: 0.15
    seed: 42

# 训练：options 原样交给后端（Ultralytics 的 train 参数）。
train:
  backend: ultralytics
  weights: yolo11s.pt
  options:
    epochs: 200
    batch: 16
    imgsz: 640
    patience: 50

# 发布门槛：min 是绝对下限（首次发布没有可对照的旧模型，只看它）；regression 与当前发布比较。
evaluate:
  gate:
    min: {{}}
    regression: {{metric: map50_95, max_drop: 0.01}}

export:
  format: onnx
  opset: 17
  batch: 1
  dynamic: false
  simplify: true
"""


# 二级分类（docs/training.md §12）：类别是父类别框上一个属性的值，样本是这些框的裁剪。
CLASSIFY_SKELETON = """# {model_id}：{display_name}。二级分类模型规格，人工维护；训练、发布与打包都从这里读取。
# 发布事实（产物、来源、指标）由发布工具写进 release.json，不手改。见 docs/training.md §12。
schema: 1
id: {model_id}
display_name: {display_name}
task: classify

# 顺序即类别号，名称是 data.from_parent.attribute 的取值，同时是二级分类结果的标签（labels.txt）。
classes:
{classes}
# 作用于父模型检测到的这些类别的目标（nvinfer process-mode=2）。按名称写，打包时换算成父模型的类别号，
# 与二级模型 nvinfer 配置的 operate-on-class-ids 对照。
cascade:
  parent: {parent}
  classes:
{parent_classes}
# 部署契约：网络输入（正方形；目标框拉伸到这个尺寸，不保持宽高比）与输出布局（softmax 概率）。
input: {{width: 128, height: 128}}
output_layout: softmax

# 数据集：样本是来源中父类别框的裁剪，类别 = 框上 attribute 的值。来源通常与父模型相同，划分沿用
# 图片的采集组，两级模型的测试集来自同一批图片。ignore_classes 列出不收入的属性值（例如“未知”）。
# 不外扩：二级 nvinfer 裁剪的就是检测框本身，训练样本与现场一致。
data:
  sources: []
  ignore_classes: []
  from_parent:
    attribute: {attribute}
    min_size: 16
  split:
    val: 0.15
    test: 0.15
    seed: 42

train:
  backend: ultralytics
  weights: yolo11s-cls.pt
  options:
    epochs: 100
    batch: 64
    patience: 20

# recall 是各类别召回率的平均：类别不均衡时 accuracy 会被多数类掩盖。
evaluate:
  gate:
    min: {{}}
    regression: {{metric: recall, max_drop: 0.01}}

export:
  format: onnx
  opset: 17
  batch: 1
  dynamic: false
  simplify: true
"""
FROM_PARENT_KEYS = ("attribute", "min_size")
# train.class_weights 取它时按训练划分的样本数取反比（总数 / (类别数 × 该类样本数)）。
BALANCED = "balanced"


def load_all(repo: Path) -> dict[str, dict]:
    """仓库中全部模型的规格（按 id 排序）。任何一个规格不成立都报错：依赖它们的结果算不全。"""
    from dsf_deploy import layout
    root = repo / layout.MODELS
    found = {}
    for directory in sorted(p for p in root.iterdir() if (p / models.SPEC).is_file()) if root.is_dir() else []:
        found[directory.name] = load(repo, directory.name)
    return found


def path(repo: Path, model_id: str) -> Path:
    return paths.model_dir(repo, model_id) / models.SPEC


def load(repo: Path, model_id: str) -> dict:
    """读取并校验规格；部署契约与训练字段的问题一次列出。"""
    spec_path = path(repo, model_id)
    if not spec_path.is_file():
        raise TrainError(f"{spec_path} does not exist; create it with: train init {model_id} --task detect")
    from dsf import yamlio
    try:
        spec = yamlio.load(spec_path.read_text(encoding="utf-8")) or {}
    except Exception as error:
        raise TrainError(f"{spec_path}: {error}") from error
    problems = models.check_spec(model_id, spec)
    if not problems:
        problems = _check_training(spec)
    if not problems and spec["task"] == "classify":
        problems = _check_parent(repo, spec)
    if problems:
        raise TrainError(f"{spec_path}:\n  " + "\n  ".join(problems))
    return spec


def _check_training(spec: dict) -> list[str]:
    problems = []
    try:
        task = tasks.get(spec.get("task"))
    except TrainError as error:
        problems.append(str(error))
        task = None
    if task is not None:
        problems += _check_train(spec.get("train"), task.NAME, spec.get("classes")) \
            + _check_evaluate(spec.get("evaluate"), task)
    problems += _check_export(spec.get("export"))
    data = spec.get("data")
    if not isinstance(data, dict):
        return problems + ["data must be a mapping"]
    if task is not None and task.NAME == "classify":
        problems += _check_classify(spec, data)
    else:
        problems += [f"{key} is only for second-stage classify models" for key, present in
                     (("cascade", spec.get("cascade") is not None), ("data.from_parent", "from_parent" in data))
                     if present]
    sources = data.get("sources")
    if not isinstance(sources, list) or not all(isinstance(s, str) and s for s in sources):
        problems.append("data.sources must be a list of source names")
    ignore = data.get("ignore_classes", [])
    if not isinstance(ignore, list) or set(ignore) & set(spec["classes"]):
        problems.append("data.ignore_classes must be a list of names that are not classes")
    split = data.get("split")
    if not isinstance(split, dict) or not isinstance(split.get("seed"), int):
        problems.append("data.split must give an integer seed")
    else:
        fractions = [split.get("val"), split.get("test")]
        if not all(isinstance(v, (int, float)) and 0 <= v < 1 for v in fractions) or sum(fractions) >= 1:
            problems.append("data.split val and test must be fractions whose sum is below 1")
    return problems


def _check_classify(spec: dict, data: dict) -> list[str]:
    """二级分类：cascade 指出父模型与触发的父类别，data.from_parent 指出作为类别的属性。整图分类还不支持。"""
    problems = []
    cascade = spec.get("cascade")
    if not isinstance(cascade, dict):
        problems.append("cascade is required: {parent: <model id>, classes: [<parent classes>]} "
                        "(whole-image classification is not supported yet)")
    else:
        parent, classes = cascade.get("parent"), cascade.get("classes")
        if not isinstance(parent, str) or not parent or parent == spec["id"]:
            problems.append("cascade.parent must name another model")
        if not isinstance(classes, list) or not classes or not all(isinstance(c, str) and c for c in classes) \
                or len(set(classes)) != len(classes):
            problems.append("cascade.classes must list parent classes by name, each once")
        unknown = sorted(set(cascade) - {"parent", "classes"})
        if unknown:
            problems.append(f"cascade: unknown keys {', '.join(unknown)}")
    settings = data.get("from_parent")
    if not isinstance(settings, dict) or not isinstance(settings.get("attribute"), str) or \
            not settings["attribute"]:
        problems.append("data.from_parent.attribute must name the box attribute whose values are the classes")
    else:
        if "min_size" in settings and not (_integer(settings["min_size"]) and settings["min_size"] >= 1):
            problems.append("data.from_parent.min_size must be an integer >= 1 (pixels)")
        if "padding" in settings:
            problems.append("data.from_parent.padding is not supported: the second-stage nvinfer crops the "
                            "detected box itself, so training crops must not be padded")
        unknown = sorted(set(settings) - set(FROM_PARENT_KEYS) - {"padding"})
        if unknown:
            problems.append(f"data.from_parent: unknown keys {', '.join(unknown)}")
    if spec["input"]["width"] != spec["input"]["height"]:
        problems.append("input of a classify model must be square (the backend trains square inputs)")
    return problems


def _check_parent(repo: Path, spec: dict) -> list[str]:
    """父模型存在、是检测模型，且有 cascade.classes 中的每个类别（按它当前的规格）。"""
    parent = spec["cascade"]["parent"]
    parent_path = path(repo, parent)
    if not parent_path.is_file():
        return [f"cascade.parent: model {parent} does not exist ({parent_path})"]
    from dsf import yamlio
    try:
        parent_spec = yamlio.load(parent_path.read_text(encoding="utf-8")) or {}
    except Exception as error:
        return [f"cascade.parent: {parent_path}: {error}"]
    if parent_spec.get("task") != "detect":
        return [f"cascade.parent: {parent} is a {parent_spec.get('task')} model; the parent must detect objects"]
    missing = [name for name in spec["cascade"]["classes"] if name not in (parent_spec.get("classes") or [])]
    if missing:
        return [f"cascade.classes: {parent} has no class {', '.join(missing)}"]
    return []


def _check_train(train: object, task: str, classes: object = None) -> list[str]:
    if not isinstance(train, dict):
        return ["train must be a mapping"]
    try:
        backend = backends.get(train.get("backend"), task)
    except TrainError as error:
        return [str(error)]
    problems = []
    weights = train.get("weights")
    if not isinstance(weights, str) or not weights:
        problems.append("train.weights must name the starting weights (a pretrained file name, run:<run> "
                        "or an absolute path)")
    options = train.get("options", {})
    if not isinstance(options, dict):
        problems.append("train.options must be a mapping")
    else:
        problems += [f"train.options: {problem}" for problem in backend.check_options(options)]
    if train.get("class_weights") is not None:
        problems += _check_class_weights(train["class_weights"], task, classes)
    return problems


def _check_class_weights(value: object, task: str, classes: object) -> list[str]:
    if task != "classify":
        return [f"train.class_weights: only classify models take class weights ({task} does not)"]
    if value == BALANCED:
        return []
    if not isinstance(value, dict) or not value:
        return [f"train.class_weights must be {BALANCED} or a mapping of class name to weight"]
    names = classes if isinstance(classes, list) else []
    problems = [f"train.class_weights: no class {name!r} in classes" for name in value if name not in names]
    problems += [f"train.class_weights: the weight of {name} must be a positive number"
                 for name, weight in value.items()
                 if isinstance(weight, bool) or not isinstance(weight, (int, float)) or weight <= 0]
    return problems


def _check_evaluate(evaluate: object, task) -> list[str]:
    """发布门槛只能引用任务定义的指标：写错名称时在训练前报告，而不是到发布时才发现门槛从未生效。"""
    if evaluate is None:
        return []
    gate = evaluate.get("gate", {}) if isinstance(evaluate, dict) else None
    if not isinstance(gate, dict):
        return ["evaluate.gate must be a mapping"]
    known = ", ".join(task.METRICS)
    problems = []
    minimum = gate.get("min") or {}
    if not isinstance(minimum, dict):
        problems.append("evaluate.gate.min must map metric names to lower bounds")
    else:
        for name, value in minimum.items():
            if name not in task.METRICS:
                problems.append(f"evaluate.gate.min: unknown metric {name!r} (task {task.NAME}: {known})")
            elif not _number(value):
                problems.append(f"evaluate.gate.min.{name} must be a number")
    regression = gate.get("regression")
    if regression is not None:
        if not isinstance(regression, dict) or regression.get("metric") not in task.METRICS or \
                not _number(regression.get("max_drop")) or regression["max_drop"] < 0:
            problems.append(f"evaluate.gate.regression must be {{metric: <{known}>, max_drop: <number >= 0>}}")
    return problems


def _check_export(export: object) -> list[str]:
    """导出参数决定发布的 ONNX 签名（固定 batch 时 nvinfer 的 batch-size 必须与之相等），写错时在训练前报告。"""
    if not isinstance(export, dict):
        return ["export must be a mapping: {format: onnx, opset, batch, dynamic, simplify}"]
    problems = []
    if export.get("format") != "onnx":
        problems.append("export.format must be onnx")
    for key, check, text in (("opset", lambda v: _integer(v) and v >= 11, "an integer >= 11"),
                             ("batch", lambda v: _integer(v) and v >= 1, "an integer >= 1"),
                             ("dynamic", lambda v: isinstance(v, bool), "true or false"),
                             ("simplify", lambda v: isinstance(v, bool), "true or false")):
        if not check(export.get(key)):
            problems.append(f"export.{key} must be {text}")
    unknown = sorted(set(export) - {"format", "opset", "batch", "dynamic", "simplify"})
    if unknown:
        problems.append(f"export: unknown keys {', '.join(unknown)}")
    return problems


def _integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def write_classes(repo: Path, model_id: str, classes: list[str]) -> None:
    """改写 model.yaml 的 classes，其余文本（注释、其他字段）不变。块列表逐项改写：保留的项连同行尾注释不动，
    改名的项只换名称，删除的项连同行删去，新增的项追加在最后一项之后；流式列表整行换成块列表。改写后重新
    读取核对，结果不符时不写入，请人手改。"""
    from dsf import yamlio
    spec_path = path(repo, model_id)
    text = spec_path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    start = next((i for i, line in enumerate(lines) if re.match(r"classes\s*:", line)), None)
    if start is None:
        raise TrainError(f"{spec_path}: no top-level classes: to rewrite")
    old = yamlio.load(text)["classes"]
    head, _, rest = lines[start].partition(":")
    block = "".join(f"  - {_scalar(name)}\n" for name in classes)
    if rest.split("#", 1)[0].strip():                      # classes: [a, b]
        lines[start:start + 1] = [f"{head}:\n", block]
    else:
        items, end = [], start + 1
        while end < len(lines) and (not lines[end].strip() or lines[end][0] in " \t#"):
            if re.match(r"\s+-\s", lines[end]):
                items.append(end)
            end += 1
        if len(items) != len(old):
            raise TrainError(f"{spec_path}: cannot locate each class on its own line; edit classes by hand")
        kept = {name: index for name, index in zip(old, items)}
        renamed = {}
        if len(old) == len(classes):
            renamed = {items[i]: classes[i] for i in range(len(old)) if old[i] != classes[i]}
        result = list(lines)
        for index, name in renamed.items():
            indent = re.match(r"(\s+-\s+)", lines[index]).group(1)
            comment = re.search(r"\s+#.*$", lines[index].rstrip("\n"))
            result[index] = f"{indent}{_scalar(name)}{comment.group(0) if comment else ''}\n"
        if not renamed:
            for name, index in kept.items():
                if name not in classes:
                    result[index] = ""
            appended = "".join(f"  - {_scalar(name)}\n" for name in classes if name not in kept)
            result[items[-1]] = result[items[-1]] + appended if result[items[-1]] else appended
        lines = result
    new_text = "".join(lines)
    try:
        written = yamlio.load(new_text).get("classes")
    except Exception:
        written = None
    if written != classes:
        raise TrainError(f"{spec_path}: rewriting classes did not give {classes}; edit classes by hand")
    fd, name = paths.temporary_file(spec_path)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(new_text)
    os.replace(name, spec_path)


def skeleton(model_id: str, task: str, classes: list[str], display_name: str = "", parent: str = "",
             parent_classes: list[str] | None = None, attribute: str = "") -> str:
    tasks.get(task)  # 尚未支持的任务在生成规格时就报告，而不是到训练时
    if not classes or len(set(classes)) != len(classes) or not all(c.strip() for c in classes):
        raise TrainError("give at least one class, each once (--class or --classes-file)")
    lines = "".join(f"  - {_scalar(name)}\n" for name in classes)
    second_stage = (parent, parent_classes, attribute)
    if task != "classify":
        if any(second_stage):
            raise TrainError("--parent, --parent-class and --attribute are only for --task classify")
        return SKELETON.format(model_id=model_id, display_name=display_name or model_id, task=task, classes=lines)
    if not all(second_stage):
        raise TrainError("a classify model is a second-stage model: give --parent <model>, --parent-class <class> "
                         "(repeat) and --attribute <box attribute whose values are the classes>")
    return CLASSIFY_SKELETON.format(
        model_id=model_id, display_name=display_name or model_id, classes=lines, parent=_scalar(parent),
        parent_classes="".join(f"    - {_scalar(name)}\n" for name in parent_classes), attribute=_scalar(attribute))


def _scalar(value: str) -> str:
    """类别名写成 YAML 标量：含有 YAML 特殊字符时加双引号。"""
    special = set(":#{}[],&*!|>'\"%@`") | {" "}
    try:
        float(value)
        numeric = True
    except ValueError:
        numeric = False
    if numeric or value != value.strip() or set(value) & special or \
            value.lower() in ("yes", "no", "on", "off", "true", "false", "null", "~"):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return value
