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

# train init 生成的骨架。类别由命令行给出；其余是检测任务在 Ultralytics 上的常用起点。
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
        problems += _check_train(spec.get("train"), task.NAME) + _check_evaluate(spec.get("evaluate"), task)
    problems += _check_export(spec.get("export"))
    if spec.get("cascade") is not None:
        problems.append("cascade (second-stage models) is not supported yet")
    data = spec.get("data")
    if not isinstance(data, dict):
        return problems + ["data must be a mapping"]
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


def _check_train(train: object, task: str) -> list[str]:
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


def skeleton(model_id: str, task: str, classes: list[str], display_name: str = "") -> str:
    tasks.get(task)  # 尚未支持的任务在生成规格时就报告，而不是到训练时
    if not classes or len(set(classes)) != len(classes) or not all(c.strip() for c in classes):
        raise TrainError("give at least one class, each once (--class or --classes-file)")
    lines = "".join(f"  - {_scalar(name)}\n" for name in classes)
    return SKELETON.format(model_id=model_id, display_name=display_name or model_id, task=task, classes=lines)


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
