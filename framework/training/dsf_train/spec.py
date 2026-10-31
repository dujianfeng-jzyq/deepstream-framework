"""模型规格（models/<id>/model.yaml）的读取与训练相关的校验。

部署契约（类别、输入尺寸、输出布局）由 dsf_deploy.models.check_spec 检查，与打包时同一套规则；
这里补充训练才需要的部分：任务是否已支持、数据来源与划分参数。
"""

from __future__ import annotations

from pathlib import Path

from dsf_deploy import models

from . import paths, tasks
from .errors import TrainError

# train init 生成的骨架。类别由命令行给出；其余是检测任务在 Ultralytics 上的常用起点。
SKELETON = """# {model_id}：{display_name}。模型规格，人工维护；训练、发布与打包都从这里读取。
# 发布事实（产物、来源、指标）由发布工具写进 release.json，不手改。见 docs/training.md。
schema: 1
id: {model_id}
display_name: {display_name}
task: {task}

# 顺序即类别号，发布后不能调换或删减；新增类别追加在末尾。名称同时是标注中的类别名与 OSD 标签。
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
        tasks.get(spec.get("task"))
    except TrainError as error:
        problems.append(str(error))
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
