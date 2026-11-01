"""任务适配器：一个任务从标注到部署的每一步由它决定（docs/training.md §11）。

适配器是模块，提供：
  NAME                         任务名，即 model.yaml 的 task
  samples(annotation, spec)    从一份已检查的标注解读出对象，返回 (objects, problems, ignored_shapes)
  statistics(samples, spec)    数据集报告中每个划分的统计（类别 → 实例数）
  METRICS                      评估指标的名称 → 比较方向（"higher" 或 "lower"），发布门槛按它比较
  prelabel_shapes(predictions, width, height, spec)
                               后端的预测 → 预标注写入的 X-AnyLabeling 形状
  LAYOUTS                      支持的部署输出布局（model.yaml 的 output_layout）
  labels_text(classes)         发布的 labels.txt 的内容（格式由 nvinfer 的网络类型决定）
  compare(layout, native, exported)
                               导出一致性：原生推理与 ONNX 对同一输入的输出差异，返回含 passed 的摘要
  corrections(shapes, annotation)
                               预标注写入的形状与人确认后的标注之间的修正量（kept、removed、added）
新增任务只加模块并登记在 TASKS 中；规划中尚未实现的任务明确报告“尚未支持”。
"""

from __future__ import annotations

from types import ModuleType

from ..errors import TrainError
from . import detect

TASKS: dict[str, ModuleType] = {detect.NAME: detect}
PLANNED = ("classify", "segment")


def get(name: object) -> ModuleType:
    if name in TASKS:
        return TASKS[name]
    if name in PLANNED:
        raise TrainError(f"task {name} is not supported yet (planned; supported: {', '.join(sorted(TASKS))})")
    raise TrainError(f"unknown task {name!r}; supported: {', '.join(sorted(TASKS))}")
