"""任务适配器：一个任务从标注到部署的每一步由它决定（docs/training.md §11）。

适配器是模块，提供：
  NAME                         任务名，即 model.yaml 的 task
  source_labels(spec)          来源中必须存在的标签（类别依赖，§15.1）：检测是 classes，二级分类是 cascade.classes
  samples(annotation, spec, ignore)
                               从一份已检查的标注解读出对象（丢弃 ignore 中的标签），返回 (objects, problems, ignored)，
                               ignored 是不收入的形状按原因（给人看的短语）计数
  manifest_fields(spec)        数据集清单中除样本外决定训练结果的规格字段（进入数据集标识）
  statistics(samples, spec)    数据集报告中每个划分的统计（类别 → 实例数）
  consistency_inputs(root, samples, spec, limit)
                               导出一致性比较的网络输入（float32 [3, 高, 宽]，最多 limit 个），与训练样本的生成方式相同
  METRICS                      评估指标的名称 → 比较方向（"higher" 或 "lower"），发布门槛按它比较
  LAYOUTS                      支持的部署输出布局（model.yaml 的 output_layout）
  labels_text(classes)         发布的 labels.txt 的内容（格式由 nvinfer 的网络类型决定）
  compare(layout, native, exported)
                               导出一致性：原生推理与 ONNX 对同一输入的输出差异，返回含 passed 的摘要
支持预标注的任务（检测）另外提供：
  prelabel_shapes(predictions, width, height, classes)
                               后端的预测（类别号按预标注模型的 classes）→ 预标注写入的 X-AnyLabeling 形状
  added_shapes(shapes, annotation, label)
                               按类别预标注时要追加的形状：与标注中已有的同类别形状重合的不写
  corrections(shapes, annotation, label=None)
                               预标注写入的形状与人确认后的标注之间的修正量（kept、removed、added）；
                               给出 label 时只比较这个类别
新增任务只加模块并登记在 TASKS 中；规划中尚未实现的任务明确报告“尚未支持”。
"""

from __future__ import annotations

from types import ModuleType

from ..errors import TrainError
from . import classify, detect

TASKS: dict[str, ModuleType] = {detect.NAME: detect, classify.NAME: classify}
PLANNED = ("segment",)


def get(name: object) -> ModuleType:
    if name in TASKS:
        return TASKS[name]
    if name in PLANNED:
        raise TrainError(f"task {name} is not supported yet (planned; supported: {', '.join(sorted(TASKS))})")
    raise TrainError(f"unknown task {name!r}; supported: {', '.join(sorted(TASKS))}")
