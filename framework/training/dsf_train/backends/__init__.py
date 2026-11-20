"""后端适配器：训练框架的接入点（docs/training.md §11）。

适配器是模块，提供：
  NAME                                     model.yaml 的 train.backend
  TASKS                                    支持的任务
  write_view(task, manifest, root, view, location)
                                           把数据集写成该后端需要的目录格式；location 是视图的最终位置
  data_path(task, view)                    训练与评估时交给后端的 data（视图中的文件或目录）
  check_options(options)                   train.options 的问题（框架决定的参数不能出现）
  version()                                后端及其依赖的版本，记入 run.json
  train(task, data, weights, options, directory, resume, workdir)
                                           训练到 directory，返回最佳权重；workdir 是下载与缓存的位置
  evaluate(task, weights, data, split, options, directory, workdir)
                                           在一个划分上评估，返回 {"metrics": {...}, "classes": {类别号: {...}}}
  predict(task, weights, images, options, conf, workdir, *, agnostic)
                                           原生推理（预标注用），逐张产出 [{"class", "box", "score"}]，
                                           box 是像素坐标 [x1, y1, x2, y2]，顺序与 images 相同；agnostic 为真时
                                           NMS 不区分类别，重叠的框只保留得分最高的一个，不论类别是否相同
  LAYOUTS                                  任务 → 导出时能追加转换的部署输出布局
  export(task, weights, layout, size, settings, target, workdir)
                                           导出 ONNX 到 target，输出已是 layout；size 是 (宽, 高)，settings 是
                                           model.yaml 的 export
  reference(task, weights, layout, batch)  原生推理在 layout 下的输出（与导出的 ONNX 比较一致性）：
                                           batch 是 float32 [n, 3, 高, 宽]，返回 numpy 数组
  BEST、LAST                               训练目录中最佳与最近检查点的相对路径（续训从 LAST 开始）
"""

from __future__ import annotations

from types import ModuleType

from ..errors import TrainError
from . import ultralytics

BACKENDS: dict[str, ModuleType] = {ultralytics.NAME: ultralytics}


def get(name: object, task: str) -> ModuleType:
    backend = BACKENDS.get(name)  # type: ignore[arg-type]
    if backend is None:
        raise TrainError(f"unknown train.backend {name!r}; available: {', '.join(sorted(BACKENDS))}")
    if task not in backend.TASKS:
        raise TrainError(f"backend {name} does not support task {task}")
    return backend
