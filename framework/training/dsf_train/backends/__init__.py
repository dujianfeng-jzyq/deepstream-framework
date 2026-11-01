"""后端适配器：训练框架的接入点（docs/training.md §11）。

适配器是模块，提供：
  NAME                                     model.yaml 的 train.backend
  TASKS                                    支持的任务
  write_view(task, manifest, root, view, location)
                                           把数据集写成该后端需要的目录格式；location 是视图的最终位置
  check_options(options)                   train.options 的问题（框架决定的参数不能出现）
  version()                                后端及其依赖的版本，记入 run.json
  train(task, data, weights, options, directory, resume, workdir)
                                           训练到 directory，返回最佳权重；workdir 是下载与缓存的位置
  evaluate(task, weights, data, split, options, directory, workdir)
                                           在一个划分上评估，返回 {"metrics": {...}, "classes": {类别号: {...}}}
  predict(task, weights, images, options, conf, workdir)
                                           原生推理（预标注用），逐张产出 [{"class", "box", "score"}]，
                                           box 是像素坐标 [x1, y1, x2, y2]，顺序与 images 相同
  BEST、LAST                               训练目录中最佳与最近检查点的相对路径（续训从 LAST 开始）
导出在后续阶段加入。
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
