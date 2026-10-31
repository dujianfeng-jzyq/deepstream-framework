"""后端适配器：训练框架的接入点（docs/training.md §11）。

适配器是模块，提供：
  NAME                                     model.yaml 的 train.backend
  TASKS                                    支持的任务
  write_view(task, manifest, root, view, location)
                                           把数据集写成该后端需要的目录格式；location 是视图的最终位置
训练、评估、原生推理与导出在后续阶段加入。
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
