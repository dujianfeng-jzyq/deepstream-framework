"""Ultralytics 的训练器类。多卡训练时 Ultralytics 在 torch.distributed 子进程中按模块路径导入训练器
（ultralytics.utils.dist.generate_ddp_file）：子进程不经过 ultralytics.train 中的设置，导入这个模块时
在子进程中设置图表的中文字体。只在训练镜像中导入。"""

from ultralytics.models.yolo.classify import ClassificationTrainer as _ClassificationTrainer
from ultralytics.models.yolo.detect import DetectionTrainer as _DetectionTrainer

from .ultralytics import use_cjk_font

use_cjk_font()


class DetectionTrainer(_DetectionTrainer):
    """与 Ultralytics 的检测训练器相同；存在的意义是让子进程导入本模块。"""


class ClassificationTrainer(_ClassificationTrainer):
    """与 Ultralytics 的分类训练器相同；存在的意义同上。"""


TRAINERS = {"detect": DetectionTrainer, "classify": ClassificationTrainer}
