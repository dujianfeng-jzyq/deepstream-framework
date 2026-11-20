"""Ultralytics 的训练器类。多卡训练时 Ultralytics 在 torch.distributed 子进程中按模块路径导入训练器
（ultralytics.utils.dist.generate_ddp_file）：子进程不经过 ultralytics.train 中的设置，导入这个模块时
在子进程中设置图表的中文字体。只在训练镜像中导入。

分类训练器按各类召回率的平均（宏平均 recall）选择 best.pt 并判断早停，而不是 Ultralytics 默认的
(top1 + top5) / 2：二分类时 top5 恒为 1，默认的 fitness 就是 accuracy，类别极不均衡时（“遮挡=是”只占
百分之几）accuracy 几乎不随少数类的召回变化，会保留错误的检查点、过早停止。宏平均 recall 与评估和发布
门槛中的 recall（tasks/classify.py）是同一个量。

model.yaml 的 train.class_weights 给出时，分类损失按类别加权（少数类的错分代价更高）。"""

import json
import os

import torch
import torch.nn.functional as F
from ultralytics.models.yolo.classify import ClassificationTrainer as _ClassificationTrainer
from ultralytics.models.yolo.detect import DetectionTrainer as _DetectionTrainer
from ultralytics.utils.metrics import ClassifyMetrics as _ClassifyMetrics

from .ultralytics import use_cjk_font

use_cjk_font()

RECALL_KEY = "metrics/recall_mean"
# 分类损失的类别权重（JSON 列表，按类别号），由 backends/ultralytics.train 设置。
CLASS_WEIGHTS_ENV = "DSF_CLASS_WEIGHTS"


class DetectionTrainer(_DetectionTrainer):
    """与 Ultralytics 的检测训练器相同；存在的意义是让子进程导入本模块。"""


def mean_recall(top1: torch.Tensor, targets: torch.Tensor) -> float:
    """各类召回率的平均，只算验证划分中出现的类别。top1 与 targets 是同长度的类别号。"""
    recalls = [float((top1[targets == c] == c).float().mean()) for c in targets.unique().tolist()]
    return sum(recalls) / len(recalls) if recalls else 0.0


class BalancedClassifyMetrics(_ClassifyMetrics):
    """在 top1、top5 之外计算宏平均 recall，并以它为 fitness（训练器据此保存 best.pt 与早停）。"""

    def __init__(self) -> None:
        super().__init__()
        self.recall = 0.0

    def process(self, targets, pred):
        super().process(targets, pred)
        self.recall = mean_recall(torch.cat(pred)[:, 0], torch.cat(targets))

    @property
    def fitness(self):
        return self.recall

    # keys 不变：验证器按 keys 的个数拼打印格式，却只传入 top1 与 top5。recall 只加在 results_dict 中，
    # 训练器从这里取 fitness 并写入 results.csv。
    @property
    def results_dict(self):
        return {**super().results_dict, RECALL_KEY: self.recall}


class WeightedClassificationLoss:
    """与 Ultralytics 的 v8ClassificationLoss 相同，交叉熵按类别加权。"""

    def __init__(self, weights: list[float]) -> None:
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def __call__(self, preds, batch):
        weight = self.weights.to(device=preds.device, dtype=preds.dtype)
        loss = F.cross_entropy(preds, batch["cls"], weight=weight, reduction="mean")
        return loss, loss.detach()


class ClassificationTrainer(_ClassificationTrainer):
    """训练中的验证改用 BalancedClassifyMetrics：fitness 为宏平均 recall。给出类别权重时损失按类别加权。"""

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg, weights, verbose)
        if os.environ.get(CLASS_WEIGHTS_ENV):
            class_weights = json.loads(os.environ[CLASS_WEIGHTS_ENV])
            if len(class_weights) != self.data["nc"]:
                raise ValueError(f"{len(class_weights)} class weights for {self.data['nc']} classes")
            model.criterion = WeightedClassificationLoss(class_weights)
        return model

    def get_validator(self):
        validator = super().get_validator()
        validator.metrics = BalancedClassifyMetrics()
        return validator


TRAINERS = {"detect": DetectionTrainer, "classify": ClassificationTrainer}
