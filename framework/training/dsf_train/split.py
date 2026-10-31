"""划分：每个采集组按自己的哈希落进训练、验证或测试，与其他样本无关（docs/training.md §6）。

加入新来源时已有组的划分不变，测试集只增加新组：旧模型从未训练过的测试样本一直留在测试集中，
发布时新旧模型的比较才有意义。代价是类别在各划分间的比例只能近似，由数据集报告给出。
"""

from __future__ import annotations

import hashlib

SPLITS = ("train", "val", "test")


def assign(group: str, seed: int, val: float, test: float) -> str:
    digest = hashlib.sha256(f"{seed}\0{group}".encode("utf-8")).digest()
    position = int.from_bytes(digest[:8], "big") / 2 ** 64
    if position < test:
        return "test"
    return "val" if position < test + val else "train"
