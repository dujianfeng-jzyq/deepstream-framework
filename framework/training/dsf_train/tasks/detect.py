"""一级检测：矩形框。"""

from __future__ import annotations

from collections import Counter

from ..xanylabeling import Annotation

NAME = "detect"
SHAPE = "rectangle"
# 标注工具在图像边缘取点时可能略微越界：容差内裁剪到图像内，超出容差视为标错。
EDGE_TOLERANCE = 2.0
# 测试划分上的指标：名称 → 比较方向。evaluate.gate 只能引用这些名称；每类指标是前两项与精确率、召回率。
METRICS = {"map50_95": "higher", "map50": "higher", "precision": "higher", "recall": "higher"}


def samples(annotation: Annotation, spec: dict) -> tuple[list[dict], list[str], Counter]:
    """返回 (对象, 问题, 被忽略的其他形状类型计数)。对象的 box 是像素坐标 [x1, y1, x2, y2]。"""
    classes = {name: index for index, name in enumerate(spec["classes"])}
    ignore = set(spec["data"].get("ignore_classes") or [])
    objects, problems, ignored = [], [], Counter()
    for shape in annotation.shapes:
        if shape.shape_type != SHAPE:
            ignored[shape.shape_type or "(none)"] += 1  # 同一来源可能还服务于分割等任务
            continue
        if shape.label in ignore:
            continue
        if shape.label not in classes:
            problems.append(f"unknown class {shape.label!r}")
            continue
        if len(shape.points) < 2:
            problems.append(f"{shape.label}: a rectangle needs at least two points")
            continue
        xs, ys = [p[0] for p in shape.points], [p[1] for p in shape.points]
        box = [min(xs), min(ys), max(xs), max(ys)]
        limits = (annotation.width, annotation.height)
        if min(box) < -EDGE_TOLERANCE or box[2] > limits[0] + EDGE_TOLERANCE or box[3] > limits[1] + EDGE_TOLERANCE:
            problems.append(f"{shape.label}: box {[round(v, 1) for v in box]} is outside the "
                            f"{limits[0]}x{limits[1]} image")
            continue
        box = [min(max(box[0], 0.0), limits[0]), min(max(box[1], 0.0), limits[1]),
               min(max(box[2], 0.0), limits[0]), min(max(box[3], 0.0), limits[1])]
        if box[2] - box[0] < 1 or box[3] - box[1] < 1:
            problems.append(f"{shape.label}: box {[round(v, 1) for v in box]} is smaller than one pixel")
            continue
        objects.append({"class": classes[shape.label], "box": [round(v, 2) for v in box],
                        "difficult": shape.difficult, "attributes": dict(sorted(shape.attributes.items()))})
    return objects, problems, ignored


def statistics(samples_: list[dict], spec: dict) -> Counter:
    """类别名 → 实例数。"""
    return Counter(spec["classes"][o["class"]] for sample in samples_ for o in sample["objects"])
