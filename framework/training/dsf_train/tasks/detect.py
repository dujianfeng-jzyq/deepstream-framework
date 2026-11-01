"""一级检测：矩形框。"""

from __future__ import annotations

from collections import Counter

from .. import xanylabeling
from ..xanylabeling import Annotation

NAME = "detect"
SHAPE = "rectangle"
# 标注工具在图像边缘取点时可能略微越界：容差内裁剪到图像内，超出容差视为标错。
EDGE_TOLERANCE = 2.0
# 测试划分上的指标：名称 → 比较方向。evaluate.gate 只能引用这些名称；每类指标是前两项与精确率、召回率。
METRICS = {"map50_95": "higher", "map50": "higher", "precision": "higher", "recall": "higher"}
# 统计预标注的修正量时，同类别且 IoU 不低于它的人工框视为保留了这个预测。
KEPT_IOU = 0.5


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


def prelabel_shapes(predictions: list[dict], width: int, height: int, spec: dict) -> list[dict]:
    """后端的预测 → X-AnyLabeling 矩形。框裁剪到图像内，坐标保留两位小数。"""
    shapes = []
    for item in predictions:
        x1, y1, x2, y2 = item["box"]
        box = [round(min(max(x1, 0.0), width), 2), round(min(max(y1, 0.0), height), 2),
               round(min(max(x2, 0.0), width), 2), round(min(max(y2, 0.0), height), 2)]
        shapes.append(xanylabeling.new_shape(spec["classes"][item["class"]], SHAPE,
                                             xanylabeling.rectangle_points(box)))
    return shapes


def corrections(predicted: list[dict], annotation: Annotation) -> Counter:
    """预标注写入的形状与人确认后的标注相比：kept（保留，可能微调了位置）、removed（删除或改了类别）、
    added（人补上的框）。按 IoU 从高到低贪心配对。"""
    final = [(shape.label, _box(shape.points)) for shape in annotation.shapes
             if shape.shape_type == SHAPE and len(shape.points) >= 2]
    proposed = [(shape["label"], _box(shape["points"])) for shape in predicted]
    pairs = sorted(((_iou(a[1], b[1]), i, j) for i, a in enumerate(proposed) for j, b in enumerate(final)
                    if a[0] == b[0]), reverse=True)
    used_p, used_f = set(), set()
    for iou, i, j in pairs:
        if iou < KEPT_IOU:
            break
        if i not in used_p and j not in used_f:
            used_p.add(i)
            used_f.add(j)
    return Counter(kept=len(used_p), removed=len(proposed) - len(used_p), added=len(final) - len(used_f))


def _box(points) -> list[float]:
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def _iou(a: list[float], b: list[float]) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
