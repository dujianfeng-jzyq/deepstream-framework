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
# 支持的部署输出布局（由解析插件定义，插件在项目中提供）：唯一输出 [batch, 候选数, 6]，
# 每行 x1, y1, x2, y2, confidence, class_id，坐标为网络输入尺寸下的像素。
LAYOUTS = ("xyxy-confidence-class_id",)
# 导出一致性：原生推理与 ONNX 对同一输入的逐行差异上限。只比较置信度不低于 COMPARE_CONF 的候选
# （低分候选在现场被阈值丢弃）；类别号只在置信度不低于 CLASS_CONF 时比较，低分候选的最高类别分数常常相近。
CONSISTENCY = {"compare_conf": 0.05, "class_conf": 0.25, "max_box_diff": 0.5, "max_conf_diff": 0.001}
# 统计预标注的修正量时，同类别且 IoU 不低于它的人工框视为保留了这个预测。
KEPT_IOU = 0.5


def samples(annotation: Annotation, spec: dict, ignore: set[str]) -> tuple[list[dict], list[str], Counter]:
    """返回 (对象, 问题, 不收入的形状按原因计数)。对象的 box 是像素坐标 [x1, y1, x2, y2]。
    ignore 中的标签丢弃（来源中本模型不用的类别与 data.ignore_classes），其余不在 classes 中的标签是错误。"""
    classes = {name: index for index, name in enumerate(spec["classes"])}
    objects, problems, ignored = [], [], Counter()
    for shape in annotation.shapes:
        if shape.shape_type != SHAPE:
            ignored[f"{shape.shape_type or '(none)'} shapes (not used by task {NAME})"] += 1  # 同一来源可能还服务于分割等任务
            continue
        if shape.label in ignore:
            continue
        if shape.label not in classes:
            problems.append(f"unknown class {shape.label!r}")
            continue
        box, problem = rectangle(shape, annotation)
        if problem:
            problems.append(problem)
            continue
        objects.append({"class": classes[shape.label], "box": box,
                        "difficult": shape.difficult, "attributes": dict(sorted(shape.attributes.items()))})
    return objects, problems, ignored


def rectangle(shape, annotation: Annotation) -> tuple[list[float], str]:
    """矩形形状 → (像素坐标 [x1, y1, x2, y2]，问题)。容差内的越界裁剪到图像内；有问题时框为空。"""
    if len(shape.points) < 2:
        return [], f"{shape.label}: a rectangle needs at least two points"
    xs, ys = [p[0] for p in shape.points], [p[1] for p in shape.points]
    box = [min(xs), min(ys), max(xs), max(ys)]
    limits = (annotation.width, annotation.height)
    if min(box) < -EDGE_TOLERANCE or box[2] > limits[0] + EDGE_TOLERANCE or box[3] > limits[1] + EDGE_TOLERANCE:
        return [], f"{shape.label}: box {[round(v, 1) for v in box]} is outside the {limits[0]}x{limits[1]} image"
    box = [min(max(box[0], 0.0), limits[0]), min(max(box[1], 0.0), limits[1]),
           min(max(box[2], 0.0), limits[0]), min(max(box[3], 0.0), limits[1])]
    if box[2] - box[0] < 1 or box[3] - box[1] < 1:
        return [], f"{shape.label}: box {[round(v, 1) for v in box]} is smaller than one pixel"
    return [round(v, 2) for v in box], ""


def manifest_fields(spec: dict) -> dict:
    """数据集清单中除样本外决定训练结果的规格字段：检测的样本是整图，没有额外字段。"""
    return {}


def source_labels(spec: dict) -> list[str]:
    """来源中必须存在的标签（类别依赖，§15.1）：检测模型的类别就是标注中的类别名。"""
    return spec["classes"]


def consistency_inputs(root, samples: list[dict], spec: dict, limit: int) -> list:
    """导出一致性比较的输入：前 limit 张图片按网络输入尺寸 letterbox。"""
    from .. import media, paths
    width, height = spec["input"]["width"], spec["input"]["height"]
    return [media.letterbox(paths.source_dir(root, s["source"]) / s["image"], width, height) for s in samples[:limit]]


def statistics(samples_: list[dict], spec: dict) -> Counter:
    """类别名 → 实例数。"""
    return Counter(spec["classes"][o["class"]] for sample in samples_ for o in sample["objects"])


def prelabel_shapes(predictions: list[dict], width: int, height: int, classes: list[str]) -> list[dict]:
    """后端的预测 → X-AnyLabeling 矩形，类别号按预标注模型的 classes 换成名称。框裁剪到图像内，坐标保留两位小数。"""
    shapes = []
    for item in predictions:
        x1, y1, x2, y2 = item["box"]
        box = [round(min(max(x1, 0.0), width), 2), round(min(max(y1, 0.0), height), 2),
               round(min(max(x2, 0.0), width), 2), round(min(max(y2, 0.0), height), 2)]
        shapes.append(xanylabeling.new_shape(classes[item["class"]], SHAPE,
                                             xanylabeling.rectangle_points(box)))
    return shapes


def added_shapes(shapes: list[dict], annotation: Annotation, label: str) -> list[dict]:
    """按类别预标注（§15.4）：标注中已有同类别框（IoU 不低于 KEPT_IOU）的候选不写，重复运行不会叠加。"""
    existing = [_box(s.points) for s in annotation.shapes
                if s.shape_type == SHAPE and s.label == label and len(s.points) >= 2]
    added = []
    for shape in shapes:
        box = _box(shape["points"])
        if all(_iou(box, other) < KEPT_IOU for other in existing):
            added.append(shape)
            existing.append(box)
    return added


def corrections(predicted: list[dict], annotation: Annotation, label: str | None = None) -> Counter:
    """预标注写入的形状与人确认后的标注相比：kept（保留，可能微调了位置）、removed（删除或改了类别）、
    added（人补上的框）。按 IoU 从高到低贪心配对。给出 label 时只看这个类别的框：按类别预标注只写了它。"""
    final = [(shape.label, _box(shape.points)) for shape in annotation.shapes
             if shape.shape_type == SHAPE and len(shape.points) >= 2 and label in (None, shape.label)]
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


def labels_text(classes: list[str]) -> str:
    """nvinfer 检测网络的 labels.txt：每行一个类别名，行号即类别号。"""
    return "".join(f"{name}\n" for name in classes)


def compare(layout: str, native, exported) -> dict:
    """比较原生推理与导出 ONNX 的输出（同一批输入，形状 [batch, 候选数, 6]），返回差异摘要，passed 为是否一致。
    候选按行对齐：同一个输入位置在两边是同一行。"""
    import numpy
    if layout not in LAYOUTS:
        raise ValueError(f"task {NAME} has no comparison for layout {layout}")
    limits = CONSISTENCY
    if native.shape != exported.shape:
        return {"passed": False, "reason": f"output shapes differ: {list(native.shape)} vs {list(exported.shape)}",
                "limits": limits}
    native, exported = native.reshape(-1, 6).astype(numpy.float64), exported.reshape(-1, 6).astype(numpy.float64)
    rows = numpy.maximum(native[:, 4], exported[:, 4]) >= limits["compare_conf"]
    confident = rows & (native[:, 4] >= limits["class_conf"])
    box = float(numpy.abs(native[rows, :4] - exported[rows, :4]).max()) if rows.any() else 0.0
    conf = float(numpy.abs(native[rows, 4] - exported[rows, 4]).max()) if rows.any() else 0.0
    classes = int((native[confident, 5] != exported[confident, 5]).sum())
    finite = bool(numpy.isfinite(exported).all())
    passed = finite and box <= limits["max_box_diff"] and conf <= limits["max_conf_diff"] and classes == 0
    return {"passed": passed, "candidates": int(rows.sum()), "confident": int(confident.sum()),
            "max_box_diff": box, "max_conf_diff": conf, "class_mismatches": classes, "finite": finite,
            "limits": limits}
