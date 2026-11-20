"""二级分类：一级模型检测到的目标再分类（docs/training.md §12）。

样本从父模型来源中父类别的矩形框裁出，类别是框上一个属性的值（data.from_parent.attribute）。一张图片是
manifest 中的一个样本，它的每个对象是一个裁剪；划分沿用图片的采集组，两级模型的测试集来自同一批图片。
不支持整图分类：现在没有需要它的模型，需要时再增加样本的另一种来源。
"""

from __future__ import annotations

from collections import Counter

from ..xanylabeling import Annotation
from . import detect

NAME = "classify"
SHAPE = detect.SHAPE
# 测试划分上的指标：accuracy 是全部裁剪的 top-1 准确率；recall 与 precision 是各类别的平均（宏平均）。
# 类别常常极不均衡（“遮挡=是”只占百分之几），全猜多数类也有很高的 accuracy，门槛应看 recall。
# 每类只有 recall 与 precision，accuracy 为空。
METRICS = {"accuracy": "higher", "recall": "higher", "precision": "higher"}
# 部署输出布局：唯一输出 [batch, 类别数] 的 softmax 概率（nvinfer 内置的分类解析按概率与
# classifier-threshold 比较）。后端的分类头输出 logits，导出时追加 Softmax。
LAYOUTS = ("softmax",)
# 导出一致性：原生推理与 ONNX 的概率差异上限；最高类别只在它领先第二名超过 max_prob_diff 时比较。
CONSISTENCY = {"max_prob_diff": 0.001}
# data.from_parent.min_size 的默认值：短边小于它的框不收入（现场同样太小，无法可靠分类）。
DEFAULT_MIN_SIZE = 16


def from_parent(spec: dict) -> dict:
    return spec["data"]["from_parent"]


def manifest_fields(spec: dict) -> dict:
    """裁剪拉伸到的尺寸与裁剪的来源进入数据集清单：它们变了，同样的标注得到不同的样本。"""
    return {"input": spec["input"], "cascade": spec["cascade"], "from_parent": from_parent(spec)}


def source_labels(spec: dict) -> list[str]:
    """来源中必须存在的标签：裁剪来自父模型的这些类别（cascade.classes），模型自己的类别是属性值。"""
    return spec["cascade"]["classes"]


def samples(annotation: Annotation, spec: dict, ignore: set[str]) -> tuple[list[dict], list[str], Counter]:
    """返回 (对象, 问题, 不收入的框按原因计数)。对象是父类别框的裁剪：box 为像素坐标，class 为属性值的类别号。
    ignore 是本模型不用的标签（来源中的其他类别）；data.ignore_classes 是不收入的属性值（例如“未知”）。"""
    classes = {name: index for index, name in enumerate(spec["classes"])}
    parents = set(source_labels(spec))
    settings = from_parent(spec)
    attribute, min_size = settings["attribute"], settings.get("min_size", DEFAULT_MIN_SIZE)
    ignored_values = set(spec["data"].get("ignore_classes") or [])
    objects, problems, ignored = [], [], Counter()
    for shape in annotation.shapes:
        if shape.shape_type != SHAPE:
            ignored[f"{shape.shape_type or '(none)'} shapes (not used by task {NAME})"] += 1
            continue
        if shape.label in ignore:
            continue
        if shape.label not in parents:
            problems.append(f"unknown class {shape.label!r}")
            continue
        value = shape.attributes.get(attribute)
        if value is None or value == "":
            ignored[f"boxes without attribute {attribute}"] += 1
            continue
        value = str(value)
        if value in ignored_values:
            ignored[f"boxes with {attribute}={value} (data.ignore_classes)"] += 1
            continue
        if value not in classes:
            problems.append(f"{shape.label}: {attribute}={value!r} is neither a class nor in data.ignore_classes")
            continue
        box, problem = detect.rectangle(shape, annotation)
        if problem:
            problems.append(problem)
            continue
        if min(box[2] - box[0], box[3] - box[1]) < min_size:
            ignored[f"boxes smaller than {min_size} px (data.from_parent.min_size)"] += 1
            continue
        objects.append({"class": classes[value], "box": box, "parent": shape.label, "difficult": shape.difficult})
    return objects, problems, ignored


def statistics(samples_: list[dict], spec: dict) -> Counter:
    """类别名 → 裁剪数。"""
    return Counter(spec["classes"][o["class"]] for sample in samples_ for o in sample["objects"])


def metrics(matrix: list[list[int]]) -> dict:
    """混淆矩阵（matrix[真实][预测] 的计数）→ {"metrics", "classes"}，格式与后端的 evaluate 相同。
    每类只列出测试划分中有样本的类别；宏平均也只算这些类别（没有样本的类别 recall 无从谈起）。"""
    size = len(matrix)
    total = sum(sum(row) for row in matrix)
    correct = sum(matrix[i][i] for i in range(size))
    classes = {}
    for index in range(size):
        actual = sum(matrix[index])
        if not actual:
            continue
        predicted = sum(matrix[row][index] for row in range(size))
        classes[index] = {"accuracy": None, "recall": matrix[index][index] / actual,
                          "precision": matrix[index][index] / predicted if predicted else 0.0}
    present = list(classes.values())
    return {"metrics": {"accuracy": correct / total if total else 0.0,
                        "recall": sum(c["recall"] for c in present) / len(present) if present else 0.0,
                        "precision": sum(c["precision"] for c in present) / len(present) if present else 0.0},
            "classes": classes}


def labels_text(classes: list[str]) -> str:
    """nvinfer 分类网络的 labels.txt：每个输出层一行，类别名以 ; 分隔，位置即类别号。"""
    return ";".join(classes) + "\n"


def compare(layout: str, native, exported) -> dict:
    """比较原生推理与导出 ONNX 的 softmax 输出（同一批输入，形状 [batch, 类别数]），返回差异摘要。"""
    import numpy
    if layout not in LAYOUTS:
        raise ValueError(f"task {NAME} has no comparison for layout {layout}")
    limits = CONSISTENCY
    if native.shape != exported.shape:
        return {"passed": False, "reason": f"output shapes differ: {list(native.shape)} vs {list(exported.shape)}",
                "limits": limits}
    native, exported = native.astype(numpy.float64), exported.astype(numpy.float64)
    diff = float(numpy.abs(native - exported).max()) if native.size else 0.0
    ordered = numpy.sort(native, axis=1)
    decided = (ordered[:, -1] - ordered[:, -2] > limits["max_prob_diff"]) if native.shape[1] > 1 else \
        numpy.ones(len(native), dtype=bool)
    mismatches = int((native.argmax(1) != exported.argmax(1))[decided].sum())
    finite = bool(numpy.isfinite(exported).all())
    passed = finite and diff <= limits["max_prob_diff"] and mismatches == 0
    return {"passed": passed, "crops": int(len(native)), "max_prob_diff": diff, "class_mismatches": mismatches,
            "finite": finite, "limits": limits}


def consistency_inputs(root, samples: list[dict], spec: dict, limit: int) -> list:
    """导出一致性比较的输入：按顺序取图片中的裁剪，最多 limit 个，与训练样本的生成方式相同。"""
    from .. import media, paths
    width, height = spec["input"]["width"], spec["input"]["height"]
    inputs = []
    for sample in samples:
        image = paths.source_dir(root, sample["source"]) / sample["image"]
        for item in sample["objects"][:limit - len(inputs)]:
            inputs.append(media.as_input(media.crop(image, item["box"], width, height)))
        if len(inputs) >= limit:
            break
    return inputs


def resolution(metric: str, counts: dict[str, int]) -> tuple[float, str | None]:
    """发布门槛能分辨的最小变化（见 publish.check_gate）：accuracy 按全部裁剪计，一个裁剪变化 1/总数；
    recall、precision 是各类的平均。"""
    if metric == "accuracy":
        total = sum(counts.values())
        return (1 / total if total else 0.0), None
    return detect.macro_resolution(counts)
