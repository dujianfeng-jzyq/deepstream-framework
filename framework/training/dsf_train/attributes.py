"""来源的 X-AnyLabeling 属性文件 sources/<来源>/attributes.json（docs/training.md §5.2）。

属性文件让标注工具对每个类别的框给出可选的属性与取值；没有它，工具里无法标二级模型的标签。它和
labels.txt 一样由工具生成、放在来源目录中，不手写，也没有单独的命令：import、prepare 与改写来源的
update 顺带重新生成（输入经过工具的地方），不会因为忘了刷新而过时。内容：
- 二级模型（model.yaml 的 cascade）声明了它用的属性：data.from_parent.attribute 出现在 cascade.classes 的
  每个类别上，取值是它的 classes 加上 data.ignore_classes。
- 来源的标注中已有的属性全部保留（还没有模型使用的属性也是标注的一部分）：文件中缺少的属性在工具中
  看不到，可能被覆盖。
- 同一来源中同名的属性是同一个属性，取值取它在各类别上的并集（类别名也是这样跨模型共用的）。

格式（X-AnyLabeling 的 shape-level 属性文件）：{类别: {属性: [取值…]}, "__widget_types__": {类别: {属性: 控件}}}。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from . import paths, sources, spec as specs, xanylabeling
from .errors import TrainError

FILE = "attributes.json"
WIDGETS = "__widget_types__"
# 取值不多时用单选按钮（一眼看全、一次点击），多了用下拉框（X-AnyLabeling 的默认控件）。
MAX_RADIO = 6


@dataclass
class Result:
    path: Path
    document: dict
    defined_by: dict[str, list[str]]             # 属性 → 声明它的二级模型
    unexpected: dict[tuple[str, str], int] = field(default_factory=dict)  # (属性, 取值) → 模型不接受的框数
    skipped: dict[str, str] = field(default_factory=dict)  # 规格不成立、未计入的模型 → 原因


def generate(repo: Path, root: Path, source: str) -> Result:
    directory = paths.source_dir(root, source)
    record = sources.read_record(directory)
    labels = sources.class_list(directory, record)
    allowed: dict[str, list[str]] = {}              # 属性 → 模型声明的取值（有序）
    defined_by: dict[str, list[str]] = defaultdict(list)
    carriers: dict[str, set[str]] = defaultdict(set)  # 类别 → 属性
    skipped = {}
    # 逐个读取规格：属性文件是其他命令的附带结果，一个暂时不成立的规格（例如父类别改名后、cascade.classes
    # 还没改写的二级模型）不能让那个命令失败，只是这次不计入它声明的属性。
    from dsf_deploy import layout
    models_root = repo / layout.MODELS
    model_ids = sorted(p.name for p in models_root.iterdir() if (p / "model.yaml").is_file()) \
        if models_root.is_dir() else []
    for model_id in model_ids:
        try:
            spec = specs.load(repo, model_id)
        except TrainError as error:
            skipped[model_id] = str(error).splitlines()[-1].strip()
            continue
        if not spec.get("cascade") or source not in spec["data"]["sources"]:
            continue
        attribute = spec["data"]["from_parent"]["attribute"]
        values = [*spec["classes"], *(spec["data"].get("ignore_classes") or [])]
        if attribute in allowed and set(allowed[attribute]) != set(values):
            raise TrainError(f"models {', '.join(defined_by[attribute])} and {model_id} give different values for "
                             f"attribute {attribute} ({allowed[attribute]} vs {values}): one attribute has one set "
                             "of values in a source")
        allowed.setdefault(attribute, values)
        defined_by[attribute].append(model_id)
        for label in spec["cascade"]["classes"]:
            carriers[label].add(attribute)

    observed: dict[str, list[str]] = defaultdict(list)   # 属性 → 标注中出现的取值（按首次出现）
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for name in sorted(record["images"]):
        path = xanylabeling.annotation_path(directory / name)
        if not path.is_file():
            continue
        for shape in xanylabeling.read(path).shapes:
            for attribute, value in shape.attributes.items():
                value = str(value)
                if value == "":
                    continue
                carriers[shape.label].add(attribute)
                if value not in observed[attribute]:
                    observed[attribute].append(value)
                counts[(attribute, value)] += 1

    vocabulary = {}
    for attribute in sorted({*allowed, *observed}):
        declared = allowed.get(attribute, [])
        vocabulary[attribute] = declared + sorted(v for v in observed[attribute] if v not in declared)
    document, widgets = {}, {}
    for label in labels:  # 按 labels.txt 的顺序；不在其中的标签由 prepare 报错，这里不为它生成
        attributes = sorted(carriers.get(label, ()))
        if attributes:
            document[label] = {a: vocabulary[a] for a in attributes}
            widgets[label] = {a: "radiobutton" if len(vocabulary[a]) <= MAX_RADIO else "combobox" for a in attributes}
    document[WIDGETS] = widgets
    unexpected = {(a, v): counts[(a, v)] for a, values in allowed.items() for v in observed[a] if v not in values}
    target = directory / FILE
    paths.write_json(target, document)
    return Result(target, document, dict(defined_by), unexpected, skipped)


def refresh(repo: Path, root: Path, source_names) -> list[str]:
    """重新生成这些来源的属性文件，返回给人看的简短摘要（附在 import、prepare、update 的输出后）。"""
    lines = []
    for source in dict.fromkeys(source_names):
        result = generate(repo, root, source)
        labels = [label for label in result.document if label != WIDGETS]
        if labels:
            lines.append(f"  attributes: {result.path} ({len(labels)} classes; load it in X-AnyLabeling: Image "
                         "Classifier (Ctrl+3) → MultiTask → the upload icon beside the object selector)")
        for (attribute, value), count in sorted(result.unexpected.items()):
            lines.append(f"  warning: {count} boxes of {source} have {attribute}={value}, which "
                         f"{', '.join(result.defined_by[attribute])} does not accept: correct them, or add the value "
                         "to classes or data.ignore_classes")
        for model_id, reason in sorted(result.skipped.items()):
            lines.append(f"  note: the attributes of {model_id} are not in {result.path.name} until its model.yaml "
                         f"is valid again ({reason})")
    return lines
