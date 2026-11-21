"""来源中给 X-AnyLabeling 的文件：每个检测模型一对（docs/training.md §5.2）。

  sources/<来源>/<模型>.classes.txt       模型的类别，载入为预设类别
  sources/<来源>/<模型>.attributes.json   形状属性文件：模型的二级模型的属性，以及标注中已有的属性

二级模型没有自己的文件：它的类别是父模型框上的属性值，标注时载入父模型的一对文件，一次看全全部属性。
X-AnyLabeling 同一时间只载入一个属性文件，文件中没有的属性在工具中看不到、可能在保存时被覆盖，所以
二级模型的属性都合在父模型的文件中，而不是每个二级模型一份。

哪些模型有文件：data.sources 中有这个来源的检测模型，以及 data.sources 中有它的二级模型的父模型。文件由工具
生成、不手写，也没有单独的命令：train update、import、prelabel 与 prepare 顺带重新生成（输入经过工具的地方），
不会因为忘了刷新而过时；不再有模型使用的文件随之删除。属性文件的内容：
- 二级模型声明的属性（data.from_parent.attribute）在它的 cascade.classes 上，取值是 classes 加
  data.ignore_classes；只计入 data.sources 中有这个来源的二级模型。
- 标注中已有的、父模型类别框上的属性全部保留（还没有模型使用的属性也是标注的一部分）。
- 同一来源中同名的属性是同一个属性，取值取它在各类别上的并集（类别名也是这样跨模型共用的）。

格式（X-AnyLabeling 的 shape-level 属性文件）：{类别: {属性: [取值…]}, "__widget_types__": {类别: {属性: 控件}}}。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from . import paths, sources, spec as specs, xanylabeling
from .errors import TrainError

CLASSES_SUFFIX = ".classes.txt"
ATTRIBUTES_SUFFIX = ".attributes.json"
# 旧版按来源生成的文件：刷新时删除（labels.txt 的类别先记入 source.json）。
LEGACY = ("labels.txt", "attributes.json")
WIDGETS = "__widget_types__"
# 取值不多时用单选按钮（一眼看全、一次点击），多了用下拉框（X-AnyLabeling 的默认控件）。
MAX_RADIO = 6


@dataclass
class Pair:
    model: str
    classes_path: Path
    attributes_path: Path
    document: dict                                   # 属性文件的内容
    defined_by: dict[str, list[str]]                 # 属性 → 声明它的二级模型
    unexpected: dict[tuple[str, str], int] = field(default_factory=dict)  # (属性, 取值) → 模型不接受的框数


@dataclass
class Result:
    source: str
    pairs: list[Pair] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)   # 规格不成立、未计入的模型 → 原因


def files(directory: Path) -> list[Path]:
    """来源目录中生成的文件（复核集带上它们）。"""
    return sorted([*directory.glob(f"*{CLASSES_SUFFIX}"), *directory.glob(f"*{ATTRIBUTES_SUFFIX}")])


def _specs(repo: Path) -> tuple[dict[str, dict], dict[str, str]]:
    """逐个读取规格：这些文件是其他命令的附带结果，一个暂时不成立的规格（例如父类别改名后、cascade.classes
    还没改写的二级模型）不能让那个命令失败，只是这次不计入它。"""
    from dsf_deploy import layout
    models_root = repo / layout.MODELS
    loaded, skipped = {}, {}
    ids = sorted(p.name for p in models_root.iterdir() if (p / "model.yaml").is_file()) if models_root.is_dir() else []
    for model_id in ids:
        try:
            loaded[model_id] = specs.load(repo, model_id)
        except TrainError as error:
            skipped[model_id] = str(error).splitlines()[-1].strip()
    return loaded, skipped


def owners(models: dict[str, dict], source: str) -> list[str]:
    """在这个来源中有文件的检测模型。"""
    found = set()
    for model_id, spec in models.items():
        if source not in spec["data"]["sources"]:
            continue
        parent = (spec.get("cascade") or {}).get("parent")
        if parent is None:
            found.add(model_id)
        elif parent in models and not models[parent].get("cascade"):
            found.add(parent)
    return sorted(found)


def _migrate(directory: Path) -> None:
    """旧版来源：类别只在 labels.txt 中时先记入 source.json，再删除按来源生成的旧文件。"""
    record = sources.read_record(directory)
    if record.get("classes") is None:
        record["classes"] = sources.class_list(directory, record)
        paths.write_json(directory / paths.SOURCE_RECORD, record)
    for name in LEGACY:
        (directory / name).unlink(missing_ok=True)


def generate(repo: Path, root: Path, source: str) -> Result:
    directory = paths.source_dir(root, source)
    _migrate(directory)
    record = sources.read_record(directory)
    models, skipped = _specs(repo)
    result = Result(source, skipped=skipped)

    # 同一来源中同名属性是同一个属性：声明它的二级模型（不论父模型）必须给出相同的取值。
    allowed: dict[str, list[str]] = {}               # 属性 → 模型声明的取值（有序）
    defined_by: dict[str, list[str]] = defaultdict(list)
    declared: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))  # 父模型 → 类别 → 属性
    for model_id, spec in sorted(models.items()):
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
            declared[spec["cascade"]["parent"]][label].add(attribute)

    observed: dict[str, list[str]] = defaultdict(list)   # 属性 → 标注中出现的取值（按首次出现）
    carried: dict[str, set[str]] = defaultdict(set)      # 类别 → 标注中它的框上出现的属性
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
                carried[shape.label].add(attribute)
                if value not in observed[attribute]:
                    observed[attribute].append(value)
                counts[(attribute, value)] += 1
    vocabulary = {}
    for attribute in sorted({*allowed, *observed}):
        known = allowed.get(attribute, [])
        vocabulary[attribute] = known + sorted(v for v in observed[attribute] if v not in known)

    wanted = owners(models, source)
    for model_id in wanted:
        classes = models[model_id]["classes"]
        document, widgets = {}, {}
        for label in classes:  # 按模型类别的顺序
            names = sorted(declared[model_id].get(label, set()) | carried.get(label, set()))
            if names:
                document[label] = {a: vocabulary[a] for a in names}
                widgets[label] = {a: "radiobutton" if len(vocabulary[a]) <= MAX_RADIO else "combobox" for a in names}
        document[WIDGETS] = widgets
        mine = {a: m for a, m in defined_by.items() if any(models[c]["cascade"]["parent"] == model_id for c in m)}
        unexpected = {(a, v): counts[(a, v)] for a in mine for v in observed[a] if v not in allowed[a]}
        classes_path = directory / f"{model_id}{CLASSES_SUFFIX}"
        classes_path.write_text("".join(f"{c}\n" for c in classes), encoding="utf-8")
        attributes_path = directory / f"{model_id}{ATTRIBUTES_SUFFIX}"
        paths.write_json(attributes_path, document)
        result.pairs.append(Pair(model_id, classes_path, attributes_path, document, mine, unexpected))
    keep = {directory / f"{m}{suffix}" for m in wanted for suffix in (CLASSES_SUFFIX, ATTRIBUTES_SUFFIX)}
    for path in files(directory):
        if path not in keep:
            path.unlink()
    return result


def refresh(repo: Path, root: Path, source_names) -> list[str]:
    """重新生成这些来源的文件，返回给人看的简短摘要（附在各命令的输出后）。"""
    lines, hint = [], False
    for source in dict.fromkeys(source_names):
        if not (paths.source_dir(root, source) / paths.SOURCE_RECORD).is_file():
            lines.append(f"  note: source {source} is in data.sources but not imported yet")  # prepare 报错
            continue
        result = generate(repo, root, source)
        if not result.pairs:
            lines.append(f"  note: no model uses source {source} yet: train update <model> source add {source}")
        for pair in result.pairs:
            labels = [label for label in pair.document if label != WIDGETS]
            lines.append(f"  labeling {source} for {pair.model}: {pair.classes_path} and {pair.attributes_path.name} "
                         f"({len(labels)} classes with attributes)")
            hint = True
            for (attribute, value), count in sorted(pair.unexpected.items()):
                lines.append(f"  warning: {count} boxes of {source} have {attribute}={value}, which "
                             f"{', '.join(pair.defined_by[attribute])} does not accept: correct them, or add the "
                             "value to classes or data.ignore_classes")
        for model_id, reason in sorted(result.skipped.items()):
            lines.append(f"  note: {model_id} is left out of the labeling files of {source} until its model.yaml "
                         f"is valid again ({reason})")
    if hint:
        lines.append("  in X-AnyLabeling: load the .classes.txt as the predefined labels and the .attributes.json "
                     "via Image Classifier (Ctrl+3) → MultiTask → the upload icon beside the object selector")
    return lines

