"""复核集：只含需要核对的图片的目录，在 X-AnyLabeling 中逐张（D 键）核对，改完再应用回来源（docs/training.md §7）。
两种来源：train eval --errors 的错分（create），与二级模型的预标注（propose，§5.4）。

reviews/<名称>/
  <来源>__<图片>          来源图像的硬链接（工具不改图像）
  <来源>__<图片名>.json   标注的副本，imagePath 改为上面的文件名
  checklist.txt          每张图要核对的框
  <模型>.classes.txt、<模型>.attributes.json  来源中给标注工具的文件（图片都来自一个来源时）
  review.json            每项的来源、图片、复制时来源标注与副本的 sha256

标注只能复制，不能链接：X-AnyLabeling 4.0.6 保存时写临时文件再 os.replace 到标注路径，链接会被替换成复核目录中
的新文件，改动永远到不了来源。所以改完要 train review apply：只写回改过的副本，且只在来源标注自复制以来没有
变化时写回，否则报告冲突、两边都不动，不会覆盖别处的修改。
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import labeling, paths, xanylabeling
from .errors import TrainError

REVIEWS = "reviews"
RECORD = "review.json"
CHECKLIST = "checklist.txt"
SCHEMA = 1


def directory(root: Path, name: str) -> Path:
    if not name or "/" in name or name.startswith("."):
        raise TrainError(f"invalid review name {name!r}")
    return root / REVIEWS / name


def create(root: Path, name: str, model: str, run: str, errors: list[dict]) -> Path:
    """为 errors（runs.errors 的条目，每项一个框）建立复核集；同一张图的多个框合为一项。"""
    grouped: dict[tuple[str, str], list[dict]] = {}
    for entry in errors:
        grouped.setdefault((entry["source"], entry["image"]), []).append(entry)
    entries = []
    for (source, image), boxes in sorted(grouped.items()):
        entries.append({"source": source, "image": image, "document": None, "proposed": False,
                        "boxes": [{k: e[k] for k in ("box", "parent", "label", "predicted", "confidence", "split")}
                                  for e in boxes],
                        "lines": [f"[{_box(e['box'])}] {e.get('parent') or ''}  labeled {e['label']}, model says "
                                  f"{e['predicted']} ({e['confidence']:.2f})  {e['split']}" for e in boxes]})
    header = (f"# review {name}: open this folder in X-AnyLabeling, step through with D, fix the labels you disagree "
              f"with\n# and save; then run: train review apply {name}")
    return _write(root, name, model, run, entries, header)


def propose(root: Path, name: str, model: str, run: str, attribute: str, proposals: list[dict]) -> Path:
    """二级模型的预标注（train prelabel）：副本中没有 attribute 取值的框已由模型填上（proposals 的每项是
    {"source", "image", "document": 填好的标注, "boxes": [{"box", "parent", "proposed", "confidence"}]}，
    proposed 为 None 的框模型没有把握、留空）。来源中的标注不变；这些副本是提议，不是修改，apply 写回
    全部副本（人删掉的副本除外），而不只是改过的。"""
    entries = []
    for item in sorted(proposals, key=lambda p: (p["source"], p["image"])):
        lines = []
        for box in item["boxes"]:
            value = (f"{attribute}={box['proposed']} proposed" if box["proposed"] is not None
                     else f"{attribute} left empty (model unsure)")
            lines.append(f"[{_box(box['box'])}] {box['parent']}  {value} ({box['confidence']:.2f})")
        entries.append({**item, "proposed": True, "lines": lines})
    header = (f"# review {name}: the model filled {attribute} on boxes that had no value. Open this folder in "
              f"X-AnyLabeling,\n# step through with D, check every listed box, correct and save. Then run: "
              f"train review apply {name}\n# It writes back every image of this review: delete the .json of "
              "images you did not check.")
    return _write(root, name, model, run, entries, header)


def _write(root: Path, name: str, model: str, run: str, entries: list[dict], header: str) -> Path:
    """写出复核集。已有同名复核集时，有未应用的修改则拒绝，否则替换。"""
    target = directory(root, name)
    if (target / RECORD).is_file():
        pending = [item["file"] for item in _items(target) if _state(target, item) == "edited"]
        if pending:
            raise TrainError(f"review {name} has {len(pending)} edited annotations that are not applied yet: "
                             f"train review apply {name} (or delete {target})")
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    items, checklist = [], []
    for entry in entries:
        source, image = entry["source"], entry["image"]
        source_image = paths.source_dir(root, source) / image
        annotation = xanylabeling.annotation_path(source_image)
        name_in_review = f"{source}__{image}"
        paths.link_or_copy(source_image, target / name_in_review)
        copy = xanylabeling.annotation_path(target / name_in_review)
        document = entry["document"] or json.loads(annotation.read_text(encoding="utf-8"))
        document = {**document, "imagePath": name_in_review}
        paths.write_json(copy, document)
        items.append({"file": name_in_review, "source": source, "image": image,
                      "source_sha256": paths.sha256(annotation), "copy_sha256": paths.sha256(copy),
                      **({"proposed": True} if entry["proposed"] else {}), "boxes": entry["boxes"]})
        checklist += [f"[ ] {name_in_review}  {line}" for line in entry["lines"]]
    # 类别与属性文件在来源中生成；图片都来自一个来源时带上它们，在复核目录中就能直接载入。
    if len({entry["source"] for entry in entries}) == 1:
        for path in labeling.files(paths.source_dir(root, entries[0]["source"])):
            shutil.copy2(path, target / path.name)
    paths.write_json(target / RECORD, {"schema": SCHEMA, "model": model, "run": run, "created_at": _now(),
                                       "items": items})
    (target / CHECKLIST).write_text(header + "\n\n" + "\n".join(checklist) + "\n", encoding="utf-8")
    return target


def _box(box: list[float]) -> str:
    return ", ".join(f"{v:.0f}" for v in box)


@dataclass
class Applied:
    written: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    unchanged: int = 0


def apply(root: Path, name: str) -> Applied:
    """把改过的副本写回来源（imagePath 恢复为来源中的文件名）。来源标注自复制以来变了的报告为冲突，不写。
    写回后更新记录：再次 apply 不做任何事。"""
    target = directory(root, name)
    record = _record(target)
    result = Applied()
    for item in record["items"]:
        state = _state(target, item)
        if state == "unchanged":
            result.unchanged += 1
            continue
        annotation = xanylabeling.annotation_path(paths.source_dir(root, item["source"]) / item["image"])
        if not annotation.is_file() or paths.sha256(annotation) != item["source_sha256"]:
            result.conflicts.append(f"{item['source']}/{item['image']}")
            continue
        copy = xanylabeling.annotation_path(target / item["file"])
        document = json.loads(copy.read_text(encoding="utf-8"))
        document["imagePath"] = item["image"]
        paths.write_json(annotation, document)
        item["source_sha256"], item["copy_sha256"] = paths.sha256(annotation), paths.sha256(copy)
        item.pop("proposed", None)  # 提议已写回：之后与普通副本一样，只在再次修改时写回
        result.written.append(f"{item['source']}/{item['image']}")
    if result.written:
        record["applied_at"] = _now()
        paths.write_json(target / RECORD, record)
    return result


def pending(root: Path, source_names) -> list[tuple[str, int]]:
    """涉及这些来源、有未应用修改的复核集：(名称, 改过的副本数)。"""
    wanted = set(source_names)
    found = []
    for path in sorted((root / REVIEWS).glob(f"*/{RECORD}")) if (root / REVIEWS).is_dir() else []:
        edited = [item for item in _items(path.parent)
                  if item["source"] in wanted and _state(path.parent, item) == "edited"]
        if edited:
            found.append((path.parent.name, len(edited)))
    return found


def listing(root: Path) -> list[str]:
    lines = []
    for path in sorted((root / REVIEWS).glob(f"*/{RECORD}")) if (root / REVIEWS).is_dir() else []:
        items = _items(path.parent)
        edited = sum(_state(path.parent, item) == "edited" for item in items)
        record = _record(path.parent)
        applied = f", last applied {record['applied_at'][:16].replace('T', ' ')}" if record.get("applied_at") else ""
        lines.append(f"  {path.parent.name}: {len(items)} images, {edited} edited and not applied{applied}  "
                     f"({path.parent})")
    return lines or [f"no reviews under {root / REVIEWS}; train eval <id> <run> --errors creates one"]


def _record(target: Path) -> dict:
    if not (target / RECORD).is_file():
        raise TrainError(f"{target} is not a review (no {RECORD})")
    return json.loads((target / RECORD).read_text(encoding="utf-8"))


def _items(target: Path) -> list[dict]:
    return _record(target)["items"]


def _state(target: Path, item: dict) -> str:
    """副本自复制（或上次应用）以来是否被改过；预标注的提议未写回前都算改过。被删除的副本视为没有改动：
    不能由此删除来源的标注，也是不写回某个提议的方式。"""
    copy = xanylabeling.annotation_path(target / item["file"])
    if not copy.is_file():
        return "unchanged"
    if item.get("proposed") or paths.sha256(copy) != item["copy_sha256"]:
        return "edited"
    return "unchanged"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
