"""X-AnyLabeling 的标注文件：图像旁的同名 JSON（LabelMe 兼容格式）。

按 4.0.6 实际保存的文件读取（样例见 framework/tests/python/data/x-anylabeling-*/）：
顶层 imagePath、imageWidth、imageHeight、shapes、flags、checked；imageData 为空（非空时忽略）。
checked 在 4.3 起有三种状态：true（检查通过）、false 或没有（未检查）、"failed"（检查不通过，等人修正）。
每个形状 label、points、shape_type、difficult、attributes；其余字段读取时忽略，改写时原样保留。
预标注写出的文件与 4.0.6 保存的字段相同（create），打开即可修正。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from . import paths
from .errors import TrainError


@dataclass
class Shape:
    label: str
    shape_type: str
    points: list[tuple[float, float]]
    difficult: bool = False
    attributes: dict = field(default_factory=dict)


@dataclass
class Annotation:
    path: Path
    image: str
    width: int
    height: int
    checked: bool                # 检查通过（checked: true）：只有它进入数据集
    flags: dict
    shapes: list[Shape]
    failed: bool = False         # 检查不通过（checked: "failed"）：人看过，认为还要修正


def annotation_path(image: Path) -> Path:
    return image.with_suffix(".json")


def _load(path: Path) -> dict:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise TrainError(f"{path}: not a readable annotation file ({error})") from error
    if not isinstance(document, dict):
        raise TrainError(f"{path}: an annotation file must be a JSON object")
    return document


def read(path: Path) -> Annotation:
    document = _load(path)

    def fail(reason: str) -> TrainError:
        return TrainError(f"{path}: {reason}")

    width, height = document.get("imageWidth"), document.get("imageHeight")
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise fail("imageWidth and imageHeight must be positive integers")
    shapes = []
    for index, raw in enumerate(document.get("shapes") or []):
        if not isinstance(raw, dict) or not isinstance(raw.get("label"), str):
            raise fail(f"shape {index} has no label")
        try:
            points = [(float(x), float(y)) for x, y in raw.get("points") or []]
        except (TypeError, ValueError) as error:
            raise fail(f"shape {index} ({raw['label']}) has invalid points") from error
        attributes = raw.get("attributes") or {}
        shapes.append(Shape(raw["label"], str(raw.get("shape_type") or ""), points,
                            bool(raw.get("difficult")), attributes if isinstance(attributes, dict) else {}))
    return Annotation(path, str(document.get("imagePath") or ""), width, height,
                      document.get("checked") is True, document.get("flags") or {}, shapes,
                      document.get("checked") == "failed")


def update(path: Path, **fields) -> None:
    """改写顶层字段，其余内容原样保留（键的顺序也不变，X-AnyLabeling 再次保存时差异最小）。"""
    document = _load(path)
    document.update(fields)
    paths.write_json(path, document)


def append_shapes(path: Path, shapes: list[dict], expected_sha256: str) -> bool:
    """在已有标注上追加形状，其余内容原样保留。文件内容与 expected_sha256 不同（选中之后人在工具中保存过）
    时不写并返回 False。"""
    if paths.sha256(path) != expected_sha256:
        return False
    document = _load(path)
    document["shapes"] = [*(document.get("shapes") or []), *shapes]
    paths.write_json(path, document)
    return True


def relabel(path: Path, mapping: dict[str, str]) -> bool:
    """按 mapping 改写形状的类别名，其余内容（包括 checked）原样保留；没有要改的形状时不写，返回是否改写。"""
    document = _load(path)
    changed = False
    for shape in document.get("shapes") or []:
        if isinstance(shape, dict) and shape.get("label") in mapping:
            shape["label"] = mapping[shape["label"]]
            changed = True
    if changed:
        paths.write_json(path, document)
    return changed


# 预标注写出的文件按这个版本保存的格式：顶层与形状的字段、键的顺序都与样例相同。
VERSION = "4.0.6"


def rewrite_attribute(path: Path, attribute: str, mapping: dict[str, str | None],
                      labels: set[str] | None = None) -> int:
    """按 mapping 改写形状属性 attribute 的取值（映射到 None 时删去这个属性，框成为没有取值），labels 给出时只改
    这些类别的框；其余内容（包括 checked）原样保留。没有要改的框时不写，返回改写的框数。"""
    document = _load(path)
    count = 0
    for shape in document.get("shapes") or []:
        values = shape.get("attributes") if isinstance(shape, dict) else None
        if not isinstance(values, dict) or str(values.get(attribute, "")) not in mapping:
            continue
        if labels is not None and shape.get("label") not in labels:
            continue
        new = mapping[str(values[attribute])]
        if new is None:
            del values[attribute]
        else:
            values[attribute] = new
        count += 1
    if count:
        paths.write_json(path, document)
    return count


def new_shape(label: str, shape_type: str, points: list[list[float]]) -> dict:
    return {"label": label, "points": points, "group_id": None, "description": "", "difficult": False,
            "shape_type": shape_type, "flags": {}, "attributes": {}, "kie_linking": []}


def rectangle_points(box: list[float]) -> list[list[float]]:
    """[x1, y1, x2, y2] → 4.0.6 保存矩形的四个角点（左上、右上、右下、左下）。"""
    x1, y1, x2, y2 = box
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def create(path: Path, image_name: str, width: int, height: int, shapes: list[dict]) -> bool:
    """写一份未检查（checked: false）的新标注；path 已存在时不写并返回 False，已有的标注永不被覆盖。
    先写临时文件再硬链接到目标：链接在目标存在时失败，X-AnyLabeling 也不会读到写了一半的文件。"""
    document = {"version": VERSION, "flags": {}, "checked": False, "shapes": shapes, "imagePath": image_name,
                "imageData": None, "imageHeight": height, "imageWidth": width, "description": ""}
    text = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
    fd, name = paths.temporary_file(path)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
        try:
            os.link(name, path)
        except FileExistsError:
            return False
        except OSError:  # 不支持硬链接的文件系统：退回独占创建
            try:
                with path.open("x", encoding="utf-8") as stream:
                    stream.write(text)
            except FileExistsError:
                return False
        return True
    finally:
        Path(name).unlink(missing_ok=True)
