"""X-AnyLabeling 的标注文件：图像旁的同名 JSON（LabelMe 兼容格式）。

按 4.0.6 实际保存的文件读取（样例见 framework/tests/python/data/x-anylabeling-*/）：
顶层 imagePath、imageWidth、imageHeight、shapes、flags、checked；imageData 为空（非空时忽略）。
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
    checked: bool
    flags: dict
    shapes: list[Shape]


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
                      document.get("checked") is True, document.get("flags") or {}, shapes)


def update(path: Path, **fields) -> None:
    """改写顶层字段，其余内容原样保留（键的顺序也不变，X-AnyLabeling 再次保存时差异最小）。"""
    document = _load(path)
    document.update(fields)
    paths.write_json(path, document)


# 预标注写出的文件按这个版本保存的格式：顶层与形状的字段、键的顺序都与样例相同。
VERSION = "4.0.6"


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
