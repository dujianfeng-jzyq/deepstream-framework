"""X-AnyLabeling 的标注文件：图像旁的同名 JSON（LabelMe 兼容格式）。

按 4.0.6 实际保存的文件读取（样例见 framework/tests/python/data/x-anylabeling-*/）：
顶层 imagePath、imageWidth、imageHeight、shapes、flags、checked；imageData 为空（非空时忽略）。
每个形状 label、points、shape_type、difficult、attributes；其余字段读取时忽略，改写时原样保留。
"""

from __future__ import annotations

import json
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
