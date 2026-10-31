"""Ultralytics 后端。这里只写数据集视图，不导入 ultralytics：数据准备在没有 GPU 的环境中也能测试。"""

from __future__ import annotations

from pathlib import Path

from .. import paths

NAME = "ultralytics"
TASKS = ("detect",)


def sample_name(sample: dict) -> str:
    """视图中的文件名：来源与原文件名，避免不同来源的同名图像冲突。"""
    return f"{sample['source']}__{sample['image']}"


def write_view(task: str, manifest: dict, root: Path, view: Path, location: Path) -> None:
    """YOLO 检测格式：images/<划分>/ 硬链接到来源图像，labels/<划分>/ 每行 `类别 cx cy w h`（归一化）。
    负样本写空的标签文件。视图写在 view，data.yaml 指向它最终所在的 location。"""
    for sample in manifest["samples"]:
        name = sample_name(sample)
        paths.link_or_copy(paths.source_dir(root, sample["source"]) / sample["image"],
                           view / "images" / sample["split"] / name)
        width, height = sample["width"], sample["height"]
        lines = []
        for item in sample["objects"]:
            x1, y1, x2, y2 = item["box"]
            lines.append(f"{item['class']} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
                         f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}\n")
        label = view / "labels" / sample["split"] / (Path(name).stem + ".txt")
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("".join(lines), encoding="utf-8")
    names = "".join(f"  {index}: {_quote(name)}\n" for index, name in enumerate(manifest["classes"]))
    splits = "".join(f"{split}: images/{split}\n" for split in ("train", "val", "test")
                     if (view / "images" / split).is_dir())
    (view / "data.yaml").write_text(f"path: {location}\n{splits}names:\n{names}", encoding="utf-8")


def _quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
