"""Ultralytics 后端。模块级不导入 ultralytics：数据视图与选项检查在没有 GPU 与训练依赖的环境中也能测试，
训练与评估只在训练镜像中运行。"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

from .. import paths

NAME = "ultralytics"
TASKS = ("detect",)
# 由框架决定的参数：数据集、输出位置与续训。写在 train.options 中会让 run 的记录与实际不符。
RESERVED = ("data", "project", "name", "exist_ok", "resume", "model", "mode", "task", "split", "save_dir")
# 后端在训练目录中的输出：Ultralytics 写入 <训练目录>/weights/。
BEST = "weights/best.pt"
LAST = "weights/last.pt"
CJK_FONT = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"  # 训练镜像中的 fonts-wqy-microhei
CJK_FAMILY = "WenQuanYi Micro Hei"


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


def check_options(options: dict) -> list[str]:
    return [f"{key} is set by the training tool and cannot be given" for key in RESERVED if key in options]


def version() -> dict:
    import torch
    import ultralytics
    return {"ultralytics": ultralytics.__version__, "torch": torch.__version__}


def train(task: str, data: Path, weights: Path, options: dict, directory: Path, resume: bool,
          workdir: Path) -> Path:
    """训练到 directory（Ultralytics 的 save_dir），返回最佳权重。resume 时 weights 是中断时的 last.pt，
    Ultralytics 从其中恢复全部参数与进度。"""
    from ultralytics import YOLO
    with _settings(workdir):
        model = YOLO(str(weights), task=task)
        if resume:
            model.train(resume=True)
        else:
            model.train(data=str(data), project=str(directory.parent), name=directory.name, exist_ok=True,
                        **options)
    return directory / BEST


def evaluate(task: str, weights: Path, data: Path, split: str, options: dict, directory: Path,
             workdir: Path) -> dict:
    """在指定划分上评估，曲线与混淆矩阵写到 directory。返回 {"metrics": {...}, "classes": {类别号: {...}}}，
    只含该划分中出现的类别。"""
    from ultralytics import YOLO
    keep = ("imgsz", "batch", "device", "half", "workers")
    arguments = {key: options[key] for key in keep if key in options}
    if "device" in arguments:  # 评估只用一张卡：多卡训练的 device 取第一张
        arguments["device"] = str(arguments["device"]).split(",")[0]
    with _settings(workdir):
        result = YOLO(str(weights), task=task).val(data=str(data), split=split, project=str(directory.parent),
                                                   name=directory.name, exist_ok=True, plots=True, **arguments)
    box = result.box
    classes = {}
    for row, index in enumerate(box.ap_class_index):
        classes[int(index)] = {"map50_95": float(box.ap[row]), "map50": float(box.ap50[row]),
                               "precision": float(box.p[row]), "recall": float(box.r[row])}
    return {"metrics": {"map50_95": float(box.map), "map50": float(box.map50), "precision": float(box.mp),
                        "recall": float(box.mr)},
            "classes": classes}


@contextlib.contextmanager
def _settings(workdir: Path):
    """Ultralytics 把下载的权重（包括 AMP 检查用的 yolo11n.pt）写到当前目录：切到训练根目录中的固定位置，
    下载一次，以后复用。同时关闭使用统计的上报，并让图表能显示中文。"""
    import seaborn
    from ultralytics import settings
    settings.update({"sync": False})
    set_theme = seaborn.set_theme
    if Path(CJK_FONT).is_file():
        _use_cjk_font()
        # 混淆矩阵绘制前调用 seaborn.set_theme，它把字体重置为 seaborn 的默认列表：之后重新加上中文字体。
        def set_theme_with_cjk_font(*args, **kwargs):
            set_theme(*args, **kwargs)
            _use_cjk_font()
        seaborn.set_theme = set_theme_with_cjk_font
    previous = Path.cwd()
    workdir.mkdir(parents=True, exist_ok=True)
    os.chdir(workdir)
    try:
        yield
    finally:
        os.chdir(previous)
        seaborn.set_theme = set_theme


def _use_cjk_font() -> None:
    """曲线与混淆矩阵中的类别名多为中文。直接登记字体文件，不依赖 matplotlib 的字体缓存（缓存在训练根目录中，
    可能早于字体安装）；中文字体排在最前，其余字体保留作为后备。"""
    import matplotlib
    from matplotlib import font_manager
    font_manager.fontManager.addfont(CJK_FONT)
    families = [name for name in matplotlib.rcParams["font.sans-serif"] if name != CJK_FAMILY]
    matplotlib.rcParams["font.sans-serif"] = [CJK_FAMILY, *families]
    matplotlib.rcParams["font.family"] = ["sans-serif"]


def _quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
