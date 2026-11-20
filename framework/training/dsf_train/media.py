"""图像与视频：读取尺寸、感知哈希、视频抽帧、网络输入。图像用 Pillow，视频用 ffmpeg 命令行。"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .errors import TrainError

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv", ".ts", ".m4v", ".flv", ".webm")


def _pillow():
    try:
        from PIL import Image
    except ModuleNotFoundError as error:
        raise TrainError("Pillow is required (it is in the training image; run through scripts/train.sh)") from error
    return Image


def image_size(path: Path) -> tuple[int, int]:
    """图像的宽高；文件不是可解码的图像时抛出 TrainError。"""
    Image = _pillow()
    try:
        with Image.open(path) as image:
            image.load()  # 完整解码：截断的文件在这里失败，而不是在训练中
            return image.size
    except Exception as error:
        raise TrainError(f"{path}: cannot decode the image ({error})") from error


def dhash(path: Path) -> int:
    """64 位差值哈希：相邻像素的明暗关系。近似画面的哈希只差少数几位，用于去掉视频中的重复帧。"""
    Image = _pillow()
    with Image.open(path) as image:
        small = image.convert("L").resize((9, 8), Image.BILINEAR)
        pixels = list(small.getdata())
    value = 0
    for row in range(8):
        for column in range(8):
            value = (value << 1) | (pixels[row * 9 + column] > pixels[row * 9 + column + 1])
    return value


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def extract_frames(video: Path, every_s: float, target: Path) -> list[tuple[Path, int]]:
    """按 every_s 秒的间隔抽帧到 target，返回 (帧文件, 时间毫秒)。时间由间隔推算，第一帧在 0 秒。"""
    if shutil.which("ffmpeg") is None:
        raise TrainError("ffmpeg is required to import videos (it is in the training image)")
    if every_s <= 0:
        raise TrainError("--every must be positive")
    target.mkdir(parents=True, exist_ok=True)
    pattern = target / "%08d.jpg"
    result = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(video),
                             "-vf", f"fps=1/{every_s}", "-q:v", "2", "-start_number", "0", str(pattern)],
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise TrainError(f"{video}: cannot extract frames ({result.stderr.strip() or result.returncode})")
    frames = sorted(target.glob("*.jpg"))
    return [(frame, round(int(frame.stem) * every_s * 1000)) for frame in frames]


# letterbox 的填充色：与 Ultralytics 训练与推理时相同。
LETTERBOX_FILL = (114, 114, 114)


def letterbox(path: Path, width: int, height: int):
    """按网络输入尺寸等比缩放、居中填充，得到 float32 [3, height, width]、RGB、取值 0..1 的数组。
    发布时用它给原生推理与导出的 ONNX 同一份输入：比较的是导出本身，与预处理是否和现场相同无关。"""
    import numpy
    Image = _pillow()
    with Image.open(path) as image:
        image = image.convert("RGB")
        scale = min(width / image.width, height / image.height)
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        canvas = Image.new("RGB", (width, height), LETTERBOX_FILL)
        canvas.paste(image.resize(size, Image.BILINEAR), ((width - size[0]) // 2, (height - size[1]) // 2))
    return numpy.asarray(canvas, dtype=numpy.float32).transpose(2, 0, 1) / 255.0


def crop(path: Path, box: list[float], width: int, height: int):
    """裁出 box（像素坐标 [x1, y1, x2, y2]）并拉伸到网络输入尺寸，返回 RGB 的 Pillow 图像。
    二级 nvinfer（process-mode=2）把目标框原样缩放到网络输入、不保持宽高比也不外扩，训练样本按同样的方式生成。"""
    Image = _pillow()
    with Image.open(path) as image:
        image = image.convert("RGB")
        left, top = int(box[0]), int(box[1])
        right, bottom = max(left + 1, round(box[2])), max(top + 1, round(box[3]))
        return image.crop((left, top, right, bottom)).resize((width, height), Image.BILINEAR)


def as_input(image):
    """Pillow 图像 → float32 [3, 高, 宽]、RGB、取值 0..1 的数组（与 net-scale-factor=1/255 相同）。"""
    import numpy
    return numpy.asarray(image.convert("RGB"), dtype=numpy.float32).transpose(2, 0, 1) / 255.0
