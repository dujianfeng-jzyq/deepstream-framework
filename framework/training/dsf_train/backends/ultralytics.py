"""Ultralytics 后端。模块级不导入 ultralytics：数据视图与选项检查在没有 GPU 与训练依赖的环境中也能测试，
训练、评估、推理与导出只在训练镜像中运行。"""

from __future__ import annotations

import contextlib
import os
import shutil
import sys
from pathlib import Path

from .. import media, paths
from ..errors import TrainError

# Ultralytics 在导出等操作前检查依赖，缺少时（例如无 GPU 时要 onnxruntime 而镜像中是 onnxruntime-gpu）
# 自动 pip 安装到用户目录：它会进入训练根目录中的家目录并持续生效，使镜像固定的依赖失去意义。
# 它在导入时读取这个变量，所以在任何导入之前设置。
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

NAME = "ultralytics"
TASKS = ("detect", "classify")
# 导出时能追加转换的输出布局：任务 → 布局。
LAYOUTS = {"detect": ("xyxy-confidence-class_id",), "classify": ("softmax",)}
# 分类视图中类别目录的名字：Ultralytics（torchvision ImageFolder）按目录名排序决定类别号，补零的序号
# 排序后就是 classes 的顺序。
CLASS_DIR_DIGITS = 3
# 检测头的原始输出 [batch, 4 + 类别数, 候选数]：中心点与宽高（输入像素）后接各类别分数。
# [cx, cy, w, h] 乘以它得到 [x1, y1, x2, y2]。
_XYWH_TO_XYXY = [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [-0.5, 0.0, 0.5, 0.0], [0.0, -0.5, 0.0, 0.5]]
# 导出 ONNX 的输入与输出名：固定下来，不随后端版本变化。
_UNSTABLE_METADATA = ("date", "description")
INPUT_NAME = "input"
OUTPUT_NAME = "output"
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


def data_path(task: str, view: Path) -> Path:
    """交给 Ultralytics 的 data：检测是 data.yaml，分类是视图目录本身（train/ val/ test/ 下按类别分目录）。"""
    return view if task == "classify" else view / "data.yaml"


def write_view(task: str, manifest: dict, root: Path, view: Path, location: Path) -> None:
    if task == "classify":
        _write_classify_view(manifest, root, view)
        return
    _write_detect_view(manifest, root, view, location)


def _write_classify_view(manifest: dict, root: Path, view: Path) -> None:
    """分类格式：<划分>/<类别序号>/<来源>__<图片>__<对象序号>.png，每个对象裁剪并拉伸到网络输入尺寸
    （manifest 的 input）。每个划分都建全部类别目录（空的也建）：torchvision 按各划分自己的目录编号，
    缺一个目录会让后面的类别号错位。训练划分中没有样本的类别无法学习，报错。"""
    width, height = manifest["input"]["width"], manifest["input"]["height"]
    classes = manifest["classes"]
    counts = {split: [0] * len(classes) for split in ("train", "val", "test")}
    for split in counts:
        for index in range(len(classes)):
            (view / split / _class_dir(index)).mkdir(parents=True, exist_ok=True)
    for sample in manifest["samples"]:
        image = paths.source_dir(root, sample["source"]) / sample["image"]
        stem = Path(sample_name(sample)).stem
        for number, item in enumerate(sample["objects"]):
            target = view / sample["split"] / _class_dir(item["class"]) / f"{stem}__{number}.png"
            media.crop(image, item["box"], width, height).save(target)  # PNG：裁剪不再经过一次有损压缩
            counts[sample["split"]][item["class"]] += 1
    missing = [name for name, count in zip(classes, counts["train"]) if not count]
    if missing:
        raise TrainError(f"the train split has no samples of {', '.join(missing)}: label more boxes with these "
                         "values, or move the value to data.ignore_classes")


def _class_dir(index: int) -> str:
    return f"{index:0{CLASS_DIR_DIGITS}d}"


def _write_detect_view(manifest: dict, root: Path, view: Path, location: Path) -> None:
    """YOLO 检测格式：images/<划分>/ 硬链接到来源图像，labels/<划分>/ 每行 `类别 cx cy w h`（归一化）。
    负样本写空的标签文件。视图写在 view，data.yaml 指向它最终所在的 location。
    Ultralytics 会就地重写的图像复制而不链接：否则重写经硬链接改动来源，来源的 sha256 不再与导入记录一致。"""
    for sample in manifest["samples"]:
        name = sample_name(sample)
        source = paths.source_dir(root, sample["source"]) / sample["image"]
        target = view / "images" / sample["split"] / name
        if _rewritten_by_ultralytics(source):
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        else:
            paths.link_or_copy(source, target)
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


def _rewritten_by_ultralytics(path: Path) -> bool:
    """Ultralytics 检查数据集时把缺少结束标记（FF D9）的 JPEG 修复后原地保存（"corrupt JPEG restored and
    saved"）。图像能解码，只是结尾被截断或附带了数据。判断条件与它相同。"""
    if path.suffix.lower() not in (".jpg", ".jpeg"):
        return False
    with path.open("rb") as stream:
        stream.seek(-2, os.SEEK_END)
        return stream.read() != b"\xff\xd9"


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

    from .ultralytics_trainer import TRAINERS
    # 多卡训练时 Ultralytics 另起子进程，按模块路径导入训练器类：子进程要能找到训练工具（PYTHONPATH），
    # 导入训练器模块时设置中文字体（子进程不经过这里的 _settings）。
    environment = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    try:
        with _settings(workdir):
            model = YOLO(str(weights), task=task)
            if resume:
                model.train(resume=True, trainer=TRAINERS[task])
            else:
                model.train(data=str(data), project=str(directory.parent), name=directory.name, exist_ok=True,
                            trainer=TRAINERS[task], **options)
    finally:
        if environment is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = environment
    return directory / BEST


def evaluate(task: str, weights: Path, data: Path, split: str, options: dict, directory: Path,
             workdir: Path) -> dict:
    """在指定划分上评估，曲线与混淆矩阵写到 directory。返回 {"metrics": {...}, "classes": {类别号: {...}}}，
    只含该划分中出现的类别。"""
    from ultralytics import YOLO
    arguments = _inference_arguments(options)
    with _settings(workdir):
        result = YOLO(str(weights), task=task).val(data=str(data), split=split, project=str(directory.parent),
                                                   name=directory.name, exist_ok=True, plots=True, **arguments)
    if task == "classify":
        from ..tasks import classify
        # Ultralytics 的混淆矩阵是 matrix[预测][真实]，任务按 [真实][预测] 计算指标。
        matrix = result.confusion_matrix.matrix
        size = len(matrix)
        return classify.metrics([[int(matrix[p][t]) for p in range(size)] for t in range(size)])
    box = result.box
    classes = {}
    for row, index in enumerate(box.ap_class_index):
        classes[int(index)] = {"map50_95": float(box.ap[row]), "map50": float(box.ap50[row]),
                               "precision": float(box.p[row]), "recall": float(box.r[row])}
    return {"metrics": {"map50_95": float(box.map), "map50": float(box.map50), "precision": float(box.mp),
                        "recall": float(box.mr)},
            "classes": classes}


def predict(task: str, weights: Path, images: list[Path], options: dict, conf: float, workdir: Path, *,
            agnostic: bool):
    """逐张推理，按 images 的顺序产出 [{"class", "box", "score"}]。stream 模式按批读入图片，
    不把全部结果留在内存中。Ultralytics 默认按类别分别做 NMS，agnostic 对应它的 agnostic_nms。"""
    from ultralytics import YOLO
    if task != "detect":
        raise ValueError(f"backend {NAME} predicts boxes for task detect only, not {task}")
    with _settings(workdir):
        model = YOLO(str(weights), task=task)
        for result in model.predict([str(path) for path in images], conf=conf, stream=True, verbose=False,
                                    agnostic_nms=agnostic, **_inference_arguments(options)):
            boxes = result.boxes
            yield [{"class": int(c), "box": [float(v) for v in xyxy], "score": float(s)}
                   for xyxy, c, s in zip(boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist())]


def export(task: str, weights: Path, layout: str, size: tuple[int, int], settings: dict, target: Path,
           workdir: Path) -> None:
    """导出 ONNX 到 target，并在输出端追加到 layout 的转换。size 是 (宽, 高)；settings 是 model.yaml 的 export
    （opset、batch、dynamic、simplify）。在 CPU 上导出：不占用正在训练的卡，结果也不依赖显卡。"""
    from ultralytics import YOLO
    if layout not in LAYOUTS.get(task, ()):
        raise ValueError(f"backend {NAME} cannot export task {task} with output layout {layout}")
    work = target.parent / f".{target.name}.export"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    try:
        copy = work / "weights.pt"  # Ultralytics 把 ONNX 写在权重旁边：复制出来，训练目录保持不变
        shutil.copy2(weights, copy)
        with _settings(workdir):
            raw = YOLO(str(copy), task=task).export(
                format="onnx", imgsz=(size[1], size[0]), opset=settings["opset"], batch=settings["batch"],
                dynamic=settings["dynamic"], simplify=settings["simplify"], nms=False, device="cpu")
        if task == "classify":
            _append_softmax(Path(raw), target)
        else:
            _append_layout(Path(raw), target)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def reference(task: str, weights: Path, layout: str, batch):
    """原生推理（PyTorch，CPU）在 layout 下的输出，用于与导出的 ONNX 比较：batch 是 float32 [n, 3, 高, 宽]，
    返回 [n, 候选数, 6] 的 numpy 数组。转换与导出时追加的完全相同。"""
    import torch
    from ultralytics import YOLO
    if layout not in LAYOUTS.get(task, ()):
        raise ValueError(f"backend {NAME} has no {layout} output for task {task}")
    model = YOLO(str(weights), task=task).model.float().eval()
    with torch.no_grad():
        if task == "classify":
            # 分类头输出 logits（Ultralytics 在 Python 的预测器中才做 softmax）；与导出时追加的 Softmax 相同。
            logits = model(torch.from_numpy(batch))
            logits = logits[0] if isinstance(logits, (list, tuple)) else logits
            return logits.softmax(dim=1).numpy()
        raw = model(torch.from_numpy(batch))
        raw = raw[0] if isinstance(raw, (list, tuple)) else raw
        rows = raw.transpose(1, 2)
        boxes = rows[..., :4] @ torch.tensor(_XYWH_TO_XYXY, dtype=rows.dtype)
        scores, classes = rows[..., 4:].max(dim=-1, keepdim=True)
        return torch.cat([boxes, scores, classes.to(rows.dtype)], dim=-1).numpy()


def _append_softmax(source: Path, target: Path) -> None:
    """在分类头的 logits [batch, 类别数] 之后追加 Softmax：nvinfer 内置的分类解析把输出当作概率，与
    classifier-threshold 比较。输入与输出改为固定的名字。"""
    import onnx
    from onnx import TensorProto, helper
    model = onnx.load(str(source))
    graph = model.graph
    if len(graph.input) != 1 or len(graph.output) != 1:
        raise ValueError(f"{source}: expected one input and one output, got {len(graph.input)} and "
                         f"{len(graph.output)}")
    raw = graph.output[0]
    dims = [d.dim_value if d.HasField("dim_value") else d.dim_param for d in raw.type.tensor_type.shape.dim]
    if len(dims) != 2:
        raise ValueError(f"{source}: expected a [batch, classes] output, got {dims}")
    graph.node.append(helper.make_node("Softmax", [raw.name], [OUTPUT_NAME], axis=1))
    graph.output.remove(raw)
    graph.output.append(helper.make_tensor_value_info(OUTPUT_NAME, TensorProto.FLOAT, dims))
    _finish(model, target)


def _finish(model, target: Path) -> None:
    """输入改为固定的名字，去掉不稳定的元数据，检查后保存。"""
    import onnx
    graph = model.graph
    old_input = graph.input[0].name
    graph.input[0].name = INPUT_NAME
    for node in graph.node:
        for index, value in enumerate(node.input):
            if value == old_input:
                node.input[index] = INPUT_NAME
    # 导出时间与含本机路径的描述：去掉后同一份权重总是导出同一个文件（发布标识与 engine 指纹随之不变）。
    keep = [item for item in model.metadata_props if item.key not in _UNSTABLE_METADATA]
    del model.metadata_props[:]
    model.metadata_props.extend(keep)
    onnx.checker.check_model(model)
    onnx.save(model, str(target))


def _append_layout(source: Path, target: Path) -> None:
    """在 Ultralytics 的检测输出 [batch, 4 + 类别数, 候选数] 之后追加：转置为每行一个候选，中心点宽高换算为
    左上右下，取最高分数作为置信度、其类别号（float）作为第 6 列。ArgMax 与 GatherElements 的轴在各 opset 中
    都是属性，不受 ReduceMax 在 opset 18 改为输入的影响。输入与输出改为固定的名字。"""
    import numpy
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    model = onnx.load(str(source))
    graph = model.graph
    if len(graph.input) != 1 or len(graph.output) != 1:
        raise ValueError(f"{source}: expected one input and one output, got {len(graph.input)} and "
                         f"{len(graph.output)}")
    raw = graph.output[0]
    dims = [d.dim_value if d.HasField("dim_value") else d.dim_param for d in raw.type.tensor_type.shape.dim]
    if len(dims) != 3:
        raise ValueError(f"{source}: expected a [batch, 4 + classes, candidates] output, got {dims}")

    def name(suffix: str) -> str:
        return f"dsf_layout/{suffix}"

    graph.initializer.extend([
        numpy_helper.from_array(numpy.array(_XYWH_TO_XYXY, dtype=numpy.float32), name("xywh_to_xyxy")),
        numpy_helper.from_array(numpy.array([0], dtype=numpy.int64), name("zero")),
        numpy_helper.from_array(numpy.array([4], dtype=numpy.int64), name("four")),
        numpy_helper.from_array(numpy.array([numpy.iinfo(numpy.int64).max], dtype=numpy.int64), name("end")),
        numpy_helper.from_array(numpy.array([2], dtype=numpy.int64), name("axis")),
    ])
    graph.node.extend([
        helper.make_node("Transpose", [raw.name], [name("rows")], perm=[0, 2, 1]),
        helper.make_node("Slice", [name("rows"), name("zero"), name("four"), name("axis")], [name("xywh")]),
        helper.make_node("Slice", [name("rows"), name("four"), name("end"), name("axis")], [name("scores")]),
        helper.make_node("MatMul", [name("xywh"), name("xywh_to_xyxy")], [name("xyxy")]),
        helper.make_node("ArgMax", [name("scores")], [name("class_index")], axis=2, keepdims=1),
        helper.make_node("GatherElements", [name("scores"), name("class_index")], [name("confidence")], axis=2),
        helper.make_node("Cast", [name("class_index")], [name("class_id")], to=TensorProto.FLOAT),
        helper.make_node("Concat", [name("xyxy"), name("confidence"), name("class_id")], [OUTPUT_NAME], axis=2),
    ])
    graph.output.remove(raw)
    graph.output.append(helper.make_tensor_value_info(OUTPUT_NAME, TensorProto.FLOAT, [dims[0], dims[2], 6]))
    _finish(model, target)


def _inference_arguments(options: dict) -> dict:
    """评估与推理沿用训练时的输入尺寸与设备；多卡训练的 device 取第一张。"""
    keep = ("imgsz", "batch", "device", "half", "workers")
    arguments = {key: options[key] for key in keep if key in options}
    if "device" in arguments:
        arguments["device"] = str(arguments["device"]).split(",")[0]
    return arguments


@contextlib.contextmanager
def _settings(workdir: Path):
    """Ultralytics 把下载的权重（包括 AMP 检查用的 yolo11n.pt）写到当前目录：切到训练根目录中的固定位置，
    下载一次，以后复用。同时关闭使用统计的上报，并让图表能显示中文。多卡训练的子进程继承当前目录与
    PYTHONPATH（见 train），中文字体由 ultralytics_trainer 在子进程中设置。"""
    from ultralytics import settings
    settings.update({"sync": False})
    use_cjk_font()
    previous = Path.cwd()
    workdir.mkdir(parents=True, exist_ok=True)
    os.chdir(workdir)
    try:
        yield
    finally:
        os.chdir(previous)


_cjk_installed = False


def use_cjk_font() -> None:
    """曲线与混淆矩阵中的类别名多为中文。直接登记字体文件，不依赖 matplotlib 的字体缓存（缓存在训练根目录中，
    可能早于字体安装）；中文字体排在最前，其余字体保留作为后备。混淆矩阵绘制前调用 seaborn.set_theme，
    它把字体重置为 seaborn 的默认列表：包装它，之后重新加上中文字体。在本进程内一直生效，重复调用无害。"""
    global _cjk_installed
    if _cjk_installed or not Path(CJK_FONT).is_file():
        return
    import seaborn
    set_theme = seaborn.set_theme

    def set_theme_with_cjk_font(*args, **kwargs):
        set_theme(*args, **kwargs)
        _apply_cjk_font()
    seaborn.set_theme = set_theme_with_cjk_font
    _apply_cjk_font()
    _cjk_installed = True


def _apply_cjk_font() -> None:
    import matplotlib
    from matplotlib import font_manager
    font_manager.fontManager.addfont(CJK_FONT)
    families = [name for name in matplotlib.rcParams["font.sans-serif"] if name != CJK_FAMILY]
    matplotlib.rcParams["font.sans-serif"] = [CJK_FAMILY, *families]
    matplotlib.rcParams["font.family"] = ["sans-serif"]


def _quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
