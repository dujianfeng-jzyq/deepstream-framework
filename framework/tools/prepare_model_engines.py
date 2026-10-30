#!/usr/bin/env python3
"""按 nvinfer 配置预构建 TensorRT engine，业务容器只加载、不构建。

扫描模型目录下的 *.txt，[property] 段同时含 onnx-file 与 model-engine-file 的就是待构建配置；
输出路径完全遵循 model-engine-file（相对路径以配置文件所在目录为基准）。

复用条件：engine 非空，且指纹与本次构建输入一致。指纹记录 ONNX 内容哈希、TensorRT 版本、
GPU 计算能力、影响产物的 nvinfer 属性、trtexec 参数，以及 engine 自身的哈希（engine 与指纹
必须配套）。同一 engine 的构建用文件锁串行化，并发启动时后到者直接复用。

额外的 trtexec 参数（动态形状、INT8 校准等）写在配置旁的 <配置名>.trtexec 文件中，每行若干参数，
# 开头为注释；这些参数计入指纹。

    python3 tools/prepare_model_engines.py --models-path /models
    python3 tools/prepare_model_engines.py /models/<model>/deployment/config.txt
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import fcntl
import hashlib
import os
import re
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

DEFAULT_TRTEXEC = Path("/usr/src/tensorrt/bin/trtexec")
DEFAULT_MODELS = Path("/models")
# 影响 engine 内容的 nvinfer 属性；其它属性（阈值、标签等）只影响后处理。
PRODUCT_PROPERTIES = ("network-mode", "batch-size", "int8-calib-file")
# 只决定用哪块卡构建、不影响产物的参数；计算能力已单独计入指纹。
IGNORED_ARG_PREFIXES = ("--device",)
TRT_VERSION = re.compile(r"TensorRT\s+v(\d+)")
# CUDA 驱动 API 的 CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR / _MINOR。
CU_CC_MAJOR, CU_CC_MINOR = 75, 76


class Failure(Exception):
    pass


def log(message: str) -> None:
    now = datetime.now().astimezone()
    stamp = now.strftime("%Y-%m-%d %H:%M:%S.") + f"{now.microsecond // 1000:03d}"
    print(f"[{stamp}] [info] [prepare-model-engines] {message}", file=sys.stderr, flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_properties(text: str) -> dict[str, str]:
    """只读取 [property] 段；其它段的同名键不参与构建。"""
    properties: dict[str, str] = {}
    in_property = False
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            in_property = line[1:-1].strip() == "property"
            continue
        if in_property and "=" in line:
            key, value = line.split("=", 1)
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            properties[key.strip()] = value
    return properties


@dataclass
class ModelConfig:
    path: Path
    properties: dict[str, str]
    sidecar_args: list[str] = field(default_factory=list)

    @property
    def buildable(self) -> bool:
        return bool(self.properties.get("onnx-file") and self.properties.get("model-engine-file"))

    @property
    def gpu_id(self) -> int:
        return int(self.properties.get("gpu-id") or "0")

    def resolve(self, key: str) -> Path:
        value = Path(self.properties[key])
        return value if value.is_absolute() else self.path.parent / value


def read_sidecar(config: Path) -> list[str]:
    sidecar = config.with_suffix(".trtexec")
    if not sidecar.is_file():
        return []
    args: list[str] = []
    for line in sidecar.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            args.extend(shlex.split(line))
    return args


def load_configs(explicit: list[Path], scan: list[Path]) -> list[ModelConfig]:
    """显式指定的配置必须可读且可构建；扫描到的普通文本静默跳过。"""
    candidates: dict[Path, bool] = {path.resolve(): True for path in explicit}
    for directory in scan:
        if not directory.is_dir():
            raise Failure(f"models path not found: {directory}")
        found = sorted(p for p in directory.rglob("*.txt") if p.is_file())
        log(f"scanned {directory}: {len(found)} .txt file(s)")
        for path in found:
            candidates.setdefault(path.resolve(), False)
    configs = []
    for path, required in candidates.items():
        try:
            config = ModelConfig(path, parse_properties(path.read_text(encoding="utf-8")))
        except (OSError, UnicodeDecodeError) as error:
            if required:
                raise Failure(f"cannot read nvinfer config {path}: {error}") from error
            continue
        if not config.buildable:
            if required:
                raise Failure(f"{path}: [property] needs onnx-file and model-engine-file")
            continue
        config.sidecar_args = read_sidecar(path)
        configs.append(config)
    return configs


def read_config_list(path: Path) -> list[Path]:
    if not path.is_file():
        raise Failure(f"config list not found: {path}")
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.split("#", 1)[0].strip()
        if value:
            candidate = Path(value)
            result.append(candidate if candidate.is_absolute() else path.parent / candidate)
    return result


def trt_version(trtexec: Path) -> str:
    # 没有 GPU 时 --version 也会先打印版本横幅再失败，所以只看输出不看退出码。
    try:
        output = subprocess.run([str(trtexec), "--version"], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, check=False).stdout
    except OSError as error:
        raise Failure(f"cannot run {trtexec}: {error}") from error
    match = TRT_VERSION.search(output or "")
    if match is None:
        raise Failure(f"cannot read the TensorRT version from {trtexec} --version")
    return f"trt{match.group(1)}"


def compute_capability(device: int) -> str:
    """经 CUDA 驱动 API 读取（例如 8.9）。NVIDIA runtime 在 x86 与 Jetson 的容器中都提供 libcuda，
    trtexec 本身也依赖它；设备编号与 trtexec --device、nvinfer gpu-id 一样按 CUDA 的顺序
    （受 CUDA_VISIBLE_DEVICES 影响），nvidia-smi --id 按 PCI 顺序，多卡时可能指向另一块卡。"""
    try:
        cuda = ctypes.CDLL("libcuda.so.1")
    except OSError as error:
        raise Failure(f"cannot load libcuda to read the compute capability of GPU {device}: {error}; "
                      "the builder needs a GPU (runtime: nvidia)") from error
    handle, major, minor = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
    if cuda.cuInit(0) != 0 or cuda.cuDeviceGet(ctypes.byref(handle), device) != 0 or \
            cuda.cuDeviceGetAttribute(ctypes.byref(major), CU_CC_MAJOR, handle) != 0 or \
            cuda.cuDeviceGetAttribute(ctypes.byref(minor), CU_CC_MINOR, handle) != 0:
        raise Failure(f"cannot read the compute capability of GPU {device} from libcuda; "
                      "the builder needs a GPU (runtime: nvidia)")
    return f"{major.value}.{minor.value}"


def build_args(config: ModelConfig, extra: list[str]) -> list[str]:
    args: list[str] = []
    mode = config.properties.get("network-mode")
    if mode == "2":
        args.append("--fp16")
    elif mode == "1":
        args.append("--int8")
        if config.properties.get("int8-calib-file"):
            args.append(f"--calib={config.resolve('int8-calib-file')}")
    for arg in [*config.sidecar_args, *extra]:
        if arg not in args:
            args.append(arg)
    return args


def fingerprint(config: ModelConfig, onnx: Path, trt: str, capability: str, args: list[str]) -> list[str]:
    kept = [a for a in args if not a.startswith(IGNORED_ARG_PREFIXES)]
    lines = [f"onnx_sha256={sha256(onnx)}", f"tensorrt={trt}", f"compute_cap={capability}"]
    lines += [f"nvinfer.{key}={config.properties.get(key, '')}" for key in PRODUCT_PROPERTIES]
    lines.append(f"args={shlex.join(kept)}")
    return lines


def ready(engine: Path, mark: Path, expected: list[str]) -> bool:
    """engine 非空、指纹的构建输入一致，且指纹记录的哈希就是这个 engine。"""
    if not engine.is_file() or engine.stat().st_size == 0 or not mark.is_file():
        return False
    try:
        recorded = mark.read_text(encoding="utf-8").splitlines()
    except OSError:
        return False
    return recorded == [*expected, f"engine_sha256={sha256(engine)}"]


def write_atomically(path: Path, text: str) -> None:
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(name, 0o644)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


@contextlib.contextmanager
def engine_lock(engine: Path):
    """同一 engine 的检查与构建串行化。目录不可写时不加锁：只有复用这条路可走。"""
    try:
        engine.parent.mkdir(parents=True, exist_ok=True)
        stream = open(engine.with_name(engine.name + ".lock"), "a")
    except OSError:
        yield
        return
    with stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def prepare(config: ModelConfig, trtexec: Path, trt: str, capability: str, extra: list[str],
            force: bool) -> bool:
    """trt 与 capability 是本次运行探测一次的 TensorRT 版本与该配置所用 GPU 的计算能力。
    返回 True 表示这次构建了新 engine。"""
    onnx = config.resolve("onnx-file")
    engine = config.resolve("model-engine-file")
    mark = engine.with_name(engine.name + ".fingerprint")
    if not onnx.is_file():
        raise Failure(f"ONNX model not found: {onnx} (from {config.path})")
    args = build_args(config, extra)
    expected = fingerprint(config, onnx, trt, capability, args)

    with engine_lock(engine):
        if not force and ready(engine, mark, expected):
            log(f"engine is ready: {engine}")
            return False
        log(f"{'forced rebuild' if force else 'building'}: {engine}")
        try:
            engine.parent.mkdir(parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(prefix=f".{engine.name}.", dir=engine.parent)
            os.close(fd)
        except OSError as error:
            raise Failure(f"cannot write next to {engine}: {error} (is the mount writable?)") from error
        temporary = Path(name)
        try:
            command = [str(trtexec), f"--onnx={onnx}", f"--saveEngine={temporary}", f"--device={config.gpu_id}", *args]
            log("run: " + shlex.join(command))
            if subprocess.run(command, check=False).returncode != 0:
                raise Failure(f"trtexec failed for {config.path}; the previous engine is kept")
            if temporary.stat().st_size == 0:
                raise Failure(f"trtexec produced an empty engine for {config.path}")
            os.chmod(temporary, 0o644)
            # 先写指纹再替换 engine 会让旧 engine 配上新指纹；反过来则中途失败时
            # 新 engine 配旧指纹——由于指纹记录 engine 哈希，这种不配套下次一定重建。
            os.replace(temporary, engine)
            write_atomically(mark, "\n".join([*expected, f"engine_sha256={sha256(engine)}"]) + "\n")
        finally:
            temporary.unlink(missing_ok=True)
    log(f"engine is ready: {engine}")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare TensorRT engines declared by nvinfer configs.")
    parser.add_argument("configs", nargs="*", type=Path, metavar="NVINFER_CONFIG")
    parser.add_argument("--models-path", action="append", default=[], type=Path, metavar="DIR",
                        help=f"递归扫描目录下的 *.txt（可重复；未指定任何配置时默认 {DEFAULT_MODELS}）")
    parser.add_argument("--config-list", action="append", default=[], type=Path, metavar="FILE",
                        help="每行一个 nvinfer 配置路径，相对路径以清单所在目录为基准")
    parser.add_argument("--trtexec", type=Path, default=None,
                        help=f"trtexec 路径（默认 $TRTEXEC_PATH 或 {DEFAULT_TRTEXEC}）")
    parser.add_argument("--trtexec-arg", action="append", default=[], metavar="ARG",
                        help="追加给所有模型的 trtexec 参数（可重复）")
    parser.add_argument("--compute-cap", default=None, help="覆盖探测到的 GPU 计算能力（无 GPU 验证用）")
    parser.add_argument("--force", action="store_true", help="忽略指纹，强制重建")
    args = parser.parse_args(argv)

    trtexec = args.trtexec or Path(os.environ.get("TRTEXEC_PATH") or DEFAULT_TRTEXEC)
    try:
        if not trtexec.is_file():
            raise Failure(f"trtexec not found: {trtexec} (set TRTEXEC_PATH or --trtexec)")
        explicit = list(args.configs)
        for path in args.config_list:
            explicit += read_config_list(path)
        scan = args.models_path or ([] if explicit else [DEFAULT_MODELS])
        configs = load_configs(explicit, scan)
        if not configs:
            log("no buildable nvinfer config found; nothing to do")
            return 0
        extra = [*shlex.split(os.environ.get("TRTEXEC_ARGS", "")), *args.trtexec_arg]
        # 构建输入在一次运行中不变：每个 GPU 只探测一次，探测失败时尚未开始任何构建。
        trt = trt_version(trtexec)
        capabilities = {device: args.compute_cap or compute_capability(device)
                        for device in sorted({c.gpu_id for c in configs})}
        built = sum(prepare(c, trtexec, trt, capabilities[c.gpu_id], extra, args.force) for c in configs)
        log(f"done: built={built} reused={len(configs) - built}")
        return 0
    except Failure as error:
        print(f"prepare-model-engines: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
