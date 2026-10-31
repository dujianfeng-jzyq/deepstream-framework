"""训练根目录的布局与公共的文件操作。全部路径约定只在这里定义（见 docs/training.md §4）。"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from dsf_deploy import envfile, layout

from .errors import TrainError

SOURCES = "sources"
DATASETS = "datasets"
RUNS = "runs"
SOURCE_RECORD = "source.json"
SOURCE_LABELS = "labels.txt"
MANIFEST = "manifest.json"
VIEW = "view"
# datasets/<id>/latest：最近一次 prepare 得到的数据集标识，train run 的默认输入。
LATEST = "latest"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def train_root(repo: Path, explicit: str | None = None) -> Path:
    """--train-root > 环境变量 TRAIN_ROOT > ~/<项目名>/training（与 HOST_DATA_PATH 的默认值同一习惯）。"""
    value = explicit or os.environ.get("TRAIN_ROOT")
    if value:
        return Path(os.path.expanduser(value)).resolve()
    project = envfile.require(envfile.read(repo / layout.PROJECT_ENV), "PROJECT_NAME", layout.PROJECT_ENV)
    return Path.home() / project / "training"


def model_dir(repo: Path, model_id: str) -> Path:
    return repo / layout.MODELS / model_id


def source_dir(root: Path, name: str) -> Path:
    if not name or "/" in name or name.startswith("."):
        raise TrainError(f"invalid source name: {name!r}")
    return root / SOURCES / name


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, document: object) -> None:
    """原子写入：读者（另一个命令或 X-AnyLabeling）不会看到写了一半的文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(document, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(name, path)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise


def link_or_copy(source: Path, target: Path) -> None:
    """同一文件系统时硬链接（不占额外空间，也不会被误改成另一份内容），否则复制。"""
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)
