"""发布包清单：SHA-256 完整性校验（不是签名），以及安装目录中“受管理文件”的跟踪。

受管理文件 = 上一次安装从包里复制进安装目录的文件。升级时只有它们会被当作“退役文件”删除，
且被现场修改过的退役文件会让安装在停服前失败；其他现场文件一律保留。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import layout
from .envfile import DeployError

# 安装目录中不由清单管理的发布包内容。
UNMANAGED_PREFIXES = ("images/",)
UNMANAGED_FILES = (layout.INVENTORY, "install.sh")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_path(root: Path, relative: str) -> Path:
    """清单中的路径只能指向根目录内部，且沿途没有符号链接。"""
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts or not value.parts:
        raise DeployError(f"invalid path in inventory: {relative}")
    path = root
    for part in value.parts:
        path = path / part
        if path.is_symlink():
            raise DeployError(f"symlink in {root}: {path}")
    return path


def write(root: Path) -> dict[str, str]:
    values = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise DeployError(f"package contains a symlink: {path}")
        if path.is_file() and path.name != layout.INVENTORY:
            values[path.relative_to(root).as_posix()] = sha256(path)
    (root / layout.INVENTORY).write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")
    return values


def verify(root: Path) -> dict[str, str]:
    """包内文件必须与清单完全一致：没有缺失、改动、多余文件或符号链接。"""
    manifest = root / layout.INVENTORY
    if not manifest.is_file():
        raise DeployError(f"{manifest} is missing; this is not a complete release package")
    values = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(values, dict) or not values:
        raise DeployError(f"{manifest} is not a valid inventory")
    paths = list(root.rglob("*"))
    links = [p for p in paths if p.is_symlink()]
    if links:
        raise DeployError(f"package contains a symlink: {links[0]}")
    # 解释器在包内运行时生成的字节码缓存不是发布内容。
    actual = {p.relative_to(root).as_posix() for p in paths
              if p.is_file() and p != manifest and "__pycache__" not in p.parts}
    if actual != set(values):
        missing, extra = sorted(set(values) - actual), sorted(actual - set(values))
        raise DeployError(f"package does not match its inventory; missing={missing[:5]} extra={extra[:5]}")
    for relative, digest in values.items():
        if sha256(safe_path(root, relative)) != digest:
            raise DeployError(f"package file is corrupted: {relative}")
    return values


def managed(values: dict[str, str]) -> dict[str, str]:
    return {name: digest for name, digest in values.items()
            if name not in UNMANAGED_FILES and not name.startswith(UNMANAGED_PREFIXES)}


def obsolete(install_dir: Path, incoming: dict[str, str]) -> list[Path]:
    """上一版受管理、新版不再包含、且现场没有改过的文件；改过的退役文件直接报错。"""
    record = install_dir / layout.MANAGED
    if not record.is_file():
        return []
    previous = json.loads(record.read_text(encoding="utf-8"))
    result = []
    for name, digest in sorted(previous.items()):
        if name in incoming:
            continue
        path = safe_path(install_dir, name)
        if not path.exists():
            continue
        if not path.is_file() or sha256(path) != digest:
            raise DeployError(f"{path} was modified locally but is no longer part of the release; "
                              "move it aside and run the installation again")
        result.append(path)
    return result
