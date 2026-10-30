"""安装目标路径的安全检查：安装器写入的路径不能经符号链接跳出安装目录，也不能与已有的
文件/目录类型冲突。在读取旧元数据之前和停服之前各检查一次。

这不是并发修改的防护：安装期间不得同时改动安装目录。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .. import layout
from ..errors import DeployError

CONTROL_FILES = (layout.ENV, layout.SITE_ENV, layout.PROJECT_ENV, layout.RELEASE_ENV, layout.MANAGED,
                 layout.COMPOSE_ROOT, layout.SITE_FIELDS)
CONTROL_DIRS = ("framework", layout.SERVICES, layout.PROFILES, layout.MODELS, layout.GENERATED)


def _check(install_dir: Path, relative: str, directory: bool) -> None:
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts or not value.parts:
        raise DeployError(f"无效的安装路径：{relative}")
    target = install_dir / value
    chain = [install_dir] + [install_dir.joinpath(*value.parts[:i]) for i in range(1, len(value.parts) + 1)]
    for path in chain:
        if path.is_symlink():
            raise DeployError(f"安装目标中有符号链接：{path}；请先移走")
        if not path.exists():
            continue
        wants_directory = path != target or directory
        if wants_directory != path.is_dir() or (not wants_directory and not path.is_file()):
            kind = "目录" if wants_directory else "普通文件"
            raise DeployError(f"安装目标必须是{kind}：{path}")


def check(package_root: Path, install_dir: Path, extra_files: Iterable[str] = ()) -> None:
    for parent in reversed(install_dir.parents):
        if parent.exists() and not parent.is_dir():
            raise DeployError(f"安装目录的上级不是目录：{parent}")
    for name in CONTROL_FILES:
        _check(install_dir, name, False)
    for name in CONTROL_DIRS:
        _check(install_dir, name, True)
    for path in sorted(package_root.rglob("*")):
        relative = path.relative_to(package_root).as_posix()
        if path.is_symlink():
            raise DeployError(f"发布包中有符号链接：{path}")
        if relative.startswith("images") or relative in ("install.sh", layout.INVENTORY):
            continue
        _check(install_dir, relative, path.is_dir())
    # generated/ 会整体替换（不跟随其中的旧链接），只检查它之外的快照路径。
    for name in extra_files:
        if Path(name).parts[0] != layout.GENERATED:
            _check(install_dir, name, False)
