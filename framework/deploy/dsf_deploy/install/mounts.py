"""按最终 Compose 定义检查并准备可写的 bind 挂载源。

只检查实际使用的可写 bind（profile 可以覆盖挂载，未使用的 HOST_DATA_PATH 不创建也不检查）。
数据挂载点必须是可写 bind 目录；任何可写源都不得与安装目录或发布包互相包含，
唯一例外是 engine 预构建服务写本次安装的 models/。显式 create_host_path: false 的源必须已存在。
"""

from __future__ import annotations

import getpass
import os
from pathlib import Path

from .. import layout
from ..errors import DeployError

ENGINE_BUILDER = "model-engine-builder"


def _overlaps(a: Path, b: Path) -> bool:
    return a == b or a in b.parents or b in a.parents


def ensure_directory(path: Path) -> None:
    """创建并确认当前用户可写；失败时指出真正阻塞的路径层级。"""
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        blocker = next((p for p in [path, *path.parents] if p.exists() and not p.is_dir()), None)
        if blocker is not None:
            raise DeployError(f"无法创建 {path}：{blocker} 是文件，请先移走") from error
        parent = next((p for p in path.parents if p.exists()), Path("/"))
        raise DeployError(f"无法创建 {path}：{getpass.getuser()} 不能写入 {parent}\n"
                          f"  修复：sudo mkdir -p {path} && sudo chown -R $(id -u):$(id -g) {path}") from error
    if not os.access(path, os.W_OK):
        owner = path.stat()
        raise DeployError(f"{getpass.getuser()} 不能写入 {path}（属主 {owner.st_uid}:{owner.st_gid}）\n"
                          f"  修复：sudo chown -R $(id -u):$(id -g) {path}\n"
                          "  （Docker 以 root 创建不存在的 bind 源目录，之前的某次启动可能这样做过）")


def prepare(services: dict, install_dir: Path, package_root: Path) -> list[Path]:
    """检查全部可写 bind 后再创建目录，坏挂载不会在发现之前留下半成品。返回准备好的路径。"""
    data_target = layout.MOUNTS["CONTAINER_DATA_PATH"]
    models_target = layout.MOUNTS["CONTAINER_MODELS_PATH"]
    directories: dict[Path, bool] = {}
    must_exist: set[Path] = set()
    for name, service in sorted(services.items()):
        for mount in service.get("volumes") or []:
            is_data = mount.get("target") == data_target
            writable_bind = mount.get("type") == "bind" and not mount.get("read_only")
            if is_data and not writable_bind:
                raise DeployError(f"{name}：{data_target} 必须是可写的 bind 挂载")
            if not writable_bind:
                continue
            source = Path(mount["source"]).resolve()
            if name == ENGINE_BUILDER and mount.get("target") == models_target and \
                    source == (install_dir / layout.MODELS).resolve():
                continue
            for protected, label in ((install_dir, "安装"), (package_root, "发布包")):
                if _overlaps(source, protected):
                    raise DeployError(f"{name}：可写挂载 {source} 与{label}目录 {protected} 重叠")
            directories[source] = directories.get(source, False) or is_data
            if (mount.get("bind") or {}).get("create_host_path") is False:
                must_exist.add(source)
    for path in sorted(must_exist):
        if not path.exists():
            raise DeployError(f"bind 源必须已存在（create_host_path: false）：{path}")
    for path, is_data in sorted(directories.items()):
        if is_data or not path.exists() or path.is_dir():
            ensure_directory(path)
        elif not os.access(path, os.W_OK):
            raise DeployError(f"可写的 bind 源不可写：{path}")
    return sorted(directories)
