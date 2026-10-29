"""项目级脚手架：初始化派生项目、生成服务单元骨架。"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from . import layout
from .errors import DeployError

NAME = re.compile(r"^[a-z][a-z0-9-]*[a-z0-9]$")
TEMPLATES = Path(__file__).resolve().parents[2] / "templates"


def _rewrite(path: Path, key: str, value: str) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*=")
    hits = [i for i, line in enumerate(lines) if pattern.match(line)]
    if not hits:
        raise DeployError(f"{path}: no {key}= line to rewrite")
    for index in hits:
        lines[index] = f"{key}={value}"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def init_project(root: Path, name: str, data_path: str | None) -> None:
    """改项目名与数据目录（其余字段与注释保留），并清除属于旧名称的渲染产物。"""
    if not NAME.match(name):
        raise DeployError(f"invalid project name {name!r}: lowercase letters, digits and '-', "
                          "starting with a letter (it names images, volumes and containers)")
    project_env = root / layout.PROJECT_ENV
    _rewrite(project_env, "PROJECT_NAME", name)
    _rewrite(project_env, "HOST_DATA_PATH", data_path or f"~/{name}/data")
    for stale in (root / layout.ENV, root / layout.GENERATED):
        if stale.is_dir():
            shutil.rmtree(stale)
        elif stale.exists():
            stale.unlink()
    print(f"project: {name}\ndata:    {data_path or f'~/{name}/data'}")


# Compose 文件 services: 下的服务名。只读两空格缩进的键，不解析 YAML（与安装器一样只用标准库）。
_SERVICE_KEY = re.compile(r"^  ([A-Za-z0-9][A-Za-z0-9_.-]*):", re.M)


def _compose_services(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    block = re.search(r"^services:\s*$(.*?)(?=^[^\s#]|\Z)", text, re.M | re.S)
    return set(_SERVICE_KEY.findall(block.group(1))) if block else set()


def _check_unit_name(root: Path, name: str) -> None:
    """名字被占用时报错：同名 Compose 服务会被 Compose 合并成一个（新单元悄悄改写已有服务），
    标识符相同的单元或插件会生成重名的 CMake 目标；dsf 是框架的命名空间与目标前缀。"""
    identifier = name.replace("-", "_")
    if identifier == "dsf" or identifier.startswith("dsf_"):
        raise DeployError(f"invalid service name {name!r}: the dsf prefix is reserved for the framework")
    services = _compose_services(root / layout.COMPOSE_ROOT)
    for compose in (root / layout.SERVICES).glob(f"*/{layout.UNIT_COMPOSE}"):
        services |= _compose_services(compose)
    if name in services:
        raise DeployError(f"invalid service name {name!r}: a Compose service with this name already exists")
    for parent in (layout.SERVICES, layout.PLUGINS):
        for unit in (root / parent).glob("*"):
            if unit.is_dir() and unit.name.replace("-", "_") == identifier:
                raise DeployError(f"invalid service name {name!r}: {parent}/{unit.name} already uses the "
                                  f"identifier {identifier} (CMake targets would clash)")


def generate_unit(kind: str, name: str, target: Path) -> None:
    """把 framework/templates/<kind>/ 生成到 target：__NAME__ 替换为服务名，__IDENT__ 替换为可作标识符的名字
    （- 换成 _）。不检查与已有单元的冲突（见 new_unit）；框架测试用它生成并编译模板单元。"""
    if not NAME.match(name):
        raise DeployError(f"invalid service name {name!r}: lowercase letters, digits and '-'")
    source = TEMPLATES / kind
    if not source.is_dir():
        raise DeployError(f"unknown unit kind {kind!r}; available: "
                          + ", ".join(sorted(p.name for p in TEMPLATES.iterdir() if p.is_dir())))
    identifier = name.replace("-", "_")
    for path in sorted(source.rglob("*")):
        if path.is_dir() or "__pycache__" in path.parts:
            continue
        relative = str(path.relative_to(source)).replace("__NAME__", name).replace("__IDENT__", identifier)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        text = path.read_text(encoding="utf-8").replace("__NAME__", name).replace("__IDENT__", identifier)
        destination.write_text(text, encoding="utf-8")
        destination.chmod(path.stat().st_mode & 0o777)


def new_unit(root: Path, name: str, kind: str) -> Path:
    """从 framework/templates/<kind>/ 生成 services/<name>/（见 generate_unit），名字不得与已有单元冲突。"""
    _check_unit_name(root, name)
    target = root / layout.SERVICES / name
    generate_unit(kind, name, target)
    print(f"created {target.relative_to(root)}/ ({kind}); next: scripts/apply-profile.sh <profile> && "
          "docker compose build deepstream-apps")
    return target
