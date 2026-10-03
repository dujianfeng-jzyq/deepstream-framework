"""项目级脚手架：初始化派生项目、生成服务单元骨架。"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from . import layout
from .envfile import DeployError

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


def new_unit(root: Path, name: str, kind: str) -> Path:
    """从 framework/templates/<kind>/ 生成 services/<name>/：__NAME__ 替换为服务名，
    __IDENT__ 替换为可作标识符的名字（- 换成 _）。"""
    if not NAME.match(name):
        raise DeployError(f"invalid service name {name!r}: lowercase letters, digits and '-'")
    source = TEMPLATES / kind
    if not source.is_dir():
        raise DeployError(f"unknown unit kind {kind!r}; available: "
                          + ", ".join(sorted(p.name for p in TEMPLATES.iterdir() if p.is_dir())))
    target = root / layout.SERVICES / name
    if target.exists():
        raise DeployError(f"{target} already exists")
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
    print(f"created {target.relative_to(root)}/ ({kind}); next: cmake --build build && "
          "scripts/apply-profile.sh")
    return target
