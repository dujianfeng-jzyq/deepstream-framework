"""最小 KEY=VALUE 文件：不执行 shell、不剥引号、不展开变量；只按需展开开头的 ~。"""

from __future__ import annotations

import os
from pathlib import Path


class DeployError(Exception):
    """部署工具的预期失败：消息直接给用户看，不打印堆栈。"""


def parse(text: str, origin: str = "", expand_home: bool = True) -> dict[str, str]:
    values: dict[str, str] = {}
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise DeployError(f"{origin}:{number}: expected KEY=VALUE, got {raw!r}")
        key, value = (part.strip() for part in line.split("=", 1))
        if not key:
            raise DeployError(f"{origin}:{number}: empty key")
        values[key] = os.path.expanduser(value) if expand_home and value.startswith("~") else value
    return values


def read(path: Path, expand_home: bool = True) -> dict[str, str]:
    try:
        return parse(path.read_text(encoding="utf-8"), str(path), expand_home)
    except FileNotFoundError as error:
        raise DeployError(f"missing file: {path}") from error


def render(values: dict[str, str], header: list[str] | tuple[str, ...] = ()) -> str:
    lines = [f"# {line}" if line else "#" for line in header]
    lines += [f"{key}={values[key]}" for key in sorted(values)]
    return "\n".join(lines) + "\n"


def write(path: Path, values: dict[str, str], header: list[str] | tuple[str, ...] = (),
          mode: int | None = None) -> None:
    path.write_text(render(values, header), encoding="utf-8")
    if mode is not None:
        path.chmod(mode)


def require(values: dict[str, str], key: str, origin: str) -> str:
    value = values.get(key, "")
    if not value:
        raise DeployError(f"{origin} does not define {key}")
    return value
