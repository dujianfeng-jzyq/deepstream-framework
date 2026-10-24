"""最小 KEY=VALUE 文件：不执行 shell、不展开变量；只按需展开开头的 ~。

三种写法：
  普通（project.env、release.env）：值按字面，不剥引号。
  Compose（.env）：Docker Compose 用自己的 dotenv 规则读取它：未加引号的值中 $ 会被插值、
    " #" 之后被当作注释、开头的引号必须闭合，现场口令因此可能被截断或让 Compose 无法解析。
    写入时一律加引号（quote_compose），读取时去掉（read_compose）。
  现场答案（site.env）：人会手工编辑，写成 KEY="值"；读取时去掉一对匹配的外层引号，
    其余字符按字面（unquote_answer），与清单 site-fields.env 中 KEY="" 表示空值一致。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable


class DeployError(Exception):
    """部署工具的预期失败：消息直接给用户看，不打印堆栈。"""


def quote_compose(value: str) -> str:
    """Compose 按字面读取的写法：单引号内不做任何处理；值含单引号时用双引号并转义反斜杠、双引号与 $。"""
    if "\n" in value or "\r" in value:
        raise DeployError("a value in .env cannot contain a line break")
    if "'" not in value:
        return f"'{value}'"
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$")
    return f'"{escaped}"'


def unquote_compose(value: str) -> str:
    """quote_compose 的逆运算；未加引号的值（旧版本渲染的 .env）原样返回。"""
    if len(value) >= 2 and value[0] == value[-1] == "'":
        return value[1:-1]
    if len(value) >= 2 and value[0] == value[-1] == '"':
        result: list[str] = []
        escaped = False
        for char in value[1:-1]:
            if escaped:
                result.append(char if char in '\\"$' else "\\" + char)
                escaped = False
            elif char == "\\":
                escaped = True
            else:
                result.append(char)
        return "".join(result)
    return value


def quote_answer(value: str) -> str:
    return f'"{value}"'


def unquote_answer(value: str) -> str:
    """去掉一对匹配的外层引号（"" 或 ''），其余按字面：现场手写 KEY="" 表示空值而不是两个引号。"""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def parse(text: str, origin: str = "", expand_home: bool = True,
          unquote: Callable[[str], str] | None = None) -> dict[str, str]:
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
        if unquote is not None and value[:1] in ("'", '"'):
            value = unquote(value)
        values[key] = os.path.expanduser(value) if expand_home and value.startswith("~") else value
    return values


def read(path: Path, expand_home: bool = True, unquote: Callable[[str], str] | None = None) -> dict[str, str]:
    try:
        return parse(path.read_text(encoding="utf-8"), str(path), expand_home, unquote)
    except FileNotFoundError as error:
        raise DeployError(f"missing file: {path}") from error


def read_compose(path: Path) -> dict[str, str]:
    """读取渲染器写的 .env：值按字面（写入前已展开 ~）。"""
    return read(path, expand_home=False, unquote=unquote_compose)


def render(values: dict[str, str], header: list[str] | tuple[str, ...] = (),
           quote: Callable[[str], str] | None = None) -> str:
    lines = [f"# {line}" if line else "#" for line in header]
    lines += [f"{key}={quote(values[key]) if quote else values[key]}" for key in sorted(values)]
    return "\n".join(lines) + "\n"


def write(path: Path, values: dict[str, str], header: list[str] | tuple[str, ...] = (),
          mode: int | None = None, quote: Callable[[str], str] | None = None) -> None:
    path.write_text(render(values, header, quote), encoding="utf-8")
    if mode is not None:
        path.chmod(mode)


def require(values: dict[str, str], key: str, origin: str) -> str:
    value = values.get(key, "")
    if not value:
        raise DeployError(f"{origin} does not define {key}")
    return value
