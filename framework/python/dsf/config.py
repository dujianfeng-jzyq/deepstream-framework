"""配置读取：读取代码即 schema，与 C++ 的 dsf::config 规则一致。

    root = Reader.load("/config/my-service.yaml")
    host = root.section("redis").get("host", "127.0.0.1")
    root.finish()          # 拒绝没有读取过的键

类型由默认值（或 required 的 kind）决定，转换规则与 C++ 相同：未加引号的标量按所需类型解析
（str 接受任何标量原文；bool 接受 true/false/yes/no/on/off，不区分大小写；int 只接受十进制整数；
float 接受整数与小数），加引号的标量只能读成 str。null 视为缺失。
错误抛 ConfigError，消息为 "<来源>: <完整路径>: <原因>"。
"""

from __future__ import annotations

import math
import re
from typing import Any, Callable, Optional

from . import yamlio

_MISSING = object()
_INTEGER = re.compile(r"-?[0-9]+")
_FLOAT = re.compile(r"[+-]?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][+-]?[0-9]+)?")
_BOOLEANS = {"true": True, "yes": True, "on": True, "false": False, "no": False, "off": False}


def _parse_plain(text: str, kind: type) -> Any:
    """未加引号的标量原文按 kind 解析；不合法返回 _MISSING。"""
    if kind is bool:
        return _BOOLEANS.get(text.lower(), _MISSING)
    if kind is int:
        return int(text) if _INTEGER.fullmatch(text) else _MISSING
    if kind is float:
        if not _FLOAT.fullmatch(text):
            return _MISSING
        value = float(text)
        return value if math.isfinite(value) else _MISSING
    return str(text)


def _convert(value: Any, kind: type) -> Any:
    """返回 kind 类型的值；不合法返回 _MISSING。value 可以是原文（yamlio.Plain）或已解析的值。"""
    if isinstance(value, yamlio.Plain):
        return _parse_plain(value, kind)
    if kind is bool:
        return value if isinstance(value, bool) else _MISSING
    if kind is int:
        return value if isinstance(value, int) and not isinstance(value, bool) else _MISSING
    if kind is float:
        valid = isinstance(value, (int, float)) and not isinstance(value, bool)
        return float(value) if valid else _MISSING
    return value if isinstance(value, str) else _MISSING


class ConfigError(Exception):
    pass


Check = Callable[[Any], Optional[str]]


def one_of(*allowed: str) -> Check:
    return lambda value: None if value in allowed else f"{value!r} is not one of: {' '.join(allowed)}"


def at_least(minimum: float) -> Check:
    return lambda value: None if value >= minimum else f"must be at least {minimum}"


def non_empty(value: Any) -> Optional[str]:
    return None if value else "must not be empty"


class _State:
    def __init__(self, origin: str, document: dict):
        self.origin = origin
        self.document = document
        self.leaves: set[str] = set()
        self.containers: set[str] = set()
        self.described: list[tuple[str, str]] = []
        self.warnings: list[str] = []


class Reader:
    def __init__(self, state: _State, node: Optional[dict], path: str):
        self._state = state
        self._node = node
        self._path = path

    @classmethod
    def load(cls, path: str) -> "Reader":
        try:
            with open(path, encoding="utf-8") as stream:
                document = yamlio.load_text(stream)
        except OSError as error:
            raise ConfigError(f"{path}: cannot open config file: {error.strerror}") from error
        except Exception as error:  # yaml.YAMLError 及其子类
            raise ConfigError(f"{path}: cannot parse YAML: {error}") from error
        return cls.from_document(document, path)

    @classmethod
    def from_document(cls, document: Any, origin: str) -> "Reader":
        if document is None:
            document = {}
        if not isinstance(document, dict):
            raise ConfigError(f"{origin}: top level must be a mapping")
        return cls(_State(origin, document), document, "")

    def path_of(self, key: str) -> str:
        return f"{self._path}.{key}" if self._path else key

    def _message(self, key: str, reason: str) -> str:
        path = self.path_of(key) if key else (self._path or "(top level)")
        return f"{self._state.origin}: {path}: {reason}"

    def fail(self, key: str, reason: str):
        raise ConfigError(self._message(key, reason))

    def warn(self, key: str, reason: str) -> None:
        """合法但多半不是想要的配置：登记警告，不中断读取；Service 在 --check-config 与运行日志中输出。"""
        self._state.warnings.append(self._message(key, reason))

    def warnings(self) -> list[str]:
        return list(self._state.warnings)

    def _child(self, key: str) -> Any:
        if self._node is None:
            return None
        return self._node.get(key)

    def _read(self, key: str, kind: type, check: Optional[Check], secret: bool, fallback: Any) -> Any:
        path = self.path_of(key)
        self._state.leaves.add(path)
        value = self._child(key)
        if value is None:
            result = fallback
        else:
            converted = _convert(value, kind)
            if converted is _MISSING:
                shown = str(value) if isinstance(value, yamlio.Plain) else value
                self.fail(key, f"expected {kind.__name__}, got {shown!r}")
            value = converted
            if check is not None:
                problem = check(value)
                if problem:
                    self.fail(key, problem)
            result = value
        if result is not _MISSING:
            self._state.described.append((path, "***" if secret else str(result)))
        return result

    def get(self, key: str, default: Any, check: Optional[Check] = None) -> Any:
        kind = float if isinstance(default, float) else type(default)
        return self._read(key, kind, check, False, default)

    def required(self, key: str, kind: type = str, check: Optional[Check] = None) -> Any:
        value = self._read(key, kind, check, False, _MISSING)
        if value is _MISSING:
            self.fail(key, "is required")
        return value

    def secret(self, key: str, default: str = "") -> str:
        return self._read(key, str, None, True, default)

    def section(self, key: str) -> "Reader":
        path = self.path_of(key)
        self._state.containers.add(path)
        value = self._child(key)
        if value is not None and not isinstance(value, dict):
            self.fail(key, "expected a mapping")
        return Reader(self._state, value, path)

    def items(self, key: str, min_items: int = 0) -> list["Reader"]:
        path = self.path_of(key)
        self._state.containers.add(path)
        value = self._child(key)
        if value is None:
            value = []
        if not isinstance(value, list):
            self.fail(key, "expected a list")
        readers = []
        for index, item in enumerate(value):
            item_path = f"{path}[{index}]"
            if not isinstance(item, dict):
                raise ConfigError(f"{self._state.origin}: {item_path}: expected a mapping")
            self._state.containers.add(item_path)
            readers.append(Reader(self._state, item, item_path))
        if len(readers) < min_items:
            self.fail(key, f"needs at least {min_items} item(s)")
        return readers

    def finish(self) -> None:
        if self._path:
            raise ConfigError("finish() must be called on the root reader")

        def walk(node: dict, path: str) -> None:
            for key, value in node.items():
                child = f"{path}.{key}" if path else key
                if child in self._state.leaves:
                    continue
                if child in self._state.containers:
                    if isinstance(value, dict):
                        walk(value, child)
                    elif isinstance(value, list):
                        for index, item in enumerate(value):
                            walk(item, f"{child}[{index}]")
                    continue
                raise ConfigError(f"{self._state.origin}: {child}: unknown field")

        walk(self._state.document, "")

    def declared_paths(self) -> list[str]:
        """读取代码声明过的全部叶子路径（包括取了默认值的），按名称排序。"""
        return sorted(self._state.leaves)

    def describe(self) -> str:
        return " ".join(f"{path}={value}" for path, value in self._state.described)
