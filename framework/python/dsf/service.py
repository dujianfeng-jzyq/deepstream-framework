"""Python 服务的生命周期，与 C++ 应用遵守同一套部署约定。

    def load(root):                       # 读取并校验配置；不得有副作用
        return root.section("redis").get("host", "127.0.0.1")

    def main(host, service):              # 主循环：用 service.running 与 service.sleep()
        while service.running:
            ...
            service.sleep(1)

    sys.exit(Service("my-service", "说明").run(load, main))

框架负责：命令行（--config、--check-config、--version）、路径（CONTAINER_* 环境变量）、
logging 段与日志格式、SIGINT/SIGTERM、就绪文件（DSF_READY_FILE，默认 /tmp/dsf.ready）。
就绪文件只在 service.sleep()/service.alive() 中更新：主循环停止推进时健康检查随之失败。
watchdog.stall_timeout_s 为正时，主循环超过时限没有推进则进程以失败退出，交给容器的重启策略：
Docker 只在进程退出时重启，不会重启 unhealthy 的容器。
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .config import ConfigError, Reader, at_least, one_of

EXIT_OK, EXIT_FAILURE, EXIT_USAGE = 0, 1, 2

LEVELS = ("trace", "debug", "info", "warn", "error", "critical", "off")
_PYTHON_LEVELS = {"trace": logging.DEBUG, "debug": logging.DEBUG, "info": logging.INFO,
                  "warn": logging.WARNING, "error": logging.ERROR, "critical": logging.CRITICAL,
                  "off": logging.CRITICAL + 10}


@dataclass(frozen=True)
class Paths:
    workspace: str
    config: str
    data: str
    models: str

    @classmethod
    def from_environment(cls) -> "Paths":
        env = os.environ
        return cls(env.get("CONTAINER_WORKSPACE") or "/workspace",
                   env.get("CONTAINER_CONFIG_PATH") or "/config",
                   env.get("CONTAINER_DATA_PATH") or "/data",
                   env.get("CONTAINER_MODELS_PATH") or "/models")


def ready_file() -> Path:
    return Path(os.environ.get("DSF_READY_FILE") or "/tmp/dsf.ready")


def payload_version(paths: Paths) -> str:
    try:
        return Path(paths.workspace, "VERSION").read_text(encoding="utf-8").strip() or "unknown"
    except OSError:
        return "unknown"


class _Formatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):  # noqa: N802 - logging 的接口名
        base = super().formatTime(record, "%Y-%m-%d %H:%M:%S")
        return f"{base}.{int(record.msecs):03d}"


class Service:
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.paths = Paths.from_environment()
        self.log = logging.getLogger(name)
        self._stop = threading.Event()
        self._ready = ready_file()
        self._signal: int | None = None
        self._last_alive = time.monotonic()

    # ---- 主循环使用 ----

    @property
    def running(self) -> bool:
        return not self._stop.is_set()

    def alive(self) -> None:
        """声明主循环仍在推进：更新就绪文件。"""
        self._last_alive = time.monotonic()
        try:
            self._ready.touch()
        except OSError as error:
            self.log.warning("cannot update readiness file %s: %s", self._ready, error)

    def sleep(self, seconds: float) -> bool:
        """可被停止信号打断的等待；返回 True 表示仍在运行。"""
        self.alive()
        return not self._stop.wait(seconds)

    def stop(self) -> None:
        self._stop.set()

    # ---- 入口 ----

    def run(self, load: Callable[[Reader], Any], main: Callable[[Any, "Service"], None],
            argv: list[str] | None = None) -> int:
        parser = argparse.ArgumentParser(prog=self.name, description=self.description)
        parser.add_argument("--config", help=f"配置文件（默认 {self.paths.config}/{self.name}.yaml）")
        parser.add_argument("--check-config", action="store_true", help="校验配置，不连接外部服务、不写文件")
        parser.add_argument("--version", action="store_true", help="输出载荷与框架版本")
        try:
            args = parser.parse_args(argv)
        except SystemExit as exit_request:
            return EXIT_OK if exit_request.code == 0 else EXIT_USAGE
        if args.version:
            print(f"{self.name} payload={payload_version(self.paths)} python={sys.version.split()[0]}")
            return EXIT_OK

        config_path = args.config or f"{self.paths.config}/{self.name}.yaml"
        try:
            root = Reader.load(config_path)
            logging_section = root.section("logging")
            level = logging_section.get("level", "info", one_of(*LEVELS))
            stall_timeout = root.section("watchdog").get("stall_timeout_s", 0, at_least(0))
            config = load(root)
            root.finish()
        except ConfigError as error:
            print(f"{self.name}: {error}", file=sys.stderr)
            return EXIT_FAILURE
        if args.check_config:
            print("configuration is valid")
            return EXIT_OK

        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"))
        logging.basicConfig(level=_PYTHON_LEVELS[level], handlers=[handler], force=True)
        self.log.info("%s payload=%s", self.name, payload_version(self.paths))
        self.log.info("config(%s): %s", config_path, root.describe())

        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, lambda number, _frame: self._on_signal(number))
        self._ready.unlink(missing_ok=True)
        self._last_alive = time.monotonic()
        if stall_timeout > 0:
            threading.Thread(target=self._watch, args=(stall_timeout,), name="watchdog", daemon=True).start()
        try:
            main(config, self)
        except Exception:  # 服务异常退出：记录完整堆栈，交给容器重启策略
            self.log.exception("service failed")
            return EXIT_FAILURE
        finally:
            self._ready.unlink(missing_ok=True)
        if self._signal is not None:
            self.log.info("stopped by signal %d", self._signal)
        else:
            self.log.info("stopped")
        return EXIT_OK

    def _on_signal(self, signum: int) -> None:
        # 信号处理在主线程的任意字节码之间执行：这里写日志可能重入正在进行的 stdout 写入
        # （RuntimeError: reentrant call），只记录信号并置位，日志由主线程在退出时输出。
        self._signal = signum
        self._stop.set()

    def _watch(self, timeout: float) -> None:
        # 主循环卡住时无法自行退出（例如阻塞在没有超时的 IO 上），只能直接结束进程。
        while not self._stop.wait(min(timeout, 1.0)):
            idle = time.monotonic() - self._last_alive
            if idle >= timeout:
                self.log.error("main loop made no progress for %.0f s; exiting so the container restarts", idle)
                logging.shutdown()
                self._ready.unlink(missing_ok=True)
                os._exit(EXIT_FAILURE)
