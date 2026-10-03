"""up -d 之后的有界就绪等待：常驻服务运行且健康检查通过，一次性服务成功退出。"""

from __future__ import annotations

import time
from pathlib import Path

from ..docker import Docker
from ..envfile import DeployError


def _has_healthcheck(definition: dict) -> bool:
    health = definition.get("healthcheck") or {}
    return bool(health) and not health.get("disable") and health.get("test") not in (["NONE"], None)


def wait(docker: Docker, install_dir: Path, timeout: float, poll: float = 1.0) -> None:
    definitions = docker.compose_config(install_dir)["services"]
    expected = {name: d for name, d in definitions.items() if not d.get("profiles")}
    deadline = time.monotonic() + timeout
    while True:
        states = {row.get("Service"): row for row in docker.compose_ps(install_dir)}
        pending = []
        for name, definition in sorted(expected.items()):
            row = states.get(name, {})
            state = row.get("State", "missing")
            health = _has_healthcheck(definition)
            one_shot = definition.get("restart", "no") == "no" and not health
            if state == "exited":
                if one_shot and int(row.get("ExitCode", -1)) == 0:
                    continue
                raise DeployError(f"service {name} exited with {row.get('ExitCode')} before becoming ready; "
                                  f"inspect: docker compose logs {name}")
            if state == "running" and not one_shot and (not health or row.get("Health") == "healthy"):
                continue
            pending.append(name)
        if not pending:
            print("all services are ready")
            return
        if time.monotonic() >= deadline:
            raise DeployError(f"services not ready after {timeout:.0f}s: {', '.join(pending)}; "
                              "inspect: docker compose logs")
        time.sleep(poll)
