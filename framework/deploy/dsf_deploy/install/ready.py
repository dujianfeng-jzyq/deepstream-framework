"""up -d 之后的有界就绪等待：常驻服务运行且健康检查通过，一次性服务成功退出。"""

from __future__ import annotations

import re
import time
from pathlib import Path

from ..docker import Docker
from ..envfile import DeployError

LOG_TAIL = 200          # 失败时读取每个未就绪服务的最近日志行数
NOTABLE_LINES = 3       # 从中摘出的告警与错误条数
MAX_CONTINUATION = 3    # 一条记录最多合并的后续行
NOTABLE = re.compile(r"\b(warn|warning|error|critical|fatal)\b", re.IGNORECASE)


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
                raise DeployError(f"service {name} exited with {row.get('ExitCode')} before becoming ready\n"
                                  f"{_describe(docker, install_dir, name, row)}\n"
                                  f"  inspect: docker compose logs {name}")
            if state == "running" and not one_shot and (not health or row.get("Health") == "healthy"):
                continue
            pending.append(name)
        if not pending:
            print("all services are ready")
            return
        if time.monotonic() >= deadline:
            details = "\n".join(_describe(docker, install_dir, name, states.get(name, {})) for name in pending)
            raise DeployError(f"services not ready after {timeout:.0f}s: {', '.join(pending)}\n{details}\n"
                              "  readiness can depend on input reaching a service, for example a camera that "
                              "is not streaming yet\n"
                              "  inspect: docker compose logs <service>")
        time.sleep(poll)


def _describe(docker: Docker, install_dir: Path, name: str, row: dict) -> str:
    """服务的状态与最近的几条告警/错误日志：只报“未就绪”时现场看不出原因（例如相机没有推流）。"""
    health = row.get("Health") or ""
    lines = [f"  {name}: {row.get('State', 'missing')}{f' ({health})' if health else ''}"]
    try:
        text = docker.compose_logs(install_dir, name, LOG_TAIL)
    except DeployError:
        text = ""  # 读不到日志不应掩盖原本的失败
    notable = [record for record in _records(text) if NOTABLE.search(record)]
    lines += [f"    {record[:300]}" for record in notable[-NOTABLE_LINES:]]
    return "\n".join(lines)


def _records(text: str) -> list[str]:
    """把日志行合并成记录：括号未闭合的行与随后的行属于同一条（GStreamer 的错误详情常换行，
    例如 RTSP 的 “Not Found (404)” 在第二行）。最多合并 MAX_CONTINUATION 行。"""
    records: list[str] = []
    open_count = continued = 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if records and open_count > 0 and continued < MAX_CONTINUATION:
            records[-1] += " " + line
            continued += 1
        else:
            records.append(line)
            open_count = continued = 0
        open_count += line.count("(") - line.count(")")
    return records
