"""离线安装：先完整预检（preflight），再停服并执行（apply）。"""

from __future__ import annotations

from ..docker import Docker
from .apply import apply
from .preflight import Options, describe, prepare


def install(options: Options, docker: Docker | None = None) -> None:
    docker = docker or Docker()
    if options.ready_timeout <= 0:
        from ..errors import DeployError
        raise DeployError("--ready-timeout must be positive")
    plan = prepare(options, docker)
    if options.dry_run:
        print("\n--dry-run: nothing was stopped or written. Plan:")
        print(describe(plan, docker))
        return
    apply(plan, docker, options.ready_timeout)
