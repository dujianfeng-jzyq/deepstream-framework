"""执行已预检的安装计划。从这里开始会停止旧服务；失败不自动回滚，可重新安装旧包恢复。

顺序：重验目标 → 用旧定义停服并删除载荷卷（使其重新播种）→ 按需清空目录 → 删除退役文件 →
复制发布文件 → 写入 project.env、site.env、受管理清单与预检过的配置快照 → 启动并等待就绪。
"""

from __future__ import annotations

import json
import shutil
import sys

from .. import envfile, inventory, layout
from ..docker import Docker
from . import ready, targets
from .preflight import Plan


def apply(plan: Plan, docker: Docker, ready_timeout: float) -> None:
    root, package = plan.install_dir, plan.package_root
    targets.check(package, root, plan.configuration)  # 预检可能耗时：停服前再确认一次

    new_volume = layout.payload_volume(plan.release["PROJECT_NAME"])
    stale_check = True
    if (root / layout.ENV).is_file():
        print("stopping the current services")
        # 不用 down -v：它删除项目的全部命名卷，服务放在命名卷中的持久数据也会丢失。
        # 只删载荷卷：卷只在为空时播种，不删就会继续运行旧载荷。
        docker.compose(root, ["down", "--remove-orphans"])
        old_volume = layout.payload_volume(envfile.read(root / layout.ENV).get("PROJECT_NAME")
                                           or plan.release["PROJECT_NAME"])
        if docker.volume_exists(old_volume):
            print(f"removing the payload volume {old_volume}")
            docker.volume_remove(old_volume)
        stale_check = old_volume != new_volume
    if stale_check and docker.volume_exists(new_volume):
        print(f"WARNING: a stale payload volume exists and will not be refreshed: {new_volume}\n"
              f"         remove it with: docker volume rm {new_volume}", file=sys.stderr)

    if plan.clean and root.exists():
        print(f"wiping {root}")
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    for path in plan.obsolete:
        path.unlink(missing_ok=True)  # 预检之后可能已被现场删除；停服之后不能因此中断
        print(f"removed obsolete file {path.relative_to(root)}")

    managed = inventory.managed(plan.files)
    for name in sorted(managed):
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(package / name, destination)
    # 目标机的 project.env 由发布事实派生；镜像名由包决定，不按目标机 profile 重新派生。
    envfile.write(root / layout.PROJECT_ENV, envfile.read(package / layout.RELEASE_ENV, expand_home=False),
                  ("GENERATED from release.env by the installer — DO NOT EDIT",))
    envfile.write(root / layout.SITE_ENV, plan.answers,
                  ("site answers — edit and rerun the installer to apply",), mode=0o600)
    (root / layout.MANAGED).write_text(json.dumps(managed, indent=2) + "\n", encoding="utf-8")

    # generated/ 整体替换；rmtree 删除其中的旧链接本身，不跟随链接。
    if (root / layout.GENERATED).exists():
        shutil.rmtree(root / layout.GENERATED)
    for name, content in plan.configuration.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        path.chmod(0o600)

    docker.compose(root, ["config", "--quiet"])
    print("starting services")
    docker.compose(root, ["up", "-d"])
    ready.wait(docker, root, ready_timeout)
    print(f"\ninstalled: {root} (profile {plan.profile})")
    print(f"logs:      cd {root} && docker compose logs -f")
    print("rollback:  install the previous release package again")
