"""执行已预检的安装计划。从这里开始会停止旧服务；失败不自动回滚，可重新安装旧包恢复。
停服之后到启动之前的任何失败或中断（包括 Ctrl-C）都会说明现场状态与恢复办法，再原样抛出；
就绪等待中的中断只结束等待，服务已经启动。

顺序：重验目标 → 用旧定义停服 → 按需清空目录 → 删除退役文件 → 复制发布文件 → 删除因此变空的目录 →
写入 project.env、site.env、受管理清单与预检过的配置快照 → 启动 → 清理旧载荷卷 → 等待就绪。
载荷卷按载荷标识命名，新载荷总是播种到新卷，停服后不需要删卷；旧卷在启动之后才不再被引用。
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from .. import envfile, inventory, layout, site
from ..docker import Docker
from ..errors import DeployError
from . import ready, targets
from .preflight import Plan


def apply(plan: Plan, docker: Docker, ready_timeout: float) -> None:
    root = plan.install_dir
    targets.check(plan.package_root, root, plan.configuration)  # 预检可能耗时：停服前再确认一次
    upgrading = (root / layout.ENV).is_file()
    # 旧工程名在写入新 .env 之前读取：改名后旧工程的载荷卷同样要清理。
    projects = {plan.release["PROJECT_NAME"]}
    if upgrading:
        projects.add(envfile.read_compose(root / layout.ENV).get("PROJECT_NAME") or plan.release["PROJECT_NAME"])
    try:
        _replace(plan, docker)
        docker.compose(root, ["config", "--quiet"])
        print("启动服务")
        docker.compose(root, ["up", "-d"])
    except BaseException:
        print(incomplete_message(root, upgrading), file=sys.stderr)
        raise
    _prune_payload_volumes(plan, sorted(projects), docker)
    try:
        ready.wait(docker, root, ready_timeout)
    except KeyboardInterrupt:
        print(f"\n已停止等待：服务已安装并继续运行\n"
              f"  查看：cd {root} && docker compose ps", file=sys.stderr)
        raise
    print(f"\n安装完成：{root}（profile {plan.profile}）")
    print(f"查看日志：cd {root} && docker compose logs -f")
    print("回滚方法：重新安装上一个版本的发布包")


def incomplete_message(root: Path, upgrading: bool) -> str:
    state = "旧服务已停止（或正在停止），" if upgrading else ""
    return (f"\n安装未完成：{state}{root} 可能只写入了一部分。\n"
            "  继续完成：用同一个发布包和参数重新运行安装器\n"
            "  退回旧版：安装上一个版本的发布包")


def _prune_payload_volumes(plan: Plan, projects: list[str], docker: Docker) -> None:
    """删除这些工程中不是本次载荷的载荷卷（旧版本、旧的固定卷名、开发卷）。回滚时重新播种即可。
    仍被容器使用的卷 docker 拒绝删除，跳过；清理失败不影响安装结果。"""
    current = layout.payload_volume(plan.release["PROJECT_NAME"], plan.release[layout.PAYLOAD_ID_KEY])
    try:
        stale = [name for project in projects
                 for name in docker.project_volumes(project, layout.PAYLOAD_VOLUME) if name != current]
    except DeployError as error:
        print(f"提示：没有清理旧载荷卷：{error}", file=sys.stderr)
        return
    for name in stale:
        try:
            docker.volume_remove(name)
            print(f"已删除旧载荷卷 {name}")
        except DeployError:
            print(f"提示：载荷卷 {name} 仍在使用，没有删除", file=sys.stderr)


def _remove_empty_directories(root: Path, retired: list[Path]) -> None:
    """删除因退役文件而变空的目录：从每个退役文件所在目录向上，到第一个非空目录或安装根为止。
    在复制新版本文件之后执行，新版本仍使用的目录不会被删除再重建；含现场文件的目录保留。"""
    for directory in sorted({path.parent for path in retired}, key=lambda path: len(path.parts), reverse=True):
        while directory != root and root in directory.parents:
            if directory.is_dir() and not directory.is_symlink():
                if any(directory.iterdir()):
                    break
                directory.rmdir()
                print(f"已删除空目录 {directory.relative_to(root)}")
            directory = directory.parent  # 已被现场删除的目录：继续检查上一级


def _replace(plan: Plan, docker: Docker) -> None:
    """停旧服务并写入新版本的全部文件。"""
    root, package = plan.install_dir, plan.package_root
    if (root / layout.ENV).is_file():
        print("停止当前服务")
        # 不用 down -v：它删除项目的全部命名卷，服务放在命名卷中的持久数据也会丢失。
        docker.compose(root, ["down", "--remove-orphans"])

    if plan.clean and root.exists():
        print(f"清空 {root}")
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    for path in plan.obsolete:
        path.unlink(missing_ok=True)  # 预检之后可能已被现场删除；停服之后不能因此中断
        print(f"已删除退役文件 {path.relative_to(root)}")

    managed = inventory.managed(plan.files)
    for name in sorted(managed):
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(package / name, destination)
    _remove_empty_directories(root, plan.obsolete)
    # 目标机的 project.env 由发布事实派生；镜像名由包决定，不按目标机 profile 重新派生。
    envfile.write(root / layout.PROJECT_ENV, envfile.read(package / layout.RELEASE_ENV, expand_home=False),
                  ("GENERATED from release.env by the installer — DO NOT EDIT",))
    site.write_answers(root / layout.SITE_ENV, plan.answers,
                       ("现场答案：修改后重新运行安装器生效",), mode=0o600)
    (root / layout.MANAGED).write_text(json.dumps(managed, indent=2) + "\n", encoding="utf-8")

    # generated/ 整体替换；rmtree 删除其中的旧链接本身，不跟随链接。
    if (root / layout.GENERATED).exists():
        shutil.rmtree(root / layout.GENERATED)
    for name, content in plan.configuration.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        path.chmod(0o600)
