"""命令行入口：render / pack / install / init / new-app。由 framework/deploy/dsf-deploy.py 调用。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import pack, project, render
from .envfile import DeployError
from .install import Options, install


def _root(value: str | None) -> Path:
    return Path(value).resolve() if value else Path(__file__).resolve().parents[3]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dsf-deploy", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("render", help="渲染 .env 与 generated/")
    p.add_argument("--root")
    p.add_argument("--profile", help="省略时沿用 .env 中记录的选择")
    p.add_argument("--quiet", action="store_true")

    p = commands.add_parser("pack", help="渲染并生成发布包（镜像须已构建）")
    p.add_argument("--root")
    p.add_argument("--profile")
    p.add_argument("--with-base", action="store_true", help="同时打包 base 镜像（首次部署或 base 变化时）")

    p = commands.add_parser("install", help="在目标机安装发布包")
    p.add_argument("--package", required=True, help="解压后的发布包目录")
    p.add_argument("--install-dir", type=Path)
    p.add_argument("--profile", help="覆盖已记录或包内建议的 profile")
    p.add_argument("--ask", action="store_true", help="重新填写现场参数")
    p.add_argument("--clean-install", action="store_true", help="清空安装目录后安装（数据目录保留）")
    p.add_argument("--dry-run", action="store_true", help="只预检并打印计划，不停服、不写安装目录")
    p.add_argument("--ready-timeout", type=float, default=180.0, help="启动后的就绪等待上限（秒）")

    p = commands.add_parser("init", help="初始化派生项目的名称与数据目录")
    p.add_argument("name")
    p.add_argument("--root")
    p.add_argument("--data-path")
    p.add_argument("--profile", help="初始化后立即渲染该 profile")

    p = commands.add_parser("new-app", help="生成服务单元骨架")
    p.add_argument("name")
    p.add_argument("--kind", choices=("cpp", "python"), default="cpp")
    p.add_argument("--root")

    args = parser.parse_args(argv)
    try:
        if args.command == "render":
            render.render(_root(args.root), args.profile, quiet=args.quiet)
        elif args.command == "pack":
            pack.pack(_root(args.root), args.profile, args.with_base)
        elif args.command == "install":
            install(Options(Path(args.package), args.install_dir, args.profile, args.ask,
                            args.clean_install, args.dry_run, args.ready_timeout))
        elif args.command == "init":
            root = _root(args.root)
            project.init_project(root, args.name, args.data_path)
            if args.profile:
                render.render(root, args.profile)
        elif args.command == "new-app":
            project.new_unit(_root(args.root), args.name, args.kind)
    except DeployError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0
