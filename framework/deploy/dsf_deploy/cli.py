"""命令行入口：render / profile-variables / check-models / pack / pack-models / install / install-models / init / new-app。由 framework/deploy/dsf-deploy.py 调用。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import modelpack, models, pack, project, render
from .errors import DeployError
from .install import Options, install
from .install.models import ModelOptions, install_models


def _root(value: str | None) -> Path:
    return Path(value).resolve() if value else Path(__file__).resolve().parents[3]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dsf-deploy", description="Render, pack and install DeepStream projects.")
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("render", help="render .env and generated/")
    p.add_argument("--root")
    p.add_argument("--profile", help="defaults to the profile recorded in .env")
    p.add_argument("--quiet", action="store_true")

    p = commands.add_parser("profile-variables", help="print the x-build variables of a profile as JSON (used by the installer)")
    p.add_argument("--root")
    p.add_argument("--profile", required=True)

    p = commands.add_parser("check-models", help="check models/ (spec, release record, artifacts, nvinfer configs)")
    p.add_argument("--root")

    p = commands.add_parser("pack", help="render and create a release package (the images must be built)")
    p.add_argument("--root")
    p.add_argument("--profile")
    p.add_argument("--with-base", action="store_true", help="include the base image (first deployment or when base changes)")

    p = commands.add_parser("pack-models", help="create a model package (only models/<id>/, no images or programs)")
    p.add_argument("models", nargs="+", metavar="MODEL")
    p.add_argument("--root")

    p = commands.add_parser("install", help="在目标机安装发布包")
    p.add_argument("--package", required=True, help="解压后的发布包目录")
    p.add_argument("--install-dir", type=Path, help="默认为 INSTALL_ROOT、XDG_DATA_HOME 或 ~/.local/share 下的项目名目录")
    p.add_argument("--profile", help="覆盖已记录或包内建议的 profile")
    p.add_argument("--ask", action="store_true", help="逐项重新填写现场参数，默认值为当前答案（需要终端）")
    p.add_argument("--clean-install", action="store_true", help="清空安装目录后安装（数据目录保留）")
    p.add_argument("--dry-run", action="store_true", help="只预检并打印计划，不停服、不写安装目录")
    p.add_argument("--ready-timeout", type=float, default=180.0, help="启动后的就绪等待上限（秒）")

    p = commands.add_parser("install-models", help="在已安装的项目中更新模型：停服前构建 engine，只重启引用模型的服务")
    p.add_argument("--package", required=True, help="解压后的模型包目录")
    p.add_argument("--install-dir", type=Path, help="默认为 INSTALL_ROOT、XDG_DATA_HOME 或 ~/.local/share 下的项目名目录")
    p.add_argument("--rollback", action="store_true", help="撤销这个模型包的安装：换回上一份模型目录")
    p.add_argument("--dry-run", action="store_true", help="只预检并打印计划，不构建、不切换、不重启")
    p.add_argument("--ready-timeout", type=float, default=180.0, help="重启后的就绪等待上限（秒）")

    p = commands.add_parser("init", help="set the name of a derived project")
    p.add_argument("name")
    p.add_argument("--root")
    p.add_argument("--profile", help="render this profile right after initializing")

    p = commands.add_parser("new-app", help="create a service unit skeleton")
    p.add_argument("name")
    p.add_argument("--kind", choices=("cpp", "python"), default="cpp")
    p.add_argument("--root")

    args = parser.parse_args(argv)
    try:
        if args.command == "render":
            render.render(_root(args.root), args.profile, quiet=args.quiet)
        elif args.command == "profile-variables":
            print(json.dumps(render.profile_variables(_root(args.root), args.profile)))
        elif args.command == "check-models":
            models.check(_root(args.root))
            print("models ok")
        elif args.command == "pack":
            pack.pack(_root(args.root), args.profile, args.with_base)
        elif args.command == "install":
            install(Options(package=Path(args.package), install_dir=args.install_dir, profile=args.profile,
                            ask=args.ask, clean=args.clean_install, dry_run=args.dry_run,
                            ready_timeout=args.ready_timeout))
        elif args.command == "pack-models":
            modelpack.pack_models(_root(args.root), args.models)
        elif args.command == "install-models":
            install_models(ModelOptions(package=Path(args.package), install_dir=args.install_dir,
                                        rollback=args.rollback, dry_run=args.dry_run,
                                        ready_timeout=args.ready_timeout))
        elif args.command == "init":
            root = _root(args.root)
            project.init_project(root, args.name)
            if args.profile:
                render.render(root, args.profile)
        elif args.command == "new-app":
            project.new_unit(_root(args.root), args.name, args.kind)
    except DeployError as error:
        # 安装器面向现场，用中文；其他命令面向开发者。安装器在 base 中调用 render 等命令时按 "error: " 剥离前缀。
        print(f"{'错误：' if args.command in ('install', 'install-models') else 'error: '}{error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("已中断" if args.command in ("install", "install-models") else "interrupted", file=sys.stderr)  # 安装器在停服之后被中断时已另行说明现场状态
        return 130
    return 0
