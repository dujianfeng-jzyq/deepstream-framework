"""安装预检：停旧服务之前完成的全部检查，产出一份可直接执行的安装计划。

顺序不可调换：
  1. 读取 release.env，核对平台与包完整性
  2. 检查安装目标路径，之后才读取旧的 profile 与现场答案
  3. 选择 profile、收集现场答案，检查清洁安装范围与退役文件冲突
  4. 加载镜像（不影响正在运行的旧容器），在临时目录渲染新配置，按最终 Compose
     检查并准备可写挂载，核对镜像平台，用新载荷执行各服务的 --check-config
预检中任何失败都不会停止旧服务或改动安装目录。
"""

from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .. import envfile, inventory, layout, naming, pack, render, site
from ..docker import Docker
from ..envfile import DeployError
from . import mounts, targets


@dataclass
class Options:
    package: Path
    install_dir: Path | None = None
    profile: str | None = None
    ask: bool = False
    clean: bool = False
    dry_run: bool = False
    ready_timeout: float = 180.0
    interactive: bool = field(default_factory=lambda: sys.stdin.isatty())


@dataclass
class Plan:
    package_root: Path
    install_dir: Path
    release: dict[str, str]
    profile: str
    answers: dict[str, str]
    fields: list[site.Field]
    files: dict[str, str]          # 包清单
    obsolete: list[Path]
    configuration: dict[str, bytes]  # 预检过的 .env 与 generated/ 快照
    data_paths: list[Path]
    clean: bool


def default_install_dir(release: dict[str, str]) -> Path:
    root = release.get("INSTALL_ROOT") or os.environ.get("XDG_DATA_HOME") or "~/.local/share"
    return Path(os.path.expanduser(root)) / release["PROJECT_NAME"]


def _choose_profile(options: Options, release: dict[str, str], install_dir: Path,
                    package_root: Path) -> tuple[str, str]:
    if options.profile:
        return options.profile, "from --profile"
    previous = install_dir / layout.ENV
    if previous.is_file() and not options.clean:
        chosen = envfile.read(previous).get("CONFIG_PROFILE", "")
        if chosen:
            return chosen, "kept from the current installation"
    suggested = envfile.require(release, "CONFIG_PROFILE", layout.RELEASE_ENV)
    choices = render.list_profiles(package_root)
    if not options.interactive or not choices:
        return suggested, "suggested by the package"
    print("profiles in this package (press Enter for the suggestion):")
    for index, (name, summary) in enumerate(choices, start=1):
        print(f"  {index}) {name}  {summary}")
    names = [name for name, _ in choices]
    while True:
        try:
            answer = input(f"profile [{suggested}]: ").strip()
        except EOFError:
            return suggested, "suggested by the package"
        if not answer:
            return suggested, "suggested by the package"
        if answer.isdigit() and 1 <= int(answer) <= len(names):
            return names[int(answer) - 1], "chosen here"
        if answer in names:
            return answer, "chosen here"
        print(f"  unknown profile {answer!r}")


def _collect_answers(options: Options, install_dir: Path, package_root: Path):
    """升级沿用已保存的答案（包括显式空答案）；--ask 与 --clean-install 重新收集。
    非交互时只保存已有答案，缺失键留给渲染按 project.env/profile 继承。"""
    fields = site.parse_fields(package_root / layout.SITE_FIELDS)
    fresh = options.ask or options.clean
    answers = {} if fresh else site.read_answers(install_dir / layout.SITE_ENV, fields)
    answers = {k: v for k, v in answers.items() if k in {f.key for f in fields}}
    if options.interactive:
        for item in fields:
            if item.key not in answers:
                answers[item.key] = site.ask(item)
    site.validate(fields, answers)
    return fields, answers


def _check_clean_scope(install_dir: Path, package_root: Path) -> None:
    if not install_dir.exists():
        return
    if install_dir in (Path(install_dir.anchor), Path.home().resolve()):
        raise DeployError(f"refusing to wipe {install_dir}")
    if install_dir == package_root or install_dir in package_root.parents or package_root in install_dir.parents:
        raise DeployError(f"refusing to wipe {install_dir}: it overlaps the package {package_root}")
    if not any((install_dir / marker).exists() for marker in (layout.RELEASE_ENV, layout.ENV, layout.MANAGED)):
        raise DeployError(f"{install_dir} does not look like an installation; remove it by hand if intended")


def _render_in_stage(stage: Path, profile: str, base_image: str, docker: Docker) -> None:
    """优先在宿主渲染；宿主没有 PyYAML 时用 base 镜像（自带 python3-yaml），不挂载荷卷。"""
    import importlib.util
    if importlib.util.find_spec("yaml") is not None:
        render.write_plan(stage, render.build_plan(stage, profile))
        return
    if not docker.image_exists(base_image):
        raise DeployError(f"this host has no PyYAML and the base image {base_image} is missing; "
                          "install python3-yaml or use a package built with --with-base")
    docker.run(["--rm", "--network", "none", "--user", f"{os.getuid()}:{os.getgid()}",
                "-v", f"{stage}:{stage}", "-w", str(stage), "--entrypoint", "python3", base_image,
                str(stage / "framework/deploy/dsf-deploy.py"), "render", "--root", str(stage),
                "--profile", profile, "--quiet"])


def _relocate(services: dict, stage: Path, install_dir: Path) -> None:
    """Compose 在临时目录解析出的 bind 源改写为安装目录中的对应路径。"""
    for service in services.values():
        for mount in service.get("volumes") or []:
            if mount.get("type") != "bind":
                continue
            source = Path(mount["source"])
            if source == stage or stage in source.parents:
                mount["source"] = str(install_dir / source.relative_to(stage))


def _check_configs(services: dict, stage: Path, release: dict[str, str], docker: Docker) -> None:
    """用新载荷离线执行声明了预检标签的服务入口：不联网、只读、不挂数据目录与旧载荷卷。"""
    checks = {name: s for name, s in services.items()
              if (s.get("labels") or {}).get(layout.CONFIG_CHECK_LABEL) == "true"}
    if not checks:
        return
    workspace = layout.MOUNTS["CONTAINER_WORKSPACE"]
    payload = stage / "payload"
    payload.mkdir()
    docker.export_directory(release["APPS_IMAGE"], workspace, payload)
    for name, service in sorted(checks.items()):
        entrypoint = service.get("entrypoint")
        if not isinstance(entrypoint, list) or not entrypoint:
            raise DeployError(f"{name}: {layout.CONFIG_CHECK_LABEL} requires an explicit entrypoint list")
        args = ["--rm", "--network", "none", "--read-only", "--tmpfs", "/tmp",
                "--user", service.get("user") or f"{os.getuid()}:{os.getgid()}",
                "--workdir", service.get("working_dir") or workspace,
                "-v", f"{payload}:{workspace}:ro",
                "-v", f"{stage / layout.GENERATED / 'config'}:{layout.MOUNTS['CONTAINER_CONFIG_PATH']}:ro"]
        for key, value in sorted((service.get("environment") or {}).items()):
            if value is not None:
                args += ["-e", f"{key}={value}"]
        args += ["--entrypoint", entrypoint[0], service["image"], *entrypoint[1:], "--check-config"]
        print(f"checking configuration: {name}")
        docker.run(args)


def prepare(options: Options, docker: Docker) -> Plan:
    package_root = options.package.resolve()
    if not docker.available():
        raise DeployError("docker is required")

    # 1. 发布事实、平台与完整性
    release = envfile.read(package_root / layout.RELEASE_ENV, expand_home=False)
    for key in ("PROJECT_NAME", "TARGET_PLATFORM", "BASE_IMAGE", "APPS_IMAGE"):
        envfile.require(release, key, layout.RELEASE_ENV)
    host = naming.host_platform()
    if release["TARGET_PLATFORM"] != host:
        raise DeployError(f"this package is for {release['TARGET_PLATFORM']}, this machine is {host}")
    files = inventory.verify(package_root)

    # 2. 安装目标
    requested = (options.install_dir or default_install_dir(release)).absolute()
    if requested.is_symlink():
        raise DeployError(f"the installation directory is a symlink: {requested}")
    install_dir = requested.resolve()
    if install_dir == package_root or install_dir in package_root.parents or package_root in install_dir.parents:
        raise DeployError("the installation directory must not overlap the package directory")
    targets.check(package_root, install_dir)

    # 3. 现场选择（读取旧值必须在任何删除之前）
    profile, source = _choose_profile(options, release, install_dir, package_root)
    fields, answers = _collect_answers(options, install_dir, package_root)
    print(f"install to:  {install_dir}")
    print(f"profile:     {profile} ({source})")
    if answers:
        print(f"site:        {site.masked(fields, answers)}")
    if options.clean:
        _check_clean_scope(install_dir, package_root)
        print(f"clean install: {install_dir} will be wiped (data directories and images are kept)")
    obsolete = [] if options.clean else inventory.obsolete(install_dir, inventory.managed(files))

    # 4. 镜像、配置与挂载
    for archive in sorted((package_root / "images").glob("*.tar")):
        print(f"loading {archive.name}")
        docker.load(archive)
    with tempfile.TemporaryDirectory(prefix="dsf-install-") as temporary:
        stage = Path(temporary).resolve()
        pack.stage_deployment_files(package_root, stage)
        # 在宿主展开 ~：容器渲染时 HOME 可能不是安装用户的家目录。
        envfile.write(stage / layout.PROJECT_ENV, envfile.read(package_root / layout.RELEASE_ENV))
        envfile.write(stage / layout.SITE_ENV, answers)
        _render_in_stage(stage, profile, release["BASE_IMAGE"], docker)
        compose_files = envfile.read(stage / layout.ENV)["COMPOSE_FILE"].split(":")
        services = docker.compose_config(stage, env_file=stage / layout.ENV,
                                         files=[stage / name for name in compose_files])["services"]
        _relocate(services, stage, install_dir)
        data_paths = mounts.prepare(services, install_dir, package_root)
        for name, service in sorted(services.items()):
            image = service.get("image")
            if image and docker.image_platform(image) != release["TARGET_PLATFORM"]:
                raise DeployError(f"{name}: image {image} is missing or not {release['TARGET_PLATFORM']}")
        _check_configs(services, stage, release, docker)
        configuration = {layout.ENV: (stage / layout.ENV).read_bytes()}
        for path in sorted((stage / layout.GENERATED).rglob("*")):
            if path.is_file():
                configuration[path.relative_to(stage).as_posix()] = path.read_bytes()
    targets.check(package_root, install_dir, configuration)
    return Plan(package_root, install_dir, release, profile, answers, fields, files, obsolete,
                configuration, data_paths, options.clean)


def describe(plan: Plan, docker: Docker) -> str:
    lines = [f"installation directory: {plan.install_dir}"]
    if (plan.install_dir / layout.ENV).is_file():
        lines.append("stop: current services (docker compose down -v) using the installed definition")
    if plan.clean:
        lines.append(f"wipe: {plan.install_dir}")
    lines += [f"remove obsolete: {path}" for path in plan.obsolete]
    lines.append(f"write: {len(inventory.managed(plan.files))} release files, "
                 f"{len(plan.configuration)} configuration files, {layout.SITE_ENV}")
    lines += [f"writable mount: {path}" for path in plan.data_paths]
    volume = f"{plan.release['PROJECT_NAME']}_{layout.PAYLOAD_VOLUME}"
    if not (plan.install_dir / layout.ENV).is_file() and docker.volume_exists(volume):
        lines.append(f"warning: stale payload volume {volume} exists and would not be refreshed; "
                     f"remove it with: docker volume rm {volume}")
    lines.append("start: docker compose up -d, then wait for readiness")
    return "\n".join(lines)

