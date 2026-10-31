"""安装预检：停旧服务之前完成的全部检查，产出一份可直接执行的安装计划。

顺序不可调换：
  1. 读取 release.env，核对平台与包完整性
  2. 检查安装目标路径与同名工程是否已由其他目录启动，之后才读取旧的 profile 与现场答案
  3. 选择 profile、收集现场答案（提问默认值含该 profile 的 x-build，在 base 中读取，
     需要时为此提前加载包内的 base），检查清洁安装范围与退役文件冲突
  4. 加载镜像（不影响正在运行的旧容器），在临时目录渲染新配置，按最终 Compose
     检查并准备可写挂载，核对镜像平台，用新载荷执行各服务的 --check-config
预检中任何失败都不会停止旧服务或改动安装目录。
"""

from __future__ import annotations

import functools
import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .. import envfile, inventory, layout, naming, pack, render, site
from ..docker import Docker
from ..errors import DeployError
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
    platform = release.get("TARGET_PLATFORM", "")
    fitting = [name for name, _ in render.fitting_profiles(package_root, platform)]
    if options.profile:
        if options.profile in [name for name, _ in render.list_profiles(package_root)] and \
                options.profile not in fitting:
            raise DeployError(f"profile {options.profile} 面向的平台不是本发布包的 {platform}；"
                              f"适用于 {platform} 的 profile：{', '.join(fitting) or '无'}")
        return options.profile, "来自 --profile"
    previous = install_dir / layout.ENV
    if previous.is_file() and not options.clean:
        chosen = envfile.read_compose(previous).get("CONFIG_PROFILE", "")
        if chosen in fitting:
            return chosen, "沿用当前安装"
        if chosen:
            # 例如新版本改了 profile 名：不沿用，按新包重新选择（非交互时用包内建议）。
            print(f"提示：本发布包没有为 {platform} 提供已安装的 profile {chosen}，重新选择")
    suggested = envfile.require(release, "CONFIG_PROFILE", layout.RELEASE_ENV)
    choices = render.fitting_profiles(package_root, platform)
    if not options.interactive or not choices:
        return suggested, "发布包建议"
    # 与现场参数的提问同一种排版（site.ask）：标题行、缩进的选项、单独的输入行。
    names = [name for name, _ in choices]
    width = max(len(name) for name in names)
    print(f"\nprofile [{suggested}]（输入编号或名称；回车接受 [ ] 中的值）")
    for index, (name, summary) in enumerate(choices, start=1):
        print(f"  {index}) {name:<{width}}  {summary}".rstrip())
    while True:
        try:
            answer = input("> ").strip()
        except EOFError:
            return suggested, "发布包建议"
        if not answer:
            return suggested, "发布包建议"
        if answer.isdigit() and 1 <= int(answer) <= len(names):
            return names[int(answer) - 1], "本次选择"
        if answer in names:
            return answer, "本次选择"
        print(f"  没有 {answer!r} 这个 profile，请输入列表中的编号或名称")


def _collect_answers(options: Options, install_dir: Path, package_root: Path,
                     inherited: Callable[[], dict[str, str]],
                     previous: Callable[[], dict[str, str]] | None = None):
    """升级沿用已保存的答案（包括显式空答案）；--clean-install 重新收集。
    --ask 逐项重新提问，默认值是已保存的答案：只想改一项时其余回车即可，口令不会因回车被清空。
    非交互时只保存已有答案，缺失键留给渲染按 project.env/profile 继承。
    没有已保存答案时，提问的默认值与渲染的继承值一致：回车保存的答案不会遮住所选 profile 的设置。
    previous 是改换 profile 前的继承值：等于旧默认值的答案视为当时回车接受的默认值，
    新 profile 的默认值不同时丢弃，改为重新提问或继承；现场输入的其他值保留。"""
    fields = site.parse_fields(package_root / layout.SITE_FIELDS)
    if options.ask and not options.interactive:
        raise DeployError(f"--ask 需要终端来逐项提问；请改为编辑 {install_dir / layout.SITE_ENV} "
                          "后重新运行安装器")
    retired: list[str] = []
    answers = {} if options.clean else site.read_answers(install_dir / layout.SITE_ENV, fields, retired)
    if retired:
        print(f"已去掉本版本不再使用的现场参数：{', '.join(retired)}")
    if answers and previous is not None:
        old, new = site.defaults(fields, previous()), site.defaults(fields, inherited())
        stale = sorted(key for key, value in answers.items() if value == old[key] != new[key])
        for key in stale:
            del answers[key]
        if stale:
            print(f"profile 已更换：{', '.join(stale)} 改用新 profile 的值（保存的是旧 profile 的默认值）")
    if not options.interactive:
        pending = []
    elif options.ask:
        pending = fields
    else:
        pending = [item for item in fields if item.key not in answers]
    if pending:
        suggested = site.defaults(fields, inherited())
        print(f"\n现场参数共 {len(pending)} 项，保存到 {install_dir / layout.SITE_ENV}；回车接受 [ ] 中的值")
        for index, item in enumerate(pending, start=1):
            answers[item.key] = site.ask(item, answers.get(item.key, suggested[item.key]),
                                         position=f"[{index}/{len(pending)}]")
        print()
    site.validate(fields, answers)
    return fields, answers


def _check_clean_scope(install_dir: Path, package_root: Path) -> None:
    if not install_dir.exists():
        return
    if install_dir in (Path(install_dir.anchor), Path.home().resolve()):
        raise DeployError(f"拒绝清空 {install_dir}")
    if install_dir == package_root or install_dir in package_root.parents or package_root in install_dir.parents:
        raise DeployError(f"拒绝清空 {install_dir}：它与发布包 {package_root} 重叠")
    if not any((install_dir / marker).exists() for marker in (layout.RELEASE_ENV, layout.ENV, layout.MANAGED)):
        raise DeployError(f"{install_dir} 看起来不是安装目录；确实要清空请手工删除")


def _load_images(package_root: Path, docker: Docker, loaded: set[str], only: str | None = None) -> None:
    """加载包内镜像（不影响正在运行的旧容器）；only 只加载该镜像，已加载的归档不重复加载。"""
    archives = sorted((package_root / "images").glob("*.tar"))
    if only is not None:
        archives = [a for a in archives if a.name == naming.image_tar_name(only)]
    for archive in archives:
        if archive.name not in loaded:
            print(f"加载镜像 {archive.name}")
            docker.load(archive)
            loaded.add(archive.name)


def _run_deploy_tool(root: Path, args: list[str], base_image: str, docker: Docker, *, writable: bool,
                     package_root: Path, loaded: set[str]) -> str:
    """在 base 镜像（自带 python3-yaml）中运行需要 PyYAML 的部署命令。安装宿主因此只需要标准库，
    渲染结果也不随宿主的 Python 与 PyYAML 版本变化：每个现场使用同一套解释器与库。
    只挂载 root，不挂载载荷卷；HOME 取宿主用户的家目录，渲染展开 ~ 时与宿主一致。"""
    if not docker.image_exists(base_image):
        _load_images(package_root, docker, loaded, only=base_image)
    if not docker.image_exists(base_image):
        raise DeployError(f"缺少 base 镜像 {base_image}，渲染配置与运行服务都需要它：请使用带 --with-base "
                          "打包的发布包，或先加载 base 镜像")
    try:
        result = docker.run(["--rm", "--network", "none", "--user", f"{os.getuid()}:{os.getgid()}",
                             "-e", f"HOME={Path.home()}",
                             "-v", f"{root}:{root}{'' if writable else ':ro'}", "-w", str(root),
                             "--entrypoint", "python3", base_image, str(root / "framework/deploy/dsf-deploy.py"),
                             *args, "--root", str(root)], capture=True)
    except DeployError as error:
        # 用户要看的是部署工具自己的错误（例如模板中未定义的变量），不是 docker run 的命令行。
        _, _, detail = str(error).partition("\n")
        raise DeployError(detail.strip().removeprefix("error: ") or str(error)) from error
    if result.stderr:
        sys.stderr.write(result.stderr)  # 渲染的提示（note: ...）
    return result.stdout


def _profile_variables(root: Path, profile: str, base_image: str, docker: Docker, package_root: Path,
                       loaded: set[str]) -> dict[str, str]:
    """root 中 profile 的 x-build；root 是发布包或旧安装目录。"""
    output = _run_deploy_tool(root, ["profile-variables", "--profile", profile], base_image, docker,
                              writable=False, package_root=package_root, loaded=loaded)
    return json.loads(output)


def _previous_inherited(options: Options, install_dir: Path, profile: str, base_image: str, docker: Docker,
                        package_root: Path, loaded: set[str]) -> Callable[[], dict[str, str]] | None:
    """升级时改换了 profile：返回旧安装的继承值（旧 project.env 与旧 profile）；否则 None。"""
    if options.ask or options.clean or not (install_dir / layout.ENV).is_file():
        return None
    old = envfile.read_compose(install_dir / layout.ENV).get("CONFIG_PROFILE", "")
    if not old or old == profile or not (install_dir / layout.PROFILES / f"{old}.yaml").is_file():
        return None
    project = envfile.read(install_dir / layout.PROJECT_ENV, expand_home=False)
    return lambda: {**project, **_profile_variables(install_dir, old, base_image, docker, package_root, loaded)}


def _render_in_stage(stage: Path, profile: str, base_image: str, docker: Docker, package_root: Path,
                     loaded: set[str]) -> None:
    """在 base 镜像中渲染（见 _run_deploy_tool）。"""
    _run_deploy_tool(stage, ["render", "--profile", profile, "--quiet"], base_image, docker,
                     writable=True, package_root=package_root, loaded=loaded)


def _relocate(services: dict, stage: Path, install_dir: Path) -> None:
    """Compose 在临时目录解析出的 bind 源改写为安装目录中的对应路径。"""
    for service in services.values():
        for mount in service.get("volumes") or []:
            if mount.get("type") != "bind":
                continue
            source = Path(mount["source"])
            if source == stage or stage in source.parents:
                mount["source"] = str(install_dir / source.relative_to(stage))


def _docker_env_file(service: str, environment: dict[str, str]) -> str:
    """docker run --env-file 的格式：每行 KEY=VALUE，值按字面到行尾，不能包含换行。"""
    lines = []
    for key, value in sorted(environment.items()):
        if "\n" in value or "\r" in value:
            raise DeployError(f"{service}：环境变量 {key} 含换行，无法传给 --check-config")
        lines.append(f"{key}={value}")
    return "\n".join(lines) + "\n"


def _check_configs(services: dict, stage: Path, release: dict[str, str], docker: Docker) -> None:
    """用新载荷离线执行声明了预检标签的服务入口：不联网、只读、不挂数据目录与旧载荷卷。"""
    checks = {name: s for name, s in services.items()
              if (s.get("labels") or {}).get(layout.CONFIG_CHECK_LABEL) == "true"}
    if not checks:
        return
    workspace = layout.MOUNTS["CONTAINER_WORKSPACE"]
    payload = stage / "payload"
    payload.mkdir()
    environments = stage / "check-env"
    environments.mkdir(mode=0o700)
    docker.export_directory(release["APPS_IMAGE"], workspace, payload)
    for name, service in sorted(checks.items()):
        entrypoint = service.get("entrypoint")
        if not isinstance(entrypoint, list) or not entrypoint:
            raise DeployError(f"{name}：{layout.CONFIG_CHECK_LABEL} 要求显式的 entrypoint 列表")
        args = ["--rm", "--network", "none", "--read-only", "--tmpfs", "/tmp",
                "--user", service.get("user") or f"{os.getuid()}:{os.getgid()}",
                "--workdir", service.get("working_dir") or workspace,
                "-v", f"{payload}:{workspace}:ro",
                "-v", f"{stage / layout.GENERATED / 'config'}:{layout.MOUNTS['CONTAINER_CONFIG_PATH']}:ro"]
        # 环境变量经 0600 文件传入：-e KEY=VALUE 会让其中的口令出现在宿主的进程列表中。
        environment = {key: str(value) for key, value in (service.get("environment") or {}).items()
                       if value is not None}
        if environment:
            env_file = environments / f"{name}.env"
            env_file.touch(mode=0o600)
            env_file.write_text(_docker_env_file(name, environment), encoding="utf-8")
            args += ["--env-file", str(env_file)]
        args += ["--entrypoint", entrypoint[0], service["image"], *entrypoint[1:], "--check-config"]
        print(f"检查配置：{name}")
        docker.run(args)


def _check_other_directories(project: str, install_dir: Path, docker: Docker) -> None:
    """同名 Compose 工程的容器由其他目录启动（例如开发仓库）时，在这里 up 会接管并重建它们，
    而那个目录的服务会被悄悄替换。只允许本安装目录启动的容器。"""
    others = sorted({directory for _, directory in docker.project_containers(project)
                     if not directory or Path(os.path.realpath(directory)) != install_dir})
    if others:
        # 不知道启动目录时（容器缺少 Compose 的标签）只能按工程名停止，那个目录的 Compose 文件不参与。
        commands = [f"cd {directory} && docker compose down" if directory else f"docker compose -p {project} down"
                    for directory in others]
        raise DeployError(f"Compose 工程 {project} 已有从 {', '.join(d or '未知目录' for d in others)} "
                          "启动的容器，在这里安装会接管它们。\n"
                          + "\n".join(f"  先停止它们：{command}" for command in commands))


def prepare(options: Options, docker: Docker) -> Plan:
    package_root = options.package.resolve()
    if not docker.available():
        raise DeployError("需要 docker")

    # 1. 发布事实、平台与完整性
    release = envfile.read(package_root / layout.RELEASE_ENV, expand_home=False)
    for key in ("PROJECT_NAME", "TARGET_PLATFORM", "BASE_IMAGE", "APPS_IMAGE", layout.PAYLOAD_ID_KEY):
        envfile.require(release, key, layout.RELEASE_ENV)
    host = naming.host_platform()
    if release["TARGET_PLATFORM"] != host:
        raise DeployError(f"这个发布包面向 {release['TARGET_PLATFORM']}，本机是 {host}")
    files = inventory.verify(package_root)

    # 2. 安装目标
    requested = (options.install_dir or default_install_dir(release)).absolute()
    if requested.is_symlink():
        raise DeployError(f"安装目录是符号链接：{requested}")
    install_dir = requested.resolve()
    if install_dir == package_root or install_dir in package_root.parents or package_root in install_dir.parents:
        raise DeployError("安装目录不能与发布包目录重叠")
    targets.check(package_root, install_dir)
    _check_other_directories(release["PROJECT_NAME"], install_dir, docker)

    # 3. 现场选择（读取旧值必须在任何删除之前）
    profile, source = _choose_profile(options, release, install_dir, package_root)
    loaded: set[str] = set()
    base_image = release["BASE_IMAGE"]
    inherited = functools.cache(lambda: {
        **release, **_profile_variables(package_root, profile, base_image, docker, package_root, loaded)})
    previous = _previous_inherited(options, install_dir, profile, base_image, docker, package_root, loaded)
    fields, answers = _collect_answers(options, install_dir, package_root, inherited, previous)
    print(f"安装目录：{install_dir}")
    print(f"profile： {profile}（{source}）")
    if answers:
        print("现场参数：")
        for line in site.masked(fields, answers):
            print(f"  {line}")
    if options.clean:
        _check_clean_scope(install_dir, package_root)
        print(f"清洁安装：{install_dir} 将被清空（数据目录与镜像保留）")
    obsolete = [] if options.clean else inventory.obsolete(install_dir, inventory.managed(files))

    # 4. 镜像、配置与挂载
    _load_images(package_root, docker, loaded)
    # 卷名按发布包记录的载荷标识：本机同名 apps 镜像（例如未重新加载的旧构建）不是这份载荷时，
    # 播种会把另一份程序放进以这个标识命名的卷。
    actual = pack.payload_id_of(docker, release["APPS_IMAGE"])
    if actual != release[layout.PAYLOAD_ID_KEY]:
        raise DeployError(f"镜像 {release['APPS_IMAGE']} 的载荷标识是 {actual}，"
                          f"发布包记录的是 {release[layout.PAYLOAD_ID_KEY]}")
    with tempfile.TemporaryDirectory(prefix="dsf-install-") as temporary:
        stage = Path(temporary).resolve()
        pack.stage_deployment_files(package_root, stage)
        # 在宿主展开 ~：容器渲染时 HOME 可能不是安装用户的家目录。
        envfile.write(stage / layout.PROJECT_ENV, envfile.read(package_root / layout.RELEASE_ENV))
        site.write_answers(stage / layout.SITE_ENV, answers)
        _render_in_stage(stage, profile, release["BASE_IMAGE"], docker, package_root, loaded)
        compose_files = envfile.read_compose(stage / layout.ENV)["COMPOSE_FILE"].split(":")
        services = docker.compose_config(stage, env_file=stage / layout.ENV,
                                         files=[stage / name for name in compose_files])["services"]
        _relocate(services, stage, install_dir)
        data_paths = mounts.prepare(services, install_dir, package_root)
        for name, service in sorted(services.items()):
            image = service.get("image")
            if image and docker.image_platform(image) != release["TARGET_PLATFORM"]:
                raise DeployError(f"{name}：镜像 {image} 不存在或不是 {release['TARGET_PLATFORM']} 平台")
        _check_configs(services, stage, release, docker)
        configuration = {layout.ENV: (stage / layout.ENV).read_bytes()}
        for path in sorted((stage / layout.GENERATED).rglob("*")):
            if path.is_file():
                configuration[path.relative_to(stage).as_posix()] = path.read_bytes()
    targets.check(package_root, install_dir, configuration)
    return Plan(package_root, install_dir, release, profile, answers, fields, files, obsolete,
                configuration, data_paths, options.clean)


def describe(plan: Plan, docker: Docker) -> str:
    lines = [f"安装目录：{plan.install_dir}"]
    if (plan.install_dir / layout.ENV).is_file():
        lines.append("停止：用已安装的定义停止当前服务（docker compose down）")
    if plan.clean:
        lines.append(f"清空：{plan.install_dir}")
    lines += [f"删除退役文件：{path}" for path in plan.obsolete]
    lines.append(f"写入：{len(inventory.managed(plan.files))} 个发布文件、"
                 f"{len(plan.configuration)} 个配置文件与 {layout.SITE_ENV}")
    lines += [f"可写挂载：{path}" for path in plan.data_paths]
    volume = layout.payload_volume(plan.release["PROJECT_NAME"], plan.release[layout.PAYLOAD_ID_KEY])
    lines.append(f"启动：docker compose up -d（载荷卷 {volume}{'，已播种' if docker.volume_exists(volume) else ''}），"
                 "清理不再使用的旧载荷卷，然后等待就绪")
    return "\n".join(lines)

