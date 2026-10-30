"""配置渲染：project.env + profile + 现场参数 → .env 与 generated/<type>/*.yaml。

变量优先级：现场清单默认值 < project.env < profile 的 x-build < site.env。
模板中的 ${VAR} 必须有定义；profile 的 x-app-<type> 在替换后按文件名深合并
（映射递归合并，列表与标量整体替换）。
profile 的 x-network-mode 给所有未显式声明 network_mode 的服务设置该值（例如内核不支持
bridge 的设备用 host），新增的服务单元自动生效。
profile 的 x-platform（amd64/arm64）声明它面向的平台：打包时镜像必须是这个平台，安装时只提供
与发布包平台、DeepStream 版本相符的 profile。
先计算完整的渲染计划，再整体替换 .env 与 generated/；校验失败时旧文件保持不变。
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from . import envfile, layout, naming, site
from .errors import DeployError

_PLACEHOLDER = re.compile(r"\$\{([A-Za-z0-9_]+)\}")
# profile 不能改变项目身份；镜像名在目标机上由发布包决定。
IDENTITY_KEYS = ("PROJECT_NAME",)
_APP_PREFIX = "x-app-"  # profile 中按类型覆盖渲染配置的键：x-app-<type>
RELEASE_IMAGE_KEYS = ("DEEPSTREAM_IMAGE", "BASE_IMAGE", "APPS_IMAGE")
BASE_DOCKERFILE = "framework/docker/base.Dockerfile"


def _yaml():
    """PyYAML 只在渲染时需要；缺失时给出可操作的提示。"""
    try:
        from dsf import yamlio  # framework/python 由入口脚本加入 sys.path
    except ModuleNotFoundError as error:
        raise DeployError(
            "rendering needs PyYAML on this host (python3 -m pip install pyyaml, "
            f"or apt-get install python3-yaml); interpreter: {sys.executable}") from error
    return yamlio


@dataclass
class Plan:
    profile: str
    variables: dict[str, str]
    files: dict[str, str]  # 相对项目根的路径 → 内容（含 .env）
    notes: list[str] = field(default_factory=list)  # 给用户的提示，由调用方输出


def list_profiles(root: Path) -> list[tuple[str, str]]:
    """profile 名与其第一行注释（去掉“profile：”前缀）。"""
    result = []
    for path in sorted((root / layout.PROFILES).glob("*.yaml")):
        summary = ""
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("#"):
                summary = line.strip().lstrip("#").strip()
                summary = summary.split("：", 1)[1].strip() if summary.startswith("profile：") else summary
                break
        result.append((path.stem, summary))
    return result


def _describe_profiles(root: Path) -> str:
    entries = list_profiles(root)
    if not entries:
        return "  (no profiles/*.yaml)"
    width = max(len(name) for name, _ in entries)
    return "\n".join(f"  {name:<{width}}  {summary}" for name, summary in entries)


def selected_profile(root: Path, requested: str | None) -> str:
    if requested:
        return requested
    env = root / layout.ENV
    if env.is_file():
        chosen = envfile.read_compose(env).get("CONFIG_PROFILE", "")
        if chosen:
            return chosen
    raise DeployError("no profile selected; pick one with scripts/apply-profile.sh <profile>\n"
                      "available profiles:\n" + _describe_profiles(root))


def units(root: Path) -> list[Path]:
    """服务单元：services/ 下含 compose.yaml 的目录，按名称排序。"""
    base = root / layout.SERVICES
    if not base.is_dir():
        return []
    return sorted(p for p in base.iterdir() if (p / layout.UNIT_COMPOSE).is_file())


def substitute(value, variables: dict[str, str], origin: str):
    if isinstance(value, str):
        def replace(match: re.Match) -> str:
            key = match.group(1)
            if key not in variables:
                raise DeployError(f"{origin}: undefined variable ${{{key}}}")
            return variables[key]
        return _PLACEHOLDER.sub(replace, value)
    if isinstance(value, dict):
        return {key: substitute(item, variables, origin) for key, item in value.items()}
    if isinstance(value, list):
        return [substitute(item, variables, origin) for item in value]
    return value


def deep_merge(base, override):
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for key, value in override.items():
            merged[key] = deep_merge(base[key], value) if key in base else value
        return merged
    return override


def _load_profile(root: Path, name: str) -> dict:
    path = root / layout.PROFILES / f"{name}.yaml"
    if not path.is_file():
        raise DeployError(f"unknown profile {name!r}\navailable profiles:\n" + _describe_profiles(root))
    try:
        profile = _yaml().load(path.read_text(encoding="utf-8")) or {}
    except Exception as error:
        raise DeployError(f"{path}: {error}") from error
    if not isinstance(profile, dict):
        raise DeployError(f"{path}: a profile must be a mapping")
    unknown = sorted(k for k in profile if k not in ("services", "x-build", "x-network-mode", "x-platform")
                     and not k.startswith(_APP_PREFIX))
    if unknown:
        raise DeployError(f"{path}: unknown top-level keys {unknown}; use services (Compose overrides), "
                          "x-build (variables), x-network-mode, x-platform or x-app-<type> "
                          "(configuration overrides)")
    platform = profile.get("x-platform")
    if platform is not None and platform not in naming.PLATFORMS:
        raise DeployError(f"{path}: x-platform must be one of {', '.join(naming.PLATFORMS)}")
    build = profile.get("x-build") or {}
    if not isinstance(build, dict) or any(isinstance(v, (dict, list)) for v in build.values()):
        raise DeployError(f"{path}: x-build must map names to scalar values")
    identity = sorted(k for k in build if k in IDENTITY_KEYS)
    if identity:
        raise DeployError(f"{path}: x-build cannot set {', '.join(identity)}; "
                          f"the project identity comes from {layout.PROJECT_ENV}")
    mode = profile.get("x-network-mode")
    if mode is not None and (not isinstance(mode, str) or not mode.strip()):
        raise DeployError(f"{path}: x-network-mode must be a Compose network_mode such as host")
    return profile


def _mismatch(profile: dict, platform: str, ds_version: str) -> str:
    declared = profile.get("x-platform")
    if declared and platform and declared != platform:
        return f"it targets {declared}, the images are {platform}"
    image = (profile.get("x-build") or {}).get("DEEPSTREAM_IMAGE")
    wanted = naming.deepstream_version(str(image)) if image else ""
    if wanted and ds_version and wanted != ds_version:
        return f"it uses DeepStream {wanted}, the images have {ds_version}"
    return ""


def profile_mismatch(root: Path, name: str, platform: str, ds_version: str) -> str:
    """profile 与给定平台、DeepStream 版本不符的原因，相符返回空串。没有声明 x-platform 或
    x-build 的 DEEPSTREAM_IMAGE 时，对应一项不检查。"""
    return _mismatch(_load_profile(root, name), platform, ds_version)


_PLATFORM_LINE = re.compile(r"^x-platform:\s*['\"]?([A-Za-z0-9_]+)['\"]?\s*(?:#.*)?$", re.M)


def fitting_profiles(root: Path, platform: str) -> list[tuple[str, str]]:
    """x-platform 与 platform 相符（或没有声明）的 profile 名与说明，供安装时选择。
    不解析 YAML（安装宿主没有 PyYAML，选择 profile 时 base 可能尚未加载）：只读顶层的 x-platform 行；完整的检查（含 DeepStream
    版本）在渲染时进行，仍在停服之前。"""
    result = []
    for name, summary in list_profiles(root):
        match = _PLATFORM_LINE.search((root / layout.PROFILES / f"{name}.yaml").read_text(encoding="utf-8"))
        if match is None or match.group(1) == platform:
            result.append((name, summary))
    return result


def _build_variables(profile: dict) -> dict[str, str]:
    return {k: str(v) for k, v in (profile.get("x-build") or {}).items()}


def profile_variables(root: Path, name: str) -> dict[str, str]:
    """profile 的 x-build 变量。安装器用它计算现场参数的提问默认值。"""
    return _build_variables(_load_profile(root, name))


def _templates(root: Path) -> dict[str, dict[str, Path]]:
    """type → {文件名: 模板路径}。不同单元产生同名输出文件时报错。"""
    result: dict[str, dict[str, Path]] = {}
    for unit in units(root):
        templates = unit / layout.UNIT_TEMPLATES
        if not templates.is_dir():
            continue
        for type_dir in sorted(p for p in templates.iterdir() if p.is_dir()):
            for template in sorted(type_dir.glob("*.yaml")):
                existing = result.setdefault(type_dir.name, {}).get(template.name)
                if existing is not None:
                    raise DeployError(f"{template} and {existing} both render "
                                      f"{layout.GENERATED}/{type_dir.name}/{template.name}")
                result[type_dir.name][template.name] = template
    return result


def _network_overrides(root: Path, compose_files: list[str], mode: str) -> dict:
    """给未显式声明 network_mode 的服务设置 mode。只要有一个服务留在默认网络，Compose 就会
    创建 bridge 网络，所以必须覆盖全部服务；显式的 network_mode（如 none）保持不变。
    extends 来源中的声明同样算显式（相对路径以项目根为基准，与 Compose 一致）。
    mode 为 host 时服务名不再经 Compose 的 DNS 解析：给这些服务加 extra_hosts，把每个服务名
    指向 127.0.0.1，配置中照常用服务名（redis、mediamtx）访问本机服务，不必按网络模式覆盖；
    并清空这些服务的 ports（host 网络下端口映射无效，Docker 每次创建容器都会为此告警）。"""
    yamlio = _yaml()
    documents: dict[Path, dict] = {}

    def services_of(path: Path) -> dict:
        if path not in documents:
            try:
                document = yamlio.load(path.read_text(encoding="utf-8")) if path.is_file() else None
            except Exception as error:
                raise DeployError(f"{path.relative_to(root)}: {error}") from error
            documents[path] = (document or {}).get("services") or {}
        return documents[path]

    explicit: dict[str, bool] = {}
    published: set[str] = set()
    for relative in compose_files:
        for name, service in services_of(root / relative).items():
            sources = [service or {}]
            extends = sources[0].get("extends")
            if isinstance(extends, str):
                extends = {"service": extends}
            if isinstance(extends, dict):
                origin = root / extends["file"] if "file" in extends else root / relative
                sources.append(services_of(origin).get(extends.get("service")) or {})
            if any("networks" in source for source in sources):
                raise DeployError(f"{relative}: service {name} declares networks, "
                                  "which cannot be combined with x-network-mode")
            explicit[name] = explicit.get(name, False) or any("network_mode" in s for s in sources)
            if any(source.get("ports") for source in sources):
                published.add(name)
    names = sorted(explicit)

    def override(service: str) -> dict:  # 每个服务一份新列表：共享的对象会被 dump 成 YAML 锚点
        result: dict = {"network_mode": mode}
        if mode == "host":
            result["extra_hosts"] = [f"{name}:127.0.0.1" for name in names]
            if service in published:  # 只在需要时使用 !reset，没有端口映射的项目不要求新版 Compose
                result["ports"] = yamlio.Reset()
        return result

    return {"services": {name: override(name) for name in names if not explicit[name]}}


def build_plan(root: Path, requested_profile: str | None = None) -> Plan:
    """计算全部渲染结果，不写文件。"""
    profile_name = selected_profile(root, requested_profile)
    profile = _load_profile(root, profile_name)
    variables, notes, released = _resolve_variables(root, profile_name, profile)
    files = _render_templates(root, profile_name, profile, variables)
    compose_files = _compose_files(root, profile_name, profile, released, files)
    variables["CONFIG_PROFILE"] = profile_name
    variables["COMPOSE_FILE"] = ":".join(compose_files)
    # Compose 按自己的 dotenv 规则读取 .env：值一律加引号，现场口令中的 $、#、引号保持原样。
    files[layout.ENV] = envfile.render(
        variables, ("GENERATED by the renderer — DO NOT EDIT", f"profile: {profile_name}"), envfile.quote_compose)
    return Plan(profile_name, variables, files, notes)


def _resolve_variables(root: Path, profile_name: str,
                       profile: dict) -> tuple[dict[str, str], list[str], bool]:
    """按优先级合并变量（见模块说明），派生镜像名与固定挂载点。返回变量、给用户的提示，
    以及这是否是发布包派生的安装。"""
    variables = envfile.read(root / layout.PROJECT_ENV)
    envfile.require(variables, "PROJECT_NAME", layout.PROJECT_ENV)
    # project.env 带有镜像名，说明这是发布包派生的安装：镜像由包决定，不再重新派生。
    released = bool(variables.get("BASE_IMAGE") and variables.get("APPS_IMAGE"))
    overrides = _build_variables(profile)
    notes: list[str] = []
    if released:
        _require_fitting(root, profile_name, profile, variables.get("TARGET_PLATFORM", ""),
                         variables.get("DEEPSTREAM_VERSION", ""))
        # 与包内相同的值（本项目每个 profile 都写明 DEEPSTREAM_IMAGE）不必提示；不同时才值得告诉用户。
        ignored = sorted(k for k, v in overrides.items() if k in RELEASE_IMAGE_KEYS and v != variables.get(k))
        if ignored:
            notes.append(f"images come from the release package; ignoring x-build {', '.join(ignored)}")
        overrides = {k: v for k, v in overrides.items() if k not in RELEASE_IMAGE_KEYS}
    variables.update(overrides)

    fields = site.parse_fields(root / layout.SITE_FIELDS)
    variables.update(site.resolve(fields, site.read_answers(root / layout.SITE_ENV, fields), variables))

    if not released:
        variables.update(_derived_images(root, profile_name, variables))
    variables.setdefault("HOST_DATA_PATH", os.path.expanduser(f"~/{variables['PROJECT_NAME']}/data"))
    variables.update(layout.MOUNTS)
    # 业务容器以渲染用户运行，写进数据目录的文件归现场用户所有。
    variables["HOST_UID"] = str(os.getuid())
    variables["HOST_GID"] = str(os.getgid())
    return variables, notes, released


def _require_fitting(root: Path, profile_name: str, profile: dict, platform: str, ds_version: str) -> None:
    """镜像已由发布包决定：面向其他平台或 DeepStream 版本的 profile 不能用来运行它们。"""
    reason = _mismatch(profile, platform, ds_version)
    if not reason:
        return
    fitting = [name for name, _ in fitting_profiles(root, platform)
               if not profile_mismatch(root, name, platform, ds_version)]
    raise DeployError(f"profile {profile_name} does not fit this package: {reason}; "
                      f"profiles that fit: {', '.join(fitting) or 'none'}")


def _derived_images(root: Path, profile_name: str, variables: dict[str, str]) -> dict[str, str]:
    """开发构建：由 DeepStream 镜像、版本号与 base 的构建输入派生镜像名，并记录源码版本。"""
    if not variables.get("DEEPSTREAM_IMAGE"):
        raise DeployError(f"profile {profile_name} must set x-build: DEEPSTREAM_IMAGE "
                          f"({layout.PROJECT_ENV} has no default)")
    for key in ("APPS_VERSION", "BASE_VERSION"):
        envfile.require(variables, key, f"{layout.PROJECT_ENV} or the profile")
    dockerfile = root / BASE_DOCKERFILE
    dockerfile_text = dockerfile.read_text(encoding="utf-8") if dockerfile.is_file() else ""
    derived = {"DEEPSTREAM_VERSION": naming.deepstream_version(variables["DEEPSTREAM_IMAGE"])}
    derived["BASE_INPUTS"] = naming.base_inputs_digest(
        dockerfile_text, variables["DEEPSTREAM_IMAGE"], variables.get("BASE_APT_PACKAGES", ""))
    derived.update(naming.image_names(variables["PROJECT_NAME"], derived["DEEPSTREAM_VERSION"],
                                      variables["APPS_VERSION"], variables["BASE_VERSION"],
                                      derived["BASE_INPUTS"]))
    derived["SOURCE_REVISION"] = naming.source_revision(root)
    return derived


def _render_templates(root: Path, profile_name: str, profile: dict, variables: dict[str, str]) -> dict[str, str]:
    """各单元 templates/<type>/*.yaml 替换变量、合并 profile 的 x-app-<type> 后的内容，按输出路径。"""
    yamlio = _yaml()
    templates = _templates(root)
    for key in profile:
        if key.startswith(_APP_PREFIX) and key[len(_APP_PREFIX):] not in templates:
            raise DeployError(f"profile {profile_name}: {key} has no matching "
                              f"templates/{key[len(_APP_PREFIX):]}/ in any unit")
    files: dict[str, str] = {}
    for type_name, entries in templates.items():
        type_overrides = profile.get(f"{_APP_PREFIX}{type_name}") or {}
        if not isinstance(type_overrides, dict):
            raise DeployError(f"profile {profile_name}: {_APP_PREFIX}{type_name} must be a mapping")
        unknown = sorted(set(type_overrides) - {Path(name).stem for name in entries})
        if unknown:
            raise DeployError(f"profile {profile_name}: {_APP_PREFIX}{type_name} targets unknown files {unknown}")
        for file_name, template in entries.items():
            relative = template.relative_to(root)
            try:
                document = yamlio.load(template.read_text(encoding="utf-8")) or {}
            except Exception as error:
                raise DeployError(f"{relative}: {error}") from error
            document = substitute(document, variables, str(relative))
            override = type_overrides.get(Path(file_name).stem)
            if override:
                document = deep_merge(document, substitute(override, variables, f"profile {profile_name}"))
            files[f"{layout.GENERATED}/{type_name}/{file_name}"] = (
                f"# GENERATED from {relative} by the renderer — DO NOT EDIT\n" + yamlio.dump(document))
    return files


def _compose_files(root: Path, profile_name: str, profile: dict, released: bool,
                   files: dict[str, str]) -> list[str]:
    """COMPOSE_FILE 的顺序：根定义、构建定义（只在开发机）、各单元、profile；有 x-network-mode 时
    最后加生成的网络覆盖（写入 files）。安装的项目没有构建定义：镜像只来自发布包，任何 Compose
    命令都不会尝试构建（包中也没有 Dockerfile）。"""
    compose_files = [layout.COMPOSE_ROOT] + ([] if released else [layout.BUILD_COMPOSE])
    compose_files += [str(unit.relative_to(root) / layout.UNIT_COMPOSE) for unit in units(root)]
    compose_files.append(f"{layout.PROFILES}/{profile_name}.yaml")
    mode = profile.get("x-network-mode")
    if mode is not None:
        files[layout.NETWORK_COMPOSE] = (
            f"# GENERATED from x-network-mode of profile {profile_name} — DO NOT EDIT\n"
            + _yaml().dump(_network_overrides(root, compose_files, mode.strip())))
        compose_files.append(layout.NETWORK_COMPOSE)
    return compose_files


def write_plan(root: Path, plan: Plan) -> list[str]:
    """暂存全部结果后替换 .env 与 generated/；替换失败时恢复旧版本。返回被移除的旧文件。"""
    generated = root / layout.GENERATED
    old = {str(p.relative_to(root)) for p in generated.rglob("*") if p.is_file()} if generated.is_dir() else set()
    with tempfile.TemporaryDirectory(prefix=".render-", dir=root) as temporary:
        stage = Path(temporary)
        new, backup = stage / "new", stage / "old"
        (new / layout.GENERATED).mkdir(parents=True)
        backup.mkdir()
        for relative, content in plan.files.items():
            path = new / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            path.chmod(0o600)  # 可能含现场凭据
        moved, installed = [], []
        try:
            for name in (layout.GENERATED, layout.ENV):
                target = root / name
                if target.exists() or target.is_symlink():
                    os.replace(target, backup / name)
                    moved.append(name)
                os.replace(new / name, target)
                installed.append(name)
        except OSError:
            for name in reversed(installed):
                target = root / name
                shutil.rmtree(target) if target.is_dir() else target.unlink()
            for name in reversed(moved):
                os.replace(backup / name, root / name)
            raise
    return sorted(old - set(plan.files))


def render(root: Path, profile: str | None = None, quiet: bool = False) -> Plan:
    plan = build_plan(root, profile)
    for note in plan.notes:
        print(f"note: {note}", file=sys.stderr)
    removed = write_plan(root, plan)
    if not quiet:
        version = plan.variables.get("DEEPSTREAM_VERSION") or "unknown"
        print(f"profile: {plan.profile} (DeepStream {version})")
        for path in sorted(plan.files):
            print(f"  wrote {path}")
        for path in removed:
            print(f"  removed {path} (template no longer exists)")
    return plan
