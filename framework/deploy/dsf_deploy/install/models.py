"""模型包的安装与回滚：只更新 models/<id>/，停服时间只有重启引用模型的服务（docs/training.md §13）。

顺序不可调换：
  1. 包完整性；安装目录中已安装同一项目、已有这些模型
  2. 新旧模型的契约：类别与输出布局不变（变化需要完整的发布包：业务规则、二级模型与解析插件随之变化）；
     模型校验（在 base 中运行，与打包时同一套规则）；nvinfer 引用的解析插件在已安装的载荷中
  3. 旧服务继续运行时构建 engine：engine 路径带发布标识，与正在使用的 engine 不冲突
  4. 切换模型目录（改名，上一份留在 .models-previous/ 供回滚），只重启挂载 models/ 的常驻服务，等待就绪
  5. 清理既不被当前、也不被上一份模型引用的 engine
1–3 的任何失败都不改动安装目录、不停任何服务。回滚是同一流程，候选是 .models-previous/ 中的上一份。
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .. import envfile, inventory, layout, models, pack
from ..docker import Docker
from ..errors import DeployError
from . import preflight, ready

PREVIOUS_MOUNT = "/models-previous"


@dataclass
class ModelOptions:
    package: Path
    install_dir: Path | None = None
    rollback: bool = False
    dry_run: bool = False
    ready_timeout: float = 180.0


@dataclass
class Change:
    model: str
    candidate: Path          # 要装入的模型目录（包内或 .models-previous/ 中）
    installed: str           # 当前的发布标识
    incoming: str            # 装入后的发布标识


def install_models(options: ModelOptions, docker: Docker | None = None) -> None:
    docker = docker or Docker()
    if not docker.available():
        raise DeployError("需要 docker")
    if options.ready_timeout <= 0:
        raise DeployError("--ready-timeout 必须大于 0")
    package = options.package.resolve()

    # 1. 包与安装目录
    info = envfile.read(package / layout.MODEL_PACKAGE_ENV, expand_home=False)
    if info.get("PACKAGE_KIND") != layout.MODEL_PACKAGE_KIND:
        raise DeployError(f"{package} 不是模型包（缺少 {layout.MODEL_PACKAGE_ENV}）")
    for key in ("PROJECT_NAME", "MODELS"):
        envfile.require(info, key, layout.MODEL_PACKAGE_ENV)
    inventory.verify(package)
    install_dir = (options.install_dir or preflight.default_install_dir(info)).absolute()
    if install_dir.is_symlink():
        raise DeployError(f"安装目录是符号链接：{install_dir}")
    install_dir = install_dir.resolve()
    project = info["PROJECT_NAME"]
    if not (install_dir / layout.ENV).is_file():
        raise DeployError(f"{install_dir} 中没有已安装的 {project}：模型包只更新已安装的项目，先安装完整的发布包")
    installed_project = envfile.read_compose(install_dir / layout.ENV).get("PROJECT_NAME", "")
    if installed_project != project:
        raise DeployError(f"{install_dir} 安装的是 {installed_project}，这个模型包属于 {project}")
    changes = _changes(package, install_dir, info["MODELS"].split(","), options.rollback)
    if not changes:
        return

    # 2. 模型校验、引用模型的服务与解析插件
    base_image = envfile.read_compose(install_dir / layout.ENV).get("BASE_IMAGE", "")
    if not base_image or not docker.image_exists(base_image):
        raise DeployError(f"缺少已安装项目的 base 镜像 {base_image or '（.env 中没有 BASE_IMAGE）'}")
    _check_models(package, changes, base_image, docker)
    config = docker.compose_config(install_dir)
    services = config["services"]
    builder = services.get(layout.ENGINE_BUILDER)
    if builder is None:
        raise DeployError(f"已安装的项目没有 {layout.ENGINE_BUILDER} 服务，无法在停服前构建 engine")
    consumers = _consumers(services, install_dir)
    payload = (config.get("volumes") or {}).get(layout.PAYLOAD_VOLUME, {}).get("name", "")
    _check_parsers(changes, payload, base_image, docker)

    action = "回滚" if options.rollback else "更新"
    print(f"安装目录：{install_dir}")
    for change in changes:
        print(f"{action}模型：{change.model} {change.installed} -> {change.incoming}")
    print(f"重启：{', '.join(consumers) or '（没有挂载 models/ 的常驻服务）'}")
    if options.dry_run:
        print("\n--dry-run：没有构建 engine，没有改动模型目录，也没有重启服务。")
        return

    # 3. 停服前构建 engine（旧服务继续运行）
    staging = install_dir / layout.STAGED_MODELS
    shutil.rmtree(staging, ignore_errors=True)  # 上次中断留下的
    staging.mkdir()
    try:
        for change in changes:
            shutil.copytree(change.candidate, staging / change.model)
        print("构建 engine（服务继续运行）")
        _run_builder(builder, config, install_dir, staging, [], docker)

        # 4. 切换并重启
        try:
            for change in changes:
                _swap(install_dir, change.model, staging / change.model)
            _update_managed(install_dir, [change.model for change in changes])
            if consumers:
                print(f"重启 {', '.join(consumers)}")
                docker.compose(install_dir, ["restart", *consumers])
        except BaseException:
            print(_incomplete(package, install_dir, consumers, options.rollback), file=sys.stderr)
            raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    try:
        ready.wait(docker, install_dir, options.ready_timeout)
    except KeyboardInterrupt:
        print(f"\n已停止等待：模型已{action}，服务继续运行\n  查看：cd {install_dir} && docker compose ps",
              file=sys.stderr)
        raise

    # 5. 清理不再引用的 engine：当前与上一份（回滚目标）都保留
    try:
        _run_builder(builder, config, install_dir, install_dir / layout.MODELS,
                     ["--keep-models-path", PREVIOUS_MOUNT, "--prune-engines",
                      f"{layout.MOUNTS['CONTAINER_DATA_PATH']}/engines"], docker,
                     previous=install_dir / layout.PREVIOUS_MODELS)
    except DeployError as error:
        print(f"提示：没有清理旧 engine：{error}", file=sys.stderr)
    print(f"\n模型已{action}：" + ", ".join(f"{c.model} {c.incoming}" for c in changes))
    if not options.rollback:
        print(f"回滚方法：在 {package} 中运行 ./install.sh --rollback")


def _release(directory: Path) -> dict:
    path = directory / models.RELEASE
    if not path.is_file():
        raise DeployError(f"{directory} 没有 {models.RELEASE}")
    return json.loads(path.read_text(encoding="utf-8"))


def _changes(package: Path, install_dir: Path, ids: list[str], rollback: bool) -> list[Change]:
    changes = []
    for model in ids:
        inventory.safe_path(install_dir, f"{layout.MODELS}/{model}")
        current = install_dir / layout.MODELS / model
        if not current.is_dir():
            raise DeployError(f"已安装的项目没有模型 {model}：新模型需要完整的发布包（应用配置随之变化）")
        installed = _release(current)
        packaged = _release(package / layout.MODELS / model)
        if rollback:
            candidate = install_dir / layout.PREVIOUS_MODELS / model
            if installed.get("release") != packaged.get("release"):
                raise DeployError(f"{model} 当前是 {installed.get('release')}，不是这个包装入的 "
                                  f"{packaged.get('release')}：回滚只撤销本包的安装")
            if not candidate.is_dir():
                raise DeployError(f"没有 {model} 的上一份模型目录（{candidate}），无法回滚")
        else:
            candidate = package / layout.MODELS / model
            if installed.get("release") == packaged.get("release"):
                print(f"{model} 已是 {installed.get('release')}，跳过")
                continue
        incoming = _release(candidate)
        for key, meaning in (("classes", "类别"), ("output_layout", "输出布局")):
            if incoming.get(key) != installed.get(key):
                raise DeployError(f"{model} 的{meaning}与已安装的不同：业务规则、二级模型与解析插件依赖它，"
                                  "需要完整的发布包")
        changes.append(Change(model, candidate, installed.get("release", ""), incoming.get("release", "")))
    if not changes:
        print("没有需要更新的模型")
    return changes


def _check_models(package: Path, changes: list[Change], base_image: str, docker: Docker) -> None:
    """在 base 中执行打包时的模型校验（需要 PyYAML）：只挂载一个含部署工具与候选模型的临时目录。"""
    with tempfile.TemporaryDirectory(prefix="dsf-models-") as temporary:
        stage = Path(temporary).resolve()
        for tree in layout.MODEL_PACKAGE_TREES:
            pack._copy_tree(package / tree, stage / tree)
        for change in changes:
            shutil.copytree(change.candidate, stage / layout.MODELS / change.model)
        print("检查模型")
        preflight._run_deploy_tool(stage, ["check-models"], base_image, docker, writable=False,
                                   package_root=package, loaded=set())


def _consumers(services: dict, install_dir: Path) -> list[str]:
    """挂载了安装目录 models/ 的常驻服务：它们在启动时加载模型，换模型后需要重启。"""
    models_dir = install_dir / layout.MODELS
    result = []
    for name, service in sorted(services.items()):
        if name == layout.ENGINE_BUILDER or service.get("profiles") or service.get("restart", "no") == "no":
            continue
        for mount in service.get("volumes") or []:
            if mount.get("type") == "bind" and Path(mount.get("source", "")).resolve() == models_dir:
                result.append(name)
                break
    return result


def _check_parsers(changes: list[Change], payload: str, base_image: str, docker: Docker) -> None:
    """nvinfer 配置引用的解析插件必须在已安装的载荷中：模型包不换程序。"""
    workspace = layout.MOUNTS["CONTAINER_WORKSPACE"]
    for change in changes:
        for config in sorted(change.candidate.glob("*.txt")):
            properties = models._properties(config) or {}
            library = properties.get("custom-lib-path", "")
            if not library.startswith(workspace + "/"):
                continue
            if not payload:
                raise DeployError("已安装的项目没有载荷卷，无法检查解析插件")
            try:
                docker.run(["--rm", "--network", "none", "-v", f"{payload}:{workspace}:ro",
                            "--entrypoint", "test", base_image, "-f", library], capture=True)
            except DeployError:
                raise DeployError(f"{change.model}/{config.name} 引用的 {library} 不在已安装的程序中："
                                  "需要带有该解析插件的完整发布包") from None


def _run_builder(builder: dict, config: dict, install_dir: Path, models_dir: Path, extra: list[str],
                 docker: Docker, previous: Path | None = None) -> None:
    """按已安装的 model-engine-builder 定义（镜像、用户、运行时、设备、挂载）运行一次，只把 /models 换成
    models_dir。不联网。previous 存在时只读挂到 PREVIOUS_MOUNT。"""
    models_mount = layout.MOUNTS["CONTAINER_MODELS_PATH"]
    named = config.get("volumes") or {}
    args = ["--rm", "--network", "none"]
    if builder.get("runtime"):
        args += ["--runtime", builder["runtime"]]
    if builder.get("user"):
        args += ["--user", str(builder["user"])]
    for group in builder.get("group_add") or []:
        args += ["--group-add", str(group)]
    for device in builder.get("devices") or []:
        if isinstance(device, dict):
            value = f"{device['source']}:{device.get('target', device['source'])}"
            device = value + (f":{device['permissions']}" if device.get("permissions") else "")
        args += ["--device", str(device)]
    for key, value in sorted((builder.get("environment") or {}).items()):
        if value is not None:
            args += ["-e", f"{key}={value}"]
    for mount in builder.get("volumes") or []:
        target, source, kind = mount["target"], mount.get("source", ""), mount.get("type")
        if target == models_mount:
            source, read_only = str(models_dir), True
        else:
            read_only = bool(mount.get("read_only"))
            if kind == "volume":
                source = (named.get(source) or {}).get("name", source)
        args += ["-v", f"{source}:{target}{':ro' if read_only else ''}"]
    if previous is not None:
        if not previous.is_dir():
            previous.mkdir(parents=True)
        args += ["-v", f"{previous}:{PREVIOUS_MOUNT}:ro"]
    if builder.get("working_dir"):
        args += ["-w", builder["working_dir"]]
    entrypoint = builder.get("entrypoint") or []
    command = [*entrypoint[1:], *(builder.get("command") or []), *extra]
    if entrypoint:
        args += ["--entrypoint", entrypoint[0]]
    docker.run([*args, builder["image"], *command])


def _swap(install_dir: Path, model: str, candidate: Path) -> None:
    """candidate 换入 models/<model>，原来的移到 .models-previous/<model>（替换更早的上一份）。
    都在安装目录中，改名即切换；服务挂载的是 models/ 目录，改名后立即看到新内容。"""
    current = install_dir / layout.MODELS / model
    previous = install_dir / layout.PREVIOUS_MODELS / model
    previous.parent.mkdir(exist_ok=True)
    if previous.exists():
        shutil.rmtree(previous)
    os.rename(current, previous)
    os.rename(candidate, current)


def _update_managed(install_dir: Path, changed: list[str]) -> None:
    """受管理清单记录装入后的模型文件：之后的完整安装把它们当作未被现场修改的发布文件。"""
    record = install_dir / layout.MANAGED
    if not record.is_file():
        return
    managed = json.loads(record.read_text(encoding="utf-8"))
    for model in changed:
        prefix = f"{layout.MODELS}/{model}/"
        managed = {name: digest for name, digest in managed.items() if not name.startswith(prefix)}
        directory = install_dir / layout.MODELS / model
        for path in sorted(p for p in directory.rglob("*") if p.is_file()):
            managed[path.relative_to(install_dir).as_posix()] = inventory.sha256(path)
    record.write_text(json.dumps(dict(sorted(managed.items())), indent=2) + "\n", encoding="utf-8")


def _incomplete(package: Path, install_dir: Path, consumers: list[str], rollback: bool) -> str:
    """切换之后、重启之前的失败：模型目录可能已是新的而服务仍在用旧 engine。再次运行安装会认为已经装好
    而跳过，所以直接给出重启命令。"""
    restart = f"cd {install_dir} && docker compose restart {' '.join(consumers)}" if consumers else ""
    lines = [f"\n{'回滚' if rollback else '安装'}未完成：模型目录可能已切换而服务没有重启。",
             f"  查看：ls {install_dir / layout.MODELS}（release.json 是当前的模型）"]
    if restart:
        lines.append(f"  已切换时完成：{restart}")
    lines.append(f"  没有切换时：在 {package} 中重新运行 ./install.sh{' --rollback' if rollback else ''}")
    return "\n".join(lines)
