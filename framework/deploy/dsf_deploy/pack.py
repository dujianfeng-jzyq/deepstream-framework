"""打包：渲染所选 profile，收集镜像与部署文件，生成可离线安装的发布包。

包内容：apps 镜像与依赖镜像（每次都带）、base 镜像（--with-base 才带）、部署文件、模型、
release.env（发布事实）与 package-files.json（完整性清单）。不含源码、Dockerfile、渲染结果、
现场答案与 engine。
"""

from __future__ import annotations

import shutil
import tarfile
import tempfile
from pathlib import Path

from . import envfile, inventory, layout, naming, render
from .docker import Docker
from .envfile import DeployError

NON_DEPENDENCY_IMAGES = ("DEEPSTREAM_IMAGE", "BASE_IMAGE", "APPS_IMAGE")
PACKAGE_FORMAT = "1"


def dependency_images(values: dict[str, str]) -> dict[str, str]:
    """按约定收集外部依赖：所有非空的 *_IMAGE，排除上游与本项目的镜像。"""
    return {key: value for key, value in sorted(values.items())
            if key.endswith("_IMAGE") and key not in NON_DEPENDENCY_IMAGES and value}


def _excluded(path: Path) -> bool:
    return path.name in layout.PACKAGE_EXCLUDED_NAMES or path.suffix in layout.PACKAGE_EXCLUDED_SUFFIXES


def _copy_tree(source: Path, target: Path) -> None:
    for path in sorted(source.rglob("*")):
        if any(part in layout.PACKAGE_EXCLUDED_NAMES for part in path.relative_to(source).parts):
            continue
        if path.is_symlink():
            raise DeployError(f"refusing to package a symlink: {path}")
        if path.is_file() and not _excluded(path):
            destination = target / path.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)


def stage_deployment_files(root: Path, target: Path, *, models: bool = True) -> None:
    """复制部署需要的文件（也被安装器用来在临时目录里重建一份可渲染的项目）。
    models=False 时不复制 models/：渲染与 Compose 解析不读取模型，只做校验时省去复制模型主体。"""
    for name in layout.PACKAGE_FILES:
        source = root / name
        if not source.is_file():
            raise DeployError(f"missing deployment file: {source}")
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    for name in layout.PACKAGE_TREES:
        if name == layout.MODELS and not models:
            continue
        if (root / name).is_dir():
            _copy_tree(root / name, target / name)
    for unit in render.units(root):
        relative = unit.relative_to(root)
        (target / relative).mkdir(parents=True, exist_ok=True)
        shutil.copy2(unit / layout.UNIT_COMPOSE, target / relative / layout.UNIT_COMPOSE)
        if (unit / layout.UNIT_TEMPLATES).is_dir():
            _copy_tree(unit / layout.UNIT_TEMPLATES, target / relative / layout.UNIT_TEMPLATES)


def _require_platform(docker: Docker, image: str) -> str:
    value = docker.image_platform(image)
    if value not in ("amd64", "arm64"):
        raise DeployError(f"image not found or platform unknown: {image}; build or pull it first")
    return value


def pack(root: Path, profile: str | None, with_base: bool, docker: Docker | None = None) -> Path:
    docker = docker or Docker()
    plan = render.render(root, profile)
    values = plan.variables
    project = values["PROJECT_NAME"]
    base_image, apps_image = values["BASE_IMAGE"], values["APPS_IMAGE"]
    dependencies = dependency_images(values)

    # 平台以实际镜像为准：构建机架构可能与镜像不同（交叉构建、手动拉取）。
    target_platform = _require_platform(docker, base_image)
    for image in dependencies.values():
        if not docker.image_exists(image):
            print(f"pulling dependency image {image} for linux/{target_platform}")
            docker.pull(image, target_platform)
    for image in (apps_image, *dependencies.values()):
        actual = _require_platform(docker, image)
        if actual != target_platform:
            raise DeployError(f"{image} is {actual} but {base_image} is {target_platform}; "
                              "all images in a package must share one platform")

    name = naming.package_name(project, values["APPS_VERSION"], values.get("DEEPSTREAM_VERSION", ""))
    dist = root / "dist"
    dist.mkdir(exist_ok=True)
    archive = dist / f"{name}.tar.gz"
    with tempfile.TemporaryDirectory(prefix="dsf-pack-") as temporary:
        package = Path(temporary) / name
        images = package / "images"
        images.mkdir(parents=True)
        selected = [apps_image, *([base_image] if with_base else []), *dependencies.values()]
        for image in selected:
            print(f"saving {image}")
            docker.save(image, images / naming.image_tar_name(image))
        stage_deployment_files(root, package)
        installer = root / "scripts" / "install.sh"
        shutil.copy2(installer, package / "install.sh")
        (package / "install.sh").chmod(0o755)

        # 发布事实：项目公共输入 + 本次构建的实际镜像与平台。不含现场答案与宿主身份。
        release = envfile.read(root / layout.PROJECT_ENV, expand_home=False)
        release.update({
            "PACKAGE_FORMAT": PACKAGE_FORMAT,
            "PROJECT_NAME": project,
            "CONFIG_PROFILE": plan.profile,
            "TARGET_PLATFORM": target_platform,
            "DEEPSTREAM_VERSION": values.get("DEEPSTREAM_VERSION", ""),
            "DEEPSTREAM_IMAGE": values["DEEPSTREAM_IMAGE"],
            "APPS_VERSION": values["APPS_VERSION"],
            "BASE_VERSION": values["BASE_VERSION"],
            "BASE_IMAGE": base_image,
            "APPS_IMAGE": apps_image,
            **dependencies,
        })
        envfile.write(package / layout.RELEASE_ENV, release, (f"release {name}", "GENERATED — DO NOT EDIT"))
        inventory.write(package)
        archive.unlink(missing_ok=True)
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(package, arcname=name)

    size = archive.stat().st_size / (1 << 20)
    print(f"\ncreated {archive} ({size:.1f} MB, {target_platform}, profile {plan.profile})")
    if not with_base:
        print(f"note: the base image is not included; the target must already have {base_image}")
    return archive
