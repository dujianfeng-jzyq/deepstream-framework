"""模型包：只更新模型，不换程序（docs/training.md §13）。

包内容：models/<id>/（已发布的模型目录）、部署工具（安装入口在包内运行，不依赖已安装的版本）、
model-package.env（发布事实）与 package-files.json（完整性清单）。不含镜像、程序与 engine：
模型包只能装到已安装同一项目、已有这些模型的目录中，engine 在目标机构建。
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from . import envfile, inventory, layout, models, naming, pack
from .errors import DeployError

PACKAGE_FORMAT = "1"


def _uncommitted(root: Path, relative: str) -> bool:
    """模型目录在 git 中有未提交的改动（不是 git 仓库时视为没有）。"""
    result = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--", relative],
                            capture_output=True, text=True)
    return result.returncode == 0 and bool(result.stdout.strip())


def pack_models(root: Path, ids: list[str]) -> Path:
    if not ids:
        raise DeployError("name the models to package: pack-models <model> [<model> ...]")
    project = envfile.require(envfile.read(root / layout.PROJECT_ENV, expand_home=False), "PROJECT_NAME",
                              layout.PROJECT_ENV)
    releases = {}
    for model in dict.fromkeys(ids):
        directory = root / layout.MODELS / model
        if not (directory / models.SPEC).is_file():
            raise DeployError(f"{directory} is not a model directory")
        if not (directory / models.RELEASE).is_file():
            raise DeployError(f"{model} has not been published ({models.RELEASE} is missing): train publish first")
        problems = models.check_model(directory)
        if problems:
            raise DeployError(f"model check failed for {model}:\n  " + "\n  ".join(problems))
        releases[model] = json.loads((directory / models.RELEASE).read_text(encoding="utf-8"))["release"]
        if _uncommitted(root, f"{layout.MODELS}/{model}"):
            print(f"WARNING: {layout.MODELS}/{model} has uncommitted changes; commit the published model "
                  "so the package can be traced to the repository", file=sys.stderr)

    name = f"{project}-models-" + "-".join(f"{model}-{release}" for model, release in releases.items())
    dist = root / "dist"
    dist.mkdir(exist_ok=True)
    archive = dist / f"{name}.tar.gz"
    with tempfile.TemporaryDirectory(prefix="dsf-pack-models-") as temporary:
        package = Path(temporary) / name
        for model in releases:
            pack._copy_tree(root / layout.MODELS / model, package / layout.MODELS / model)
        for tree in layout.MODEL_PACKAGE_TREES:
            pack._copy_tree(root / tree, package / tree)
        shutil.copy2(root / "scripts" / "install-models.sh", package / "install.sh")
        (package / "install.sh").chmod(0o755)
        project_env = envfile.read(root / layout.PROJECT_ENV, expand_home=False)
        envfile.write(package / layout.MODEL_PACKAGE_ENV, {
            "PACKAGE_KIND": layout.MODEL_PACKAGE_KIND,
            "PACKAGE_FORMAT": PACKAGE_FORMAT,
            "PROJECT_NAME": project,
            "INSTALL_ROOT": project_env.get("INSTALL_ROOT", ""),
            "MODELS": ",".join(releases),
            "MODEL_RELEASES": ",".join(f"{model}={release}" for model, release in releases.items()),
            "SOURCE_REVISION": naming.source_revision(root),
        }, (f"model package {name}", "GENERATED — DO NOT EDIT"))
        inventory.write(package)
        archive.unlink(missing_ok=True)
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(package, arcname=name)
    size = archive.stat().st_size / (1 << 20)
    print(f"created {archive} ({size:.1f} MB): " + ", ".join(f"{m} {r}" for m, r in releases.items()))
    print("install on the target: tar xzf <package> && cd <package> && ./install.sh")
    return archive
