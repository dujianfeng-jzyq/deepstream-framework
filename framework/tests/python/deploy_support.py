"""部署测试的公共支持：临时项目、假 Docker。

假 Docker 记录所有调用并模拟镜像、卷与服务状态；只有 `docker compose config` 委托给真实 CLI
（只解析文件，不需要 Docker 守护进程）。没有 docker CLI 的环境里相关测试 SKIP。
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from dsf_deploy import layout, naming
from dsf_deploy.docker import Docker

ROOT = Path(__file__).resolve().parents[3]

PROJECT_ENV = """PROJECT_NAME=demo
DEEPSTREAM_IMAGE=nvcr.io/nvidia/deepstream:8.0-samples-multiarch
APPS_VERSION=1.2.0
BASE_VERSION=1.0.0
BASE_APT_PACKAGES=
PIP_INDEX_URL=
HOST_DATA_PATH={data}
INSTALL_ROOT=
REDIS_IMAGE=redis:7.4.2-alpine
"""

SITE_FIELDS = """# 相机编号
CAMERA_ID=cam0

# 口令
RTSP_PASSWORD=""

# 必填项
SITE_NAME=
"""

UNIT_COMPOSE = """services:
  app:
    extends: {file: framework/compose/defaults.yaml, service: dsf-service}
    entrypoint: [bin/app]
  redis:
    image: ${REDIS_IMAGE}
"""

UNIT_TEMPLATE = """camera: ${CAMERA_ID}
site: ${SITE_NAME}
password: ${RTSP_PASSWORD}
nested:
  keep: 1
  list: [1, 2]
"""

PROFILE = """# profile：测试方案
x-build:
  APPS_VERSION: 1.2.1
x-app-config:
  app:
    nested:
      list: [3]
"""


def docker_cli_available() -> bool:
    return shutil.which("docker") is not None


class FakeDocker(Docker):
    def __init__(self, platform: str | None = None):
        self.calls: list[list[str]] = []
        self.images: dict[str, str] = {}
        self.volumes: set[str] = set()
        self.platform = platform or naming.host_platform()
        self.ps_rows: list[dict] | None = None
        self.fail_on: str | None = None
        self.env_files: list[tuple[int, str]] = []
        self.labels: dict[str, dict[str, str]] = {}

    def available(self) -> bool:
        return True

    def _record(self, *argv: str) -> None:
        self.calls.append(list(argv))
        if self.fail_on and self.fail_on in " ".join(argv):
            from dsf_deploy.envfile import DeployError
            raise DeployError(f"simulated failure: {' '.join(argv)}")

    def image_exists(self, image):
        return image in self.images

    def image_platform(self, image):
        return self.images.get(image, "")

    def image_label(self, image, key):
        return self.labels.get(image, {}).get(key, "")

    def pull(self, image, platform):
        self._record("pull", image, platform)
        self.images[image] = platform

    def save(self, image, output):
        self._record("save", image, output.name)
        Path(output).write_text(f"image {image}\n")

    def load(self, archive):
        self._record("load", archive.name)
        image = archive.read_text().split(" ", 1)[1].strip()
        self.images[image] = self.platform

    def volume_exists(self, name):
        return name in self.volumes

    def volume_remove(self, name):
        self._record("volume-rm", name)
        self.volumes.discard(name)

    def export_directory(self, image, directory, target):
        self._record("export", image, directory)
        (Path(target) / "bin").mkdir(parents=True, exist_ok=True)

    def run(self, args, capture=False):
        self._record("run", *args)
        if "--env-file" in args:  # 文件在预检的临时目录中，调用返回后即被删除
            env_file = Path(args[args.index("--env-file") + 1])
            self.env_files.append((env_file.stat().st_mode & 0o777, env_file.read_text()))

    def compose(self, project_dir, args, capture=False, env_file=None, files=None):
        self._record("compose", str(project_dir), *args)

    def compose_config(self, project_dir, env_file=None, files=None):
        self._record("compose-config", str(project_dir))
        return Docker().compose_config(project_dir, env_file=env_file, files=files)

    def compose_ps(self, project_dir):
        if self.ps_rows is not None:
            return self.ps_rows
        services = Docker().compose_config(project_dir)["services"]
        rows = []
        for name, definition in services.items():
            if definition.get("restart", "no") == "no" and not definition.get("healthcheck"):
                rows.append({"Service": name, "State": "exited", "ExitCode": 0})
            else:
                rows.append({"Service": name, "State": "running", "Health": "healthy"})
        return rows


class ProjectTestCase(unittest.TestCase):
    """在临时目录中搭一个最小但完整的项目：框架部署文件 + 一个服务单元。"""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="dsf-deploy-")).resolve()
        self.root = self.tmp / "project"
        self.data = self.tmp / "data"
        for relative in ("framework/compose", "framework/deploy", "framework/python/dsf"):
            shutil.copytree(ROOT / relative, self.root / relative,
                            ignore=shutil.ignore_patterns("__pycache__"))
        (self.root / "framework/docker").mkdir(parents=True)
        shutil.copy2(ROOT / "framework/docker/base.Dockerfile", self.root / "framework/docker")
        shutil.copy2(ROOT / layout.COMPOSE_ROOT, self.root / layout.COMPOSE_ROOT)
        (self.root / "scripts").mkdir()
        shutil.copy2(ROOT / "scripts/install.sh", self.root / "scripts/install.sh")
        (self.root / layout.PROJECT_ENV).write_text(PROJECT_ENV.format(data=self.data))
        (self.root / layout.SITE_FIELDS).write_text(SITE_FIELDS)
        (self.root / "profiles").mkdir()
        (self.root / "profiles/test.yaml").write_text(PROFILE)
        unit = self.root / "services/app"
        (unit / "templates/config").mkdir(parents=True)
        (unit / "compose.yaml").write_text(UNIT_COMPOSE)
        (unit / "templates/config/app.yaml").write_text(UNIT_TEMPLATE)
        (unit / "src").mkdir()
        (unit / "src/main.cpp").write_text("int main() {}\n")
        model = self.root / "models/m/deployment"
        model.mkdir(parents=True)
        (model / "config.txt").write_text("[property]\nonnx-file=../m.onnx\n")
        (model / "x.engine").write_text("device-specific")
        (self.root / layout.SITE_ENV).write_text("SITE_NAME=lab\n")
        self.environ = dict(os.environ)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.environ)
        for path in self.tmp.rglob("*"):  # 测试可能留下只读目录
            if path.is_dir() and not path.is_symlink():
                path.chmod(0o755)
        shutil.rmtree(self.tmp)
