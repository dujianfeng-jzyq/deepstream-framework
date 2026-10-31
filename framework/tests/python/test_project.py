"""项目脚手架：生成服务单元骨架、初始化派生项目。"""

import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from dsf_deploy import layout, project
from dsf_deploy.errors import DeployError

ROOT = Path(__file__).resolve().parents[3]


class NewUnitTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="dsf-project-"))
        self.addCleanup(shutil.rmtree, self.root)
        shutil.copy2(ROOT / layout.COMPOSE_ROOT, self.root / layout.COMPOSE_ROOT)
        (self.root / layout.PLUGINS / "fast_codec").mkdir(parents=True)
        infra = self.root / layout.SERVICES / "infra"
        infra.mkdir(parents=True)
        (infra / layout.UNIT_COMPOSE).write_text(
            "# 顶层注释不结束 services 段\nservices:\n  cache:\n    image: redis\n# 段内的顶层注释\n"
            "  broker:\n    image: nats\nvolumes:\n  data:\n")

    def new_unit(self, name: str, kind: str = "cpp") -> Path:
        with redirect_stdout(StringIO()):
            return project.new_unit(self.root, name, kind)

    def test_generates_units_with_names_substituted(self):
        for name, kind in (("lane-watch", "cpp"), ("event-sink", "python")):
            with self.subTest(kind=kind):
                target = self.new_unit(name, kind)
                files = [p for p in target.rglob("*") if p.is_file()]
                self.assertIn(target / "templates/config" / f"{name}.yaml", files)
                for path in files:
                    for placeholder in ("__NAME__", "__IDENT__"):  # 路径与内容中的占位符都已替换
                        self.assertNotIn(placeholder, str(path.relative_to(target)))
                        self.assertNotIn(placeholder, path.read_text(encoding="utf-8"), path)
        self.assertIn("namespace lane_watch", (self.root / "services/lane-watch/src/app.h").read_text())
        self.assertIn("runtime: nvidia", (self.root / "services/lane-watch/compose.yaml").read_text())

    def test_refuses_names_of_existing_compose_services(self):
        # 同名服务会被 Compose 合并：新单元会悄悄改写框架的 engine 构建或其他单元的服务。
        for name in ("model-engine-builder", "deepstream-apps", "cache", "broker", "infra"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(DeployError, "already"):
                    self.new_unit(name, "python")
                self.assertFalse((self.root / layout.SERVICES / name / "main.py").exists())

    def test_refuses_identifiers_used_by_units_or_plugins(self):
        with self.assertRaisesRegex(DeployError, "plugins/fast_codec .*CMake targets would clash"):
            self.new_unit("fast-codec")
        self.new_unit("lane-watch")
        with self.assertRaisesRegex(DeployError, "already"):
            self.new_unit("lane-watch", "python")

    def test_reserves_the_framework_prefix(self):
        for name in ("dsf", "dsf-test-config"):
            with self.subTest(name=name), self.assertRaisesRegex(DeployError, "reserved for the framework"):
                self.new_unit(name)

    def test_rejects_invalid_names_and_kinds(self):
        with self.assertRaisesRegex(DeployError, "invalid service name"):
            self.new_unit("Lane_Watch")
        with self.assertRaisesRegex(DeployError, "unknown unit kind"):
            self.new_unit("lane-watch", "rust")


class InitProjectTest(unittest.TestCase):
    def test_renames_the_project_and_removes_stale_render_output(self):
        root = Path(tempfile.mkdtemp(prefix="dsf-project-"))
        self.addCleanup(shutil.rmtree, root)
        (root / layout.PROJECT_ENV).write_text("# 项目名\nPROJECT_NAME=old\nAPPS_VERSION=1.0.0\n")
        (root / layout.ENV).write_text("PROJECT_NAME=old\n")
        (root / layout.GENERATED / "config").mkdir(parents=True)
        with redirect_stdout(StringIO()):
            project.init_project(root, "acme-vision")
        self.assertEqual((root / layout.PROJECT_ENV).read_text(),
                         "# 项目名\nPROJECT_NAME=acme-vision\nAPPS_VERSION=1.0.0\n")
        self.assertFalse((root / layout.ENV).exists())
        self.assertFalse((root / layout.GENERATED).exists())
        with self.assertRaisesRegex(DeployError, "invalid project name"):
            project.init_project(root, "Acme")


if __name__ == "__main__":
    unittest.main()
