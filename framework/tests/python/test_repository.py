"""仓库约定：框架与产品的边界、镜像契约、脚本语法与 Python 静态检查。"""

import importlib.util
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FRAMEWORK = ROOT / "framework"


def product_names() -> set[str]:
    """带代码的服务单元、插件与模型的名字：框架代码不得引用它们。
    只有 compose.yaml 的基础设施单元（如 redis）与框架支持的技术同名，不计入。"""
    names = set()
    for parent in ("services", "plugins"):
        names |= {p.name for p in (ROOT / parent).glob("*/CMakeLists.txt") for p in [p.parent]}
    names |= {p.name for p in (ROOT / "models").glob("*") if p.is_dir()}  # 镜像构建上下文中没有 models/
    return names


class RepositoryTest(unittest.TestCase):
    def test_framework_does_not_reference_products(self):
        names = product_names()
        offenders = []
        for path in FRAMEWORK.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix in (".pyc",):
                continue
            if path.is_relative_to(FRAMEWORK / "tests"):
                continue  # 测试可以用产品作为样例
            text = path.read_text(encoding="utf-8", errors="ignore")
            for name in names:
                if re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", text):
                    offenders.append(f"{path.relative_to(ROOT)}: {name}")
        self.assertEqual(offenders, [])

    def test_apps_image_contract(self):
        text = (FRAMEWORK / "docker/apps.Dockerfile").read_text()
        stages = re.findall(r"^FROM\s+(\S+)", text, re.M)
        self.assertEqual(stages[:2], ["${BASE_IMAGE}", "${BASE_IMAGE}"])
        self.assertRegex(stages[-1], r"^busybox:\d")  # 载荷不继承 DeepStream 层
        self.assertIn("COPY --from=runtime-check /workspace /workspace", text)  # 只复制自检过的载荷
        self.assertIn("USER 65534:65534", text)
        self.assertIn("--network=none", text)
        self.assertIn("-DDSF_REQUIRE_DEEPSTREAM=ON", text)
        for dockerfile in ("apps.Dockerfile", "base.Dockerfile"):
            content = (FRAMEWORK / "docker" / dockerfile).read_text()
            self.assertTrue(content.startswith("# check=skip=InvalidDefaultArgInFrom"))
            self.assertNotRegex(content, r"ARG (DEEPSTREAM|BASE)_IMAGE=")  # 上游镜像必须由 Compose 传入

    def test_business_services_use_the_base_image_and_readonly_payload(self):
        text = (FRAMEWORK / "compose/defaults.yaml").read_text()
        self.assertIn("image: ${BASE_IMAGE}", text)
        self.assertIn("app-files:${CONTAINER_WORKSPACE}:ro", text)
        self.assertIn("../../generated/config:${CONTAINER_CONFIG_PATH}:ro", text)
        self.assertIn('io.deepstream.config-check: "true"', text)
        for compose in (ROOT / "services").glob("*/compose.yaml"):
            self.assertNotIn("container_name", compose.read_text(), compose)

    def test_shell_scripts_parse(self):
        scripts = [*ROOT.glob("scripts/*.sh"), *ROOT.glob("tools/*.sh")]
        self.assertTrue(scripts)
        for script in scripts:
            result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, f"{script}: {result.stderr}")

    def test_deploy_tool_rejects_old_python_with_a_clear_message(self):
        code = ("import runpy, sys; sys.version_info = (3, 8, 10, 'final', 0); sys.argv = ['dsf-deploy', '--help']; "
                "runpy.run_path(sys.argv_path, run_name='__main__')").replace("sys.argv_path", repr(
                    str(FRAMEWORK / "deploy/dsf-deploy.py")))
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("Python 3.9+ is required", result.stderr)

    def test_python_passes_pyflakes(self):
        if importlib.util.find_spec("pyflakes") is None:
            self.skipTest("pyflakes not installed")
        targets = [FRAMEWORK / "python", FRAMEWORK / "deploy", FRAMEWORK / "tools", FRAMEWORK / "tests",
                   FRAMEWORK / "templates", ROOT / "services"]
        result = subprocess.run([sys.executable, "-m", "pyflakes", *map(str, targets)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_deploy_tools_need_only_the_standard_library(self):
        """安装器在没有 PyYAML 的目标机上也必须能启动。"""
        code = ("import sys; sys.modules['yaml'] = None; sys.path[:0] = [sys.argv[1]]; "
                "import dsf_deploy.cli, dsf_deploy.install, dsf_deploy.pack")
        result = subprocess.run([sys.executable, "-c", code, str(FRAMEWORK / "deploy")],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
