"""配置渲染与现场参数。"""

import os
import unittest
from pathlib import Path

from deploy_support import ProjectTestCase
from dsf_deploy import layout, naming, render, site
from dsf_deploy.envfile import DeployError, parse


class RenderTest(ProjectTestCase):
    def plan(self, profile="test"):
        return render.build_plan(self.root, profile)

    def test_variables_follow_the_priority_order(self):
        plan = self.plan()
        values = plan.variables
        self.assertEqual(values["APPS_VERSION"], "1.2.1")          # profile 覆盖 project.env
        self.assertEqual(values["CAMERA_ID"], "cam0")             # 清单默认值
        self.assertEqual(values["SITE_NAME"], "lab")              # 现场答案
        self.assertEqual(values["DEEPSTREAM_VERSION"], "8.0")
        self.assertEqual(values["APPS_IMAGE"], "demo/deepstream-8.0-apps:1.2.1")
        self.assertTrue(values["BASE_IMAGE"].startswith("demo/deepstream-8.0-base:1.0.0-"))
        self.assertEqual(values["CONTAINER_DATA_PATH"], "/data")
        self.assertEqual(values["HOST_UID"], str(os.getuid()))
        self.assertEqual(values["COMPOSE_FILE"],
                         "docker-compose.yaml:services/app/compose.yaml:profiles/test.yaml")

    def test_templates_are_substituted_and_overridden(self):
        content = self.plan().files["generated/config/app.yaml"]
        self.assertIn("camera: cam0", content)
        self.assertIn("password: ''", content)
        self.assertIn("keep: 1", content)           # 映射递归合并
        self.assertIn("list:\n  - 3", content)       # 列表整体替换

    def test_profile_selection(self):
        with self.assertRaisesRegex(DeployError, "no profile selected(.|\n)*test  测试方案"):
            render.build_plan(self.root, None)
        render.render(self.root, "test", quiet=True)
        self.assertEqual(render.build_plan(self.root, None).profile, "test")  # 沿用 .env 的选择
        with self.assertRaisesRegex(DeployError, "unknown profile 'nope'"):
            self.plan("nope")

    def test_profile_structure_is_validated(self):
        profile = self.root / "profiles/test.yaml"
        profile.write_text("DEEPSTREAM_IMAGE: x\n")
        with self.assertRaisesRegex(DeployError, "unknown top-level keys"):
            self.plan()
        profile.write_text("x-app-other:\n  a: {}\n")
        with self.assertRaisesRegex(DeployError, "no matching templates/other"):
            self.plan()
        profile.write_text("x-app-config:\n  missing: {}\n")
        with self.assertRaisesRegex(DeployError, "unknown files"):
            self.plan()
        profile.write_text("x-build:\n  PROJECT_NAME: other\n")
        self.assertEqual(self.plan().variables["PROJECT_NAME"], "demo")  # 身份不可覆盖

    def test_undefined_placeholders_fail(self):
        (self.root / "services/app/templates/config/app.yaml").write_text("x: ${NOPE}\n")
        with self.assertRaisesRegex(DeployError, r"undefined variable \$\{NOPE\}"):
            self.plan()

    def test_duplicate_yaml_keys_fail(self):
        (self.root / "services/app/templates/config/app.yaml").write_text("a: 1\na: 2\n")
        with self.assertRaisesRegex(DeployError, "duplicate key"):
            self.plan()

    def test_units_cannot_render_the_same_file(self):
        other = self.root / "services/other/templates/config"
        other.mkdir(parents=True)
        (other.parent.parent / "compose.yaml").write_text("services: {}\n")
        (other / "app.yaml").write_text("x: 1\n")
        with self.assertRaisesRegex(DeployError, "both render generated/config/app.yaml"):
            self.plan()

    def test_site_answers(self):
        answers = self.root / layout.SITE_ENV
        answers.write_text("SITE_NAME=lab\nUNKNOWN=1\n")
        with self.assertRaisesRegex(DeployError, "undeclared site fields"):
            self.plan()
        answers.write_text("CAMERA_ID=\nSITE_NAME=lab\n")
        with self.assertRaisesRegex(DeployError, "CAMERA_ID must not be empty"):
            self.plan()
        answers.write_text("")
        with self.assertRaisesRegex(DeployError, "not answered: SITE_NAME"):
            self.plan()
        (self.root / layout.PROJECT_ENV).write_text((self.root / layout.PROJECT_ENV).read_text()
                                                    + "SITE_NAME=from-project\n")
        self.assertEqual(self.plan().variables["SITE_NAME"], "from-project")  # 继承低优先级值

    def test_base_tag_follows_its_build_inputs(self):
        before = self.plan().variables["BASE_IMAGE"]
        env = self.root / layout.PROJECT_ENV
        env.write_text(env.read_text().replace("BASE_APT_PACKAGES=", "BASE_APT_PACKAGES=libfoo"))
        self.assertNotEqual(self.plan().variables["BASE_IMAGE"], before)

    def test_release_installations_keep_packaged_images(self):
        env = self.root / layout.PROJECT_ENV
        env.write_text(env.read_text() + "BASE_IMAGE=demo/base:fixed\nAPPS_IMAGE=demo/apps:fixed\n")
        profile = self.root / "profiles/test.yaml"
        profile.write_text("x-build:\n  DEEPSTREAM_IMAGE: other:7.1\n")
        values = self.plan().variables
        self.assertEqual(values["BASE_IMAGE"], "demo/base:fixed")
        self.assertEqual(values["DEEPSTREAM_IMAGE"], "nvcr.io/nvidia/deepstream:8.0-samples-multiarch")

    def test_write_replaces_outputs_and_reports_stale_files(self):
        stale = self.root / "generated/config/old.yaml"
        stale.parent.mkdir(parents=True)
        stale.write_text("x")
        removed = render.write_plan(self.root, self.plan())
        self.assertEqual(removed, ["generated/config/old.yaml"])
        self.assertFalse(stale.exists())
        self.assertEqual((self.root / ".env").stat().st_mode & 0o777, 0o600)
        self.assertFalse(list(self.root.glob(".render-*")))

    def test_failed_validation_keeps_previous_outputs(self):
        render.render(self.root, "test", quiet=True)
        before = (self.root / ".env").read_text()
        (self.root / "services/app/templates/config/app.yaml").write_text("x: ${NOPE}\n")
        with self.assertRaises(DeployError):
            render.render(self.root, "test", quiet=True)
        self.assertEqual((self.root / ".env").read_text(), before)


class SiteFieldsTest(unittest.TestCase):
    def fields(self, text):
        path = Path(self._testMethodName + ".env")
        path.write_text(text)
        try:
            return site.parse_fields(path)
        finally:
            path.unlink()

    def test_prompts_defaults_and_secrets(self):
        fields = self.fields("# 文件说明\n\n# 第一行\n# 第二行\nA=1\nB_PASSWORD=\"\"\nC=\n")
        self.assertEqual(fields[0].prompt, "第一行 第二行")
        self.assertEqual(fields[1].prompt, "B_PASSWORD")
        self.assertTrue(fields[1].nullable and fields[1].secret and not fields[1].required)
        self.assertTrue(fields[2].required)

    def test_reserved_keys_are_rejected(self):
        for key in ("PROJECT_NAME", "REDIS_IMAGE", "APPS_VERSION", "CONTAINER_X", "COMPOSE_FILE"):
            with self.assertRaisesRegex(DeployError, "cannot be a site field"):
                self.fields(f"{key}=x\n")

    def test_ask(self):
        required = site.Field("A", "prompt", "", False)
        answers = iter(["", "value"])
        self.assertEqual(site.ask(required, prompt=lambda _: next(answers)), "value")
        default = site.Field("B", "prompt", "d", False)
        self.assertEqual(site.ask(default, prompt=lambda _: ""), "d")
        secret = site.Field("X_TOKEN", "p", "", True)
        self.assertEqual(site.ask(secret, prompt=lambda _: 1 / 0, prompt_secret=lambda _: ""), "")
        self.assertEqual(site.masked([secret], {"X_TOKEN": "s"}), "X_TOKEN=***")


class NamingTest(unittest.TestCase):
    def test_names(self):
        self.assertEqual(naming.deepstream_version("nvcr.io/nvidia/deepstream:7.1-triton"), "7.1")
        self.assertEqual(naming.deepstream_version("localhost:5000/ds"), "")
        self.assertEqual(naming.deepstream_version("x:latest"), "")
        self.assertEqual(naming.image_tar_name("localhost:5000/a/b:1.0"), "localhost-5000-a-b-1.0.tar")
        self.assertEqual(naming.image_tar_name("redis"), "redis.tar")
        self.assertEqual(naming.package_name("p", "1.0", ""), "p-v1.0")
        self.assertEqual(naming.docker_arch("aarch64"), "arm64")
        with self.assertRaises(DeployError):
            naming.docker_arch("mips")

    def test_env_files_are_literal(self):
        values = parse("A='q'\nB=${X}\nC=~/x\n# c\n", expand_home=False)
        self.assertEqual(values, {"A": "'q'", "B": "${X}", "C": "~/x"})
        self.assertEqual(parse("C=~/x\n")["C"], os.path.expanduser("~/x"))
        with self.assertRaises(DeployError):
            parse("not a pair\n")


if __name__ == "__main__":
    unittest.main()
