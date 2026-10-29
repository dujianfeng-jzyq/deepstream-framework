"""配置渲染与现场参数。"""

import io
import os
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock
from pathlib import Path

from deploy_support import IMAGE_BUILD, PROFILE, ProjectTestCase, docker_cli_available
from dsf import yamlio
from dsf_deploy import layout, naming, render, site
from dsf_deploy.docker import Docker
from dsf_deploy.envfile import parse, quote_compose, read_compose, unquote_compose
from dsf_deploy.errors import DeployError


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
        profile.write_text(IMAGE_BUILD + "x-app-other:\n  a: {}\n")
        with self.assertRaisesRegex(DeployError, "no matching templates/other"):
            self.plan()
        profile.write_text(IMAGE_BUILD + "x-app-config:\n  missing: {}\n")
        with self.assertRaisesRegex(DeployError, "unknown files"):
            self.plan()
        profile.write_text(IMAGE_BUILD + "  PROJECT_NAME: other\n")
        with self.assertRaisesRegex(DeployError, "x-build cannot set PROJECT_NAME"):  # 身份不可覆盖
            self.plan()

    def test_network_mode_covers_every_service(self):
        profile = self.root / "profiles/test.yaml"
        profile.write_text(PROFILE + "x-network-mode: host\nservices:\n  redis:\n    network_mode: \"service:app\"\n")
        plan = self.plan()
        self.assertTrue(plan.variables["COMPOSE_FILE"].endswith(":profiles/test.yaml:" + layout.NETWORK_COMPOSE))
        overrides = yamlio.load(plan.files[layout.NETWORK_COMPOSE])["services"]
        # deepstream-apps 显式为 none，redis 在 profile 中显式声明：二者保持不变。
        # 其余服务的 /etc/hosts 把每个服务名（包括未覆盖的）指向回环地址，服务名照常可用。
        hosts = [f"{name}:127.0.0.1" for name in ("app", "deepstream-apps", "deepstream-base",
                                                     "model-engine-builder", "redis")]
        self.assertEqual(overrides, {name: {"network_mode": "host", "extra_hosts": hosts}
                                     for name in ("app", "deepstream-base", "model-engine-builder")})
        profile.write_text(PROFILE)
        self.assertNotIn(layout.NETWORK_COMPOSE, self.plan().files)  # 未设置时行为不变

    @unittest.skipUnless(docker_cli_available(), "docker CLI not installed (needed for compose config)")
    def test_network_mode_leaves_no_service_on_the_default_network(self):
        (self.root / "profiles/test.yaml").write_text(PROFILE + "x-network-mode: host\n")
        compose = self.root / "services/app/compose.yaml"
        compose.write_text(compose.read_text().replace("    entrypoint:", "    ports: [\"8554:8554\"]\n    entrypoint:"))
        render.render(self.root, "test", quiet=True)
        # host 网络下端口映射无效且 Docker 会告警：渲染器清空它；没有映射的服务不加 !reset
        self.assertEqual((self.root / layout.NETWORK_COMPOSE).read_text().count("ports: !reset []"), 1)
        config = Docker().compose_config(self.root)
        self.assertNotIn("ports", config["services"]["app"])
        self.assertNotIn("networks", config)  # 没有服务使用默认网络，Compose 不会创建 bridge
        modes = {name: service.get("network_mode") for name, service in config["services"].items()}
        self.assertEqual(modes.pop("deepstream-apps"), "none")
        self.assertEqual(set(modes.values()), {"host"})

    def test_network_mode_is_validated(self):
        profile = self.root / "profiles/test.yaml"
        profile.write_text("x-network-mode: ''\n")
        with self.assertRaisesRegex(DeployError, "x-network-mode must be"):
            self.plan()
        profile.write_text(IMAGE_BUILD + "x-network-mode: host\n")
        compose = self.root / "services/app/compose.yaml"
        compose.write_text(compose.read_text() + "    networks: [default]\n")
        with self.assertRaisesRegex(DeployError, "service redis declares networks"):
            self.plan()

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

    def test_site_answers_may_be_quoted(self):
        (self.root / layout.SITE_ENV).write_text('SITE_NAME="a"b"\nRTSP_PASSWORD=""\nCAMERA_ID=\'c\'\n')
        values = self.plan().variables
        # 与清单相同的写法：KEY="" 是空值而不是两个引号；外层一对引号去掉，内部按字面
        self.assertEqual((values["SITE_NAME"], values["RTSP_PASSWORD"], values["CAMERA_ID"]), ('a"b', "", "c"))

    @unittest.skipUnless(docker_cli_available(), "docker CLI not installed (needed for compose config)")
    def test_special_characters_reach_compose_and_configs_unchanged(self):
        # 未加引号时 Compose 会把 $w 插值为空、" #" 之后当作注释、${ 与开头的引号直接报错。
        for secret in ["p$w #x ${y \\b", "'it's $x \"q\" \\b"]:
            answers = {"SITE_NAME": "lab", "RTSP_PASSWORD": secret}
            site.write_answers(self.root / layout.SITE_ENV, answers)
            render.render(self.root, "test", quiet=True)
            self.assertEqual(site.read_answers(self.root / layout.SITE_ENV, site.parse_fields(
                self.root / layout.SITE_FIELDS)), answers)
            self.assertEqual(read_compose(self.root / layout.ENV)["RTSP_PASSWORD"], secret)
            config = yamlio.load((self.root / "generated/config/app.yaml").read_text())
            self.assertEqual(config["password"], secret)
            probe = self.root / "probe.yaml"
            probe.write_text('services:\n  probe:\n    image: busybox\n    command: ["echo", "${RTSP_PASSWORD}"]\n')
            services = Docker().compose_config(self.root, env_file=self.root / layout.ENV, files=[probe])["services"]
            # compose config 的输出把字面的 $ 写成 $$
            self.assertEqual(services["probe"]["command"][1], secret.replace("$", "$$"))

    def test_each_profile_names_its_deepstream_image(self):
        (self.root / "profiles/test.yaml").write_text("x-build:\n  APPS_VERSION: 1.2.1\n")
        with self.assertRaisesRegex(DeployError, "profile test must set x-build: DEEPSTREAM_IMAGE "
                                                 "\\(project.env has no default\\)"):
            self.plan()
        # 框架仍接受 project.env 中的默认值：派生项目可以让多个 profile 共用一个版本。
        env = self.root / layout.PROJECT_ENV
        env.write_text(env.read_text() + "DEEPSTREAM_IMAGE=nvcr.io/nvidia/deepstream:7.1-samples-multiarch\n")
        self.assertEqual(self.plan().variables["DEEPSTREAM_VERSION"], "7.1")

    def test_platform_is_validated(self):
        profile = self.root / "profiles/test.yaml"
        profile.write_text("x-platform: mips\n")
        with self.assertRaisesRegex(DeployError, "x-platform must be one of amd64, arm64"):
            self.plan()
        profile.write_text(IMAGE_BUILD + "x-platform: arm64\n")
        self.plan()  # 开发渲染不按宿主检查：arm64 的 profile 可以在任何机器上渲染

    def test_release_render_rejects_profiles_for_another_package(self):
        env = self.root / layout.PROJECT_ENV
        env.write_text(env.read_text() + "BASE_IMAGE=demo/base:fixed\nAPPS_IMAGE=demo/apps:fixed\n"
                       "TARGET_PLATFORM=amd64\nDEEPSTREAM_VERSION=8.0\n")
        profiles = self.root / "profiles"
        (profiles / "server.yaml").write_text("x-platform: amd64\n")
        (profiles / "test.yaml").write_text("x-platform: arm64\n")
        with self.assertRaisesRegex(DeployError, "profile test does not fit this package: it targets arm64, "
                                                 "the images are amd64; profiles that fit: server"):
            self.plan()
        (profiles / "test.yaml").write_text("x-platform: amd64\nx-build:\n  DEEPSTREAM_IMAGE: nvcr.io/x:7.1-y\n")
        with self.assertRaisesRegex(DeployError, "it uses DeepStream 7.1, the images have 8.0"):
            self.plan()
        (profiles / "test.yaml").write_text("x-platform: amd64\nx-build:\n  DEEPSTREAM_IMAGE: nvcr.io/x:8.0-y\n")
        self.assertEqual(self.plan().profile, "test")

    def test_fitting_profiles_read_the_platform_without_yaml(self):
        profiles = self.root / "profiles"
        (profiles / "a.yaml").write_text("# profile：甲\nx-platform: amd64\n")
        (profiles / "b.yaml").write_text("# profile：乙\nx-platform: 'arm64'  # Jetson\nservices: {}\n")
        # 安装宿主可能没有 PyYAML：列出可选 profile 时不能解析 YAML。
        with mock.patch.object(render, "_yaml", side_effect=AssertionError("parsed YAML")):
            self.assertEqual(render.fitting_profiles(self.root, "amd64"), [("a", "甲"), ("test", "测试方案")])
            self.assertEqual([name for name, _ in render.fitting_profiles(self.root, "arm64")], ["b", "test"])

    def test_base_tag_follows_its_build_inputs(self):
        before = self.plan().variables["BASE_IMAGE"]
        env = self.root / layout.PROJECT_ENV
        env.write_text(env.read_text().replace("BASE_APT_PACKAGES=", "BASE_APT_PACKAGES=libfoo"))
        self.assertNotEqual(self.plan().variables["BASE_IMAGE"], before)

    def test_release_installations_keep_packaged_images(self):
        env = self.root / layout.PROJECT_ENV
        env.write_text(env.read_text() + "BASE_IMAGE=demo/base:fixed\nAPPS_IMAGE=demo/apps:fixed\n"
                       "DEEPSTREAM_IMAGE=nvcr.io/nvidia/deepstream:8.0-samples-multiarch\n")  # 来自 release.env
        profile = self.root / "profiles/test.yaml"
        profile.write_text("x-build:\n  DEEPSTREAM_IMAGE: other:7.1\n")
        plan = self.plan()
        self.assertEqual(plan.variables["BASE_IMAGE"], "demo/base:fixed")
        self.assertEqual(plan.variables["DEEPSTREAM_IMAGE"], "nvcr.io/nvidia/deepstream:8.0-samples-multiarch")
        # 计算渲染计划不输出任何东西：提示随计划返回，由 render() 输出（安装器同样经过它）。
        self.assertEqual(plan.notes, ["images come from the release package; ignoring x-build DEEPSTREAM_IMAGE"])
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as errors:
            render.render(self.root, "test", quiet=True)
        self.assertIn("note: images come from the release package", errors.getvalue())

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
        with redirect_stdout(io.StringIO()) as output:  # 空答案时重新提问
            self.assertEqual(site.ask(required, prompt=lambda _: next(answers)), "value")
        self.assertIn("A 是必填项", output.getvalue())
        default = site.Field("B", "prompt", "d", False)
        self.assertEqual(site.ask(default, prompt=lambda _: ""), "d")
        self.assertEqual(site.ask(default, "inherited", prompt=lambda _: ""), "inherited")  # 继承值优先于清单
        prompts = []
        self.assertEqual(site.ask(required, "inherited", prompt=lambda text: prompts.append(text) or ""),
                         "inherited")  # 有继承值的必填项回车即可
        self.assertIn("[inherited]", prompts[0])
        secret = site.Field("X_TOKEN", "p", "", True)
        self.assertEqual(site.ask(secret, prompt=lambda _: 1 / 0, prompt_secret=lambda _: ""), "")
        prompts = []
        self.assertEqual(site.ask(secret, "saved", prompt_secret=lambda text: prompts.append(text) or ""),
                         "saved")  # 回车保留已有口令，提示中不显示它
        self.assertIn("[***]", prompts[0])
        self.assertIn(site.CLEAR, prompts[0])
        self.assertEqual(site.ask(secret, "saved", prompt_secret=lambda _: site.CLEAR), "")
        self.assertEqual(site.ask(default, prompt=lambda _: site.CLEAR), site.CLEAR)  # 不可留空的字段照字面
        self.assertEqual(site.masked([secret], {"X_TOKEN": "s"}), "X_TOKEN=***")

    def test_defaults_follow_inherited_values(self):
        fields = [site.Field("A", "a", "manifest", False), site.Field("B", "b", "", True)]
        self.assertEqual(site.defaults(fields, {"A": "profile", "OTHER": "x"}), {"A": "profile", "B": ""})


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

    def test_compose_quoting_round_trips(self):
        for value in ["", "plain", "p$w #x ${y", "it's", "'", '"', "a\\b", "~/x", "$$"]:
            self.assertEqual(unquote_compose(quote_compose(value)), value)
            self.assertEqual(parse(f"K={quote_compose(value)}\n", expand_home=False, unquote=unquote_compose)["K"], value)
        self.assertEqual(quote_compose("p$w"), "'p$w'")
        self.assertEqual(quote_compose("it's $x"), '"it\'s \\$x"')
        self.assertEqual(unquote_compose("legacy"), "legacy")  # 旧版本渲染的未加引号的值
        with self.assertRaises(DeployError):
            quote_compose("a\nb")

    def test_env_files_are_literal(self):
        values = parse("A='q'\nB=${X}\nC=~/x\n# c\n", expand_home=False)
        self.assertEqual(values, {"A": "'q'", "B": "${X}", "C": "~/x"})
        self.assertEqual(parse("C=~/x\n")["C"], os.path.expanduser("~/x"))
        with self.assertRaises(DeployError):
            parse("not a pair\n")


if __name__ == "__main__":
    unittest.main()
