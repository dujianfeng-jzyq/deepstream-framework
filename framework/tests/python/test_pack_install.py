"""打包与安装：用假 Docker 走完整流程，验证顺序、内容与安全规则。"""

import contextlib
import io
import json
import os
import subprocess
import tarfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from deploy_support import PROFILE, FakeDocker, ProjectTestCase, docker_cli_available
from dsf_deploy import cli, inventory, layout, naming, pack, render, site
from dsf_deploy.envfile import read, unquote_answer
from dsf_deploy.errors import DeployError
from dsf_deploy.install import Options, install, preflight, ready
from dsf_deploy.install.apply import apply


def answers(install_dir: Path) -> dict:
    """安装器写入的现场答案（KEY="值"）。"""
    return read(install_dir / layout.SITE_ENV, unquote=unquote_answer)


@unittest.skipUnless(docker_cli_available(), "docker CLI not installed (needed for compose config)")
class PackInstallTest(ProjectTestCase):
    def setUp(self):
        super().setUp()
        # 非交互安装无法回答必填项：这里给它默认值（必填项的失败另有用例）。
        fields = self.root / layout.SITE_FIELDS
        fields.write_text(fields.read_text().replace("SITE_NAME=\n", "SITE_NAME=lab\n"))
        self.docker = FakeDocker()
        values = render.build_plan(self.root, "test").variables
        self.base, self.apps = values["BASE_IMAGE"], values["APPS_IMAGE"]
        self.platform = naming.host_platform()
        self.other = "arm64" if self.platform == "amd64" else "amd64"
        self.docker.images.update({self.base: self.platform, self.apps: self.platform})
        self.install_dir = self.tmp / "installed"
        os.environ["XDG_DATA_HOME"] = str(self.tmp / "xdg")

    # ---- helpers ----

    def pack(self, with_base=False) -> Path:
        with redirect_stdout(io.StringIO()):
            archive = pack.pack(self.root, "test", with_base, self.docker)
        self.packed = getattr(self, "packed", 0) + 1
        target = self.tmp / f"unpacked-{self.packed}"
        with tarfile.open(archive) as tar:
            tar.extractall(target)
        return target / archive.name[: -len(".tar.gz")]

    def install(self, package, target=None, images=None, **options):
        """images：目标机上已有的镜像（镜像名 → 平台），默认没有。"""
        target = target or FakeDocker()
        target.images = dict(images or {})
        options.setdefault("install_dir", self.install_dir)
        options.setdefault("interactive", False)
        options.setdefault("ready_timeout", 5)
        output = io.StringIO()
        with redirect_stdout(output):
            install(Options(package, **options), target)
        return target, output.getvalue()

    def press_enter(self, prompts, replies=None):
        """交互提问：replies 中的键输入对应的值，其余回车；提示文本记入 prompts。"""
        real_ask = site.ask

        def answer(item, default=None):
            def record(text):
                prompts.append(text)
                return (replies or {}).get(item.key, "")
            return real_ask(item, default, prompt=record, prompt_secret=record)

        return mock.patch.object(site, "ask", answer)

    def set_profile_build(self, name, values):
        """写 profiles/<name>.yaml：在测试 profile 之上加 x-build 变量。"""
        build = "".join(f"  {key}: {value}\n" for key, value in values.items())
        (self.root / f"profiles/{name}.yaml").write_text(PROFILE.replace("x-build:\n", "x-build:\n" + build))

    # ---- pack ----

    def test_package_contents(self):
        package = self.pack()
        files = json.loads((package / layout.INVENTORY).read_text())
        self.assertIn("install.sh", files)
        self.assertIn("framework/deploy/dsf-deploy.py", files)
        self.assertIn("services/app/compose.yaml", files)
        self.assertIn("services/app/templates/config/app.yaml", files)
        self.assertIn("models/m/deployment/config.txt", files)
        for excluded in ("services/app/src/main.cpp", "models/m/deployment/x.engine", "site.env", ".env",
                         "generated/config/app.yaml", "framework/docker/base.Dockerfile", layout.BUILD_COMPOSE):
            self.assertNotIn(excluded, files)
        saved = sorted(call[1] for call in self.docker.calls if call[0] == "save")
        self.assertEqual(saved, sorted([self.apps, "redis:7.4.2-alpine"]))  # 依赖镜像按约定收集、缺失时拉取
        self.assertIn(["pull", "redis:7.4.2-alpine", self.platform], self.docker.calls)
        release = read(package / layout.RELEASE_ENV, expand_home=False)
        self.assertEqual(release["APPS_VERSION"], "1.2.1")   # 取渲染结果（含 profile 覆盖）
        self.assertEqual(release["TARGET_PLATFORM"], self.platform)
        self.assertEqual(release["CONFIG_PROFILE"], "test")
        self.assertNotIn("SITE_NAME", release)              # 现场答案不进包
        self.assertTrue(package.name.endswith("demo-v1.2.1-ds8.0"))

    def test_pack_refuses_an_apps_image_built_from_other_source(self):
        def git(*args):
            subprocess.run(["git", "-C", str(self.root), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                           check=True, capture_output=True)
        (self.root / ".gitignore").write_text("/.env\n/site.env\n/generated/\n/dist/\n")
        git("init", "-q")
        git("add", "-A")
        git("commit", "-q", "-m", "initial")
        revision = naming.source_revision(self.root)
        self.assertRegex(revision, r"^[0-9a-f]{12}$")
        with self.assertRaisesRegex(DeployError, "built from source \\(unknown\\) but the source is now"):
            self.pack()  # 镜像没有记录版本：可能是旧构建
        self.docker.labels[self.apps] = {layout.SOURCE_REVISION_LABEL: revision}
        package = self.pack()
        self.assertEqual(read(package / layout.RELEASE_ENV, expand_home=False)["SOURCE_REVISION"], revision)

        (self.root / "services/app/src/main.cpp").write_text("int main() { return 1; }\n")
        with self.assertRaisesRegex(DeployError, f"built from source {revision} but the source is now {revision}-dirty"):
            self.pack()  # 改了源码却没有重新构建
        self.docker.labels[self.apps] = {layout.SOURCE_REVISION_LABEL: f"{revision}-dirty"}
        errors = io.StringIO()
        with redirect_stderr(errors):
            self.pack()
        self.assertIn("uncommitted changes", errors.getvalue())

    def test_pack_refuses_a_profile_for_another_platform(self):
        profile = self.root / "profiles/test.yaml"
        profile.write_text(f"x-platform: {self.other}\n" + profile.read_text())
        with self.assertRaisesRegex(DeployError, f"profile test does not fit the built images: "
                                                 f"it targets {self.other}, the images are {self.platform}"):
            self.pack()
        self.assertFalse(any(call[0] == "save" for call in self.docker.calls))  # 检查在保存镜像之前

    def test_base_is_packed_on_request_and_platforms_must_match(self):
        self.pack(with_base=True)
        self.assertIn(self.base, [call[1] for call in self.docker.calls if call[0] == "save"])
        self.docker.images["redis:7.4.2-alpine"] = self.other
        with self.assertRaisesRegex(DeployError, "must share one platform"):
            self.pack()

    # ---- install ----

    def test_fresh_install(self):
        package = self.pack(with_base=True)
        target, output = self.install(package)
        root = self.install_dir
        self.assertEqual(answers(root), {})  # 非交互安装只保存已有答案
        self.assertTrue((root / "generated/config/app.yaml").is_file())
        self.assertEqual((root / layout.SITE_ENV).stat().st_mode & 0o777, 0o600)
        project = read(root / layout.PROJECT_ENV, expand_home=False)
        self.assertEqual(project["APPS_IMAGE"], self.apps)
        managed = json.loads((root / layout.MANAGED).read_text())
        self.assertIn("services/app/compose.yaml", managed)
        self.assertNotIn("install.sh", managed)
        self.assertTrue(self.data.is_dir())  # 数据目录按最终 Compose 准备
        # 安装的项目没有构建定义：docker compose run/up 不会尝试构建包中没有的 Dockerfile。
        services = target.compose_config(root, env_file=root / layout.ENV)["services"]
        self.assertEqual([name for name, service in services.items() if "build" in service], [])
        # 预检用新载荷、关闭网络执行带标签服务的 --check-config
        check = next(call for call in target.calls if call[0] == "run" and "--check-config" in call)
        self.assertIn("--network", check)
        self.assertEqual(check[check.index("--network") + 1], "none")
        # 环境变量经 0600 文件传入，不出现在命令行（进程列表）中
        self.assertNotIn("-e", check)
        self.assertEqual(len(target.env_files), 1)
        mode, environment = target.env_files[0]
        self.assertEqual(mode, 0o600)
        self.assertIn("DSF_READY_FILE=/tmp/dsf.ready\n", environment)
        # 首次安装：没有 down；最后 up -d
        compose = [call[2:] for call in target.calls if call[0] == "compose"]
        self.assertFalse([call for call in compose if call[:1] == ["down"]])
        self.assertEqual(compose[-1], ["up", "-d"])
        self.assertIn("all services are ready", output)

    def test_missing_required_answers_fail_before_anything_changes(self):
        (self.root / layout.SITE_FIELDS).write_text("SITE_NAME=\nCAMERA_ID=c\nRTSP_PASSWORD=\"\"\n")
        package = self.pack(with_base=True)
        with self.assertRaisesRegex(DeployError, "not answered: SITE_NAME"):
            self.install(package)
        self.assertFalse(self.install_dir.exists())

    def test_dry_run_changes_nothing(self):
        package = self.pack(with_base=True)
        target, output = self.install(package, dry_run=True)
        self.assertFalse(self.install_dir.exists())
        self.assertIn("--dry-run: nothing was stopped or written", output)
        self.assertFalse([c for c in target.calls if c[0] == "compose" and "up" in c])

    def test_upgrade_stops_old_services_and_prunes_unmodified_obsolete_files(self):
        package = self.pack(with_base=True)
        self.install(package)
        (self.install_dir / "notes.txt").write_text("site file")  # 现场文件保留
        (self.root / "models/m/deployment/config.txt").unlink()
        (self.root / "models/m/deployment/new.txt").write_text("[property]\n")
        upgraded = self.pack(with_base=True)
        target = FakeDocker()
        target.volumes = {"demo_app-files", "demo_redis-data"}
        target, _ = self.install(upgraded, target)
        compose = [call[2:] for call in target.calls if call[0] == "compose"]
        self.assertEqual(compose[0], ["down", "--remove-orphans"])  # 先用旧定义停服；不带 -v
        # 只删除载荷卷使其重新播种，服务的其他命名卷保留
        self.assertIn(["volume-rm", "demo_app-files"], target.calls)
        self.assertEqual(target.volumes, {"demo_redis-data"})
        self.assertFalse((self.install_dir / "models/m/deployment/config.txt").exists())
        self.assertTrue((self.install_dir / "models/m/deployment/new.txt").exists())
        self.assertTrue((self.install_dir / "notes.txt").exists())

    def test_first_install_removes_a_leftover_payload_volume(self):
        # 删掉安装目录后重装：遗留的载荷卷不删就会继续运行旧程序。
        package = self.pack(with_base=True)
        target = FakeDocker()
        target.volumes = {"demo_app-files", "demo_redis-data"}
        _, output = self.install(package, target, dry_run=True)
        self.assertIn("remove: leftover payload volume demo_app-files", output)
        self.assertEqual(target.volumes, {"demo_app-files", "demo_redis-data"})
        self.install(package, target)
        removed = target.calls.index(["volume-rm", "demo_app-files"])
        started = next(i for i, call in enumerate(target.calls) if call[0] == "compose" and "up" in call)
        self.assertLess(removed, started)
        self.assertEqual(target.volumes, {"demo_redis-data"})

    def test_refuses_to_take_over_the_project_started_elsewhere(self):
        # 开发仓库的栈与安装共用工程名：在这里 up 会接管并重建它的容器。
        package = self.pack(with_base=True)
        target = FakeDocker()
        target.containers = [("demo-app-1", str(self.root)), ("demo-redis-1", str(self.root))]
        for dry_run in (True, False):
            with self.subTest(dry_run=dry_run), \
                    self.assertRaisesRegex(DeployError, f"started from {self.root}; installing here.*\n"
                                                        f"  stop them first: cd {self.root} && docker compose down$"):
                self.install(package, target, dry_run=dry_run)
        target.containers = [("demo-app-1", "")]  # 容器缺少启动目录的标签：只能按工程名停止
        with self.assertRaisesRegex(DeployError, "started from \\(unknown directory\\);.*\n"
                                                 "  stop them first: docker compose -p demo down$"):
            self.install(package, target, dry_run=True)
        target.containers = [("demo-app-1", str(self.root))]
        self.assertFalse(self.install_dir.exists())
        self.assertFalse([call for call in target.calls if call[0] in ("compose", "load", "volume-rm")])
        self.install(package)
        target.containers = [("demo-app-1", str(self.install_dir))]  # 本安装启动的容器：照常升级
        self.install(package, target)

    def test_obsolete_file_removed_after_preflight_does_not_abort_the_upgrade(self):
        package = self.pack(with_base=True)
        self.install(package)
        (self.root / "models/m/deployment/config.txt").unlink()
        upgraded = self.pack(with_base=True)
        target = FakeDocker()
        with redirect_stdout(io.StringIO()):
            plan = preflight.prepare(Options(upgraded, self.install_dir, interactive=False), target)
            obsolete = self.install_dir / "models/m/deployment/config.txt"
            self.assertEqual(plan.obsolete, [obsolete])
            obsolete.unlink()  # 现场在预检与执行之间删除了它
            apply(plan, target, 5)
        self.assertEqual([call[2:] for call in target.calls if call[0] == "compose"][-1], ["up", "-d"])

    def test_modified_obsolete_file_blocks_the_upgrade_before_stopping(self):
        package = self.pack(with_base=True)
        self.install(package)
        (self.install_dir / "models/m/deployment/config.txt").write_text("edited on site")
        (self.root / "models/m/deployment/config.txt").unlink()
        upgraded = self.pack(with_base=True)
        with self.assertRaisesRegex(DeployError, "modified locally"):
            self.install(upgraded)

    def test_clean_install(self):
        package = self.pack(with_base=True)
        self.install(package)
        (self.install_dir / "notes.txt").write_text("x")
        target, _ = self.install(package, clean=True)
        self.assertFalse((self.install_dir / "notes.txt").exists())
        foreign = self.tmp / "foreign"
        foreign.mkdir()
        (foreign / "important").write_text("x")
        with self.assertRaisesRegex(DeployError, "does not look like an installation"):
            self.install(package, clean=True, install_dir=foreign)

    def test_saved_answers_are_reused(self):
        package = self.pack(with_base=True)
        self.install(package)
        (self.install_dir / layout.SITE_ENV).write_text("CAMERA_ID=site-cam\n")
        self.install(package)
        self.assertIn("camera: site-cam", (self.install_dir / "generated/config/app.yaml").read_text())

    def test_ask_offers_the_saved_answers(self):
        # 现场用 --ask 只改一项：其余回车保留，口令不会因回车被清空。
        package = self.pack(with_base=True)
        self.install(package)
        saved = {"CAMERA_ID": "site-cam", "RTSP_PASSWORD": "s3cret", "SITE_NAME": "custom"}
        site.write_answers(self.install_dir / layout.SITE_ENV, saved)
        prompts = []
        with self.press_enter(prompts):
            self.install(package, ask=True, interactive=True)
        self.assertEqual(answers(self.install_dir), saved)
        self.assertEqual(len(prompts), 3)  # 已有答案的字段也逐项提问
        self.assertIn("[site-cam]", prompts[0])
        self.assertIn("[***]", prompts[1])
        self.assertNotIn("s3cret", "".join(prompts))
        with self.press_enter([], {"CAMERA_ID": "gate-2", "RTSP_PASSWORD": site.CLEAR}):
            self.install(package, ask=True, interactive=True)
        self.assertEqual(answers(self.install_dir), {**saved, "CAMERA_ID": "gate-2", "RTSP_PASSWORD": ""})

    def test_ask_without_a_terminal_fails_before_stopping(self):
        package = self.pack(with_base=True)
        self.install(package)
        before = (self.install_dir / layout.SITE_ENV).read_text()
        target = FakeDocker()
        with self.assertRaisesRegex(DeployError, "--ask needs a terminal"):
            self.install(package, target=target, ask=True)
        self.assertFalse([call for call in target.calls if "down" in call])
        self.assertEqual((self.install_dir / layout.SITE_ENV).read_text(), before)

    def test_upgrade_drops_answers_for_retired_fields(self):
        # 新版本删除了现场字段：升级不能因旧 site.env 中的键失败，丢弃它并重写 site.env。
        package = self.pack(with_base=True)
        self.install(package)
        (self.install_dir / layout.SITE_ENV).write_text('CAMERA_ID="site-cam"\nREDIS_HOST="redis"\n')
        _, output = self.install(package)
        self.assertIn("dropped site fields no longer used by this release: REDIS_HOST", output)
        self.assertEqual(answers(self.install_dir), {"CAMERA_ID": "site-cam"})
        self.assertIn("camera: site-cam", (self.install_dir / "generated/config/app.yaml").read_text())

    def test_interactive_defaults_follow_the_profile(self):
        # 回车保存的答案必须是 profile 的值，否则 site.env 会遮住 profile 的 x-build。
        self.set_profile_build("test", {"CAMERA_ID": "profile-cam"})
        package = self.pack(with_base=True)
        prompts = []
        with self.press_enter(prompts):
            self.install(package, interactive=True, profile="test")
        self.assertTrue(any("[profile-cam]" in text for text in prompts), prompts)
        self.assertEqual(answers(self.install_dir)["CAMERA_ID"], "profile-cam")
        self.assertIn("camera: profile-cam", (self.install_dir / "generated/config/app.yaml").read_text())

    def test_changing_profile_drops_answers_that_were_old_defaults(self):
        self.set_profile_build("other", {"CAMERA_ID": "other-cam", "SITE_NAME": "other-site"})
        package = self.pack(with_base=True)
        self.install(package)
        # 当时回车接受的清单默认值 cam0，与现场输入的 custom
        (self.install_dir / layout.SITE_ENV).write_text("CAMERA_ID=cam0\nSITE_NAME=custom\n")
        _, output = self.install(package, profile="other")
        self.assertIn("profile changed: CAMERA_ID follow the new profile", output)
        self.assertEqual(answers(self.install_dir), {"SITE_NAME": "custom"})
        config = (self.install_dir / "generated/config/app.yaml").read_text()
        self.assertIn("camera: other-cam", config)
        self.assertIn("site: custom", config)
        self.install(package, profile="other")  # profile 不变时答案原样沿用
        self.assertEqual(answers(self.install_dir), {"SITE_NAME": "custom"})

    def test_rendering_runs_in_the_base_image(self):
        self.set_profile_build("test", {"CAMERA_ID": "profile-cam"})
        package = self.pack(with_base=True)
        prompts = []
        with self.press_enter(prompts):
            target, _ = self.install(package, interactive=True, profile="test")
        self.assertTrue(any("[profile-cam]" in text for text in prompts), prompts)
        self.assertIn("camera: profile-cam", (self.install_dir / "generated/config/app.yaml").read_text())
        # 提问前只提前加载 base，之后统一加载时不重复
        loads = [i for i, call in enumerate(target.calls) if call[0] == "load"]
        base = [i for i in loads if target.calls[i][1] == naming.image_tar_name(self.base)]
        runs = {i: call for i, call in enumerate(target.calls)
                if call[0] == "run" and any(arg.endswith("dsf-deploy.py") for arg in call)}
        tools = [next(call[j + 1] for j, arg in enumerate(call) if arg.endswith("dsf-deploy.py"))
                 for call in runs.values()]
        self.assertEqual(tools, ["profile-variables", "render"])
        self.assertEqual(len(base), 1)
        self.assertLess(base[0], min(runs))
        self.assertEqual(len(loads), len(list((package / "images").glob("*.tar"))))
        for call in runs.values():
            # 离线、以安装用户运行；HOME 与宿主一致，渲染展开 ~ 的结果不随容器变化
            self.assertIn(self.base, call)
            self.assertEqual(call[call.index("--network") + 1], "none")
            self.assertEqual(call[call.index("--user") + 1], f"{os.getuid()}:{os.getgid()}")
            self.assertIn(f"HOME={Path.home()}", call)

    def test_package_without_base_renders_with_the_installed_base(self):
        package = self.pack()  # 升级的常态：base 没有变化，包里不带
        with self.assertRaisesRegex(DeployError, "the base image .* is missing; .*--with-base"):
            self.install(package)
        self.assertFalse(self.install_dir.exists())  # 预检失败：什么都没有写
        target, _ = self.install(package, images={self.base: self.platform})
        self.assertIn("camera:", (self.install_dir / "generated/config/app.yaml").read_text())
        self.assertNotIn(["load", naming.image_tar_name(self.base)], target.calls)

    def test_deploy_tool_errors_are_reported_without_the_docker_command(self):
        docker = FakeDocker()
        docker.images[self.base] = self.platform
        with self.assertRaises(DeployError) as caught:
            preflight._run_deploy_tool(self.root, ["render", "--profile", "missing"], self.base, docker,
                                       writable=False, package_root=self.root, loaded=set())
        self.assertIn("unknown profile 'missing'", str(caught.exception))
        self.assertNotIn("docker run", str(caught.exception))
        self.assertFalse(str(caught.exception).startswith("error:"))

    def test_integrity_and_platform_are_checked_first(self):
        package = self.pack(with_base=True)
        (package / "profiles/test.yaml").write_text("tampered\n")
        with self.assertRaisesRegex(DeployError, "corrupted: profiles/test.yaml"):
            self.install(package)
        package = self.pack(with_base=True)
        (package / "extra.txt").write_text("x")
        with self.assertRaisesRegex(DeployError, "does not match its inventory"):
            self.install(package)
        (package / "extra.txt").unlink()
        release = package / layout.RELEASE_ENV
        release.write_text(release.read_text().replace(f"TARGET_PLATFORM={self.platform}", "TARGET_PLATFORM=mips"))
        with self.assertRaisesRegex(DeployError, "this package is for mips"):
            self.install(package)

    def test_unsafe_targets_are_rejected(self):
        package = self.pack(with_base=True)
        self.install_dir.mkdir()
        outside = self.tmp / "outside"
        outside.mkdir()
        (self.install_dir / "profiles").symlink_to(outside)
        with self.assertRaisesRegex(DeployError, "symlink in the installation target"):
            self.install(package)
        (self.install_dir / "profiles").unlink()
        (self.install_dir / "profiles").write_text("not a directory")
        with self.assertRaisesRegex(DeployError, "must be a directory"):
            self.install(package)
        with self.assertRaisesRegex(DeployError, "must not overlap the package"):
            self.install(package, install_dir=package / "inside")

    def test_writable_mounts_may_not_enter_the_installation(self):
        env = self.root / layout.PROJECT_ENV
        env.write_text(env.read_text().replace(f"HOST_DATA_PATH={self.data}",
                                               f"HOST_DATA_PATH={self.install_dir}/data"))
        package = self.pack(with_base=True)
        with self.assertRaisesRegex(DeployError, "overlaps the installation directory"):
            self.install(package)

    def test_preflight_failure_keeps_the_running_installation(self):
        package = self.pack(with_base=True)
        self.install(package)
        before = (self.install_dir / layout.ENV).read_text()
        target = FakeDocker()
        target.fail_on = "--check-config"
        with self.assertRaisesRegex(DeployError, "simulated failure"), redirect_stdout(io.StringIO()):
            install(Options(package, self.install_dir, interactive=False), target)
        self.assertFalse([c for c in target.calls if c[0] == "compose" and "down" in c])
        self.assertEqual((self.install_dir / layout.ENV).read_text(), before)

    def test_failure_after_stopping_explains_the_state(self):
        package = self.pack(with_base=True)
        self.install(package)
        for failure in ("up -d", KeyboardInterrupt):
            target = FakeDocker()
            errors = io.StringIO()
            if failure == KeyboardInterrupt:
                # Ctrl-C 落在停服之后（查找载荷卷只发生在 down 之后）
                patch = mock.patch.object(target, "volume_exists", side_effect=KeyboardInterrupt)
            else:
                target.fail_on = failure
                patch = contextlib.nullcontext()
            with patch, redirect_stdout(io.StringIO()), redirect_stderr(errors), \
                    self.assertRaises((DeployError, KeyboardInterrupt)):
                install(Options(package, self.install_dir, interactive=False), target)
            self.assertIn("installation incomplete: the previous services are stopped", errors.getvalue())
            self.assertIn("run the installer again", errors.getvalue())

    def test_interrupt_is_reported_by_the_command_line(self):
        errors = io.StringIO()
        with mock.patch.object(cli, "install", side_effect=KeyboardInterrupt), redirect_stderr(errors):
            code = cli.main(["install", "--package", str(self.tmp)])
        self.assertEqual((code, errors.getvalue().strip()), (130, "interrupted"))

    def test_failed_readiness_is_reported(self):
        package = self.pack(with_base=True)
        target = FakeDocker()
        target.ps_rows = [{"Service": "app", "State": "exited", "ExitCode": 1}]
        with self.assertRaisesRegex(DeployError, "service app exited with 1"), redirect_stdout(io.StringIO()):
            install(Options(package, self.install_dir, interactive=False, ready_timeout=2), target)

    def test_readiness_timeout_shows_why_a_service_is_not_ready(self):
        # 只报“未就绪”时现场看不出原因：附上状态与最近的告警，例如相机没有推流。
        package = self.pack(with_base=True)
        target = FakeDocker()
        target.ps_rows = [{"Service": "app", "State": "running", "Health": "starting"}]
        target.logs["app"] = "\n".join(["[info] starting"] + [f"[warning] src: Not Found (404) #{i}"
                                                               for i in range(5)] + ["[info] retrying"])
        with self.assertRaises(DeployError) as caught, redirect_stdout(io.StringIO()):
            install(Options(package, self.install_dir, interactive=False, ready_timeout=0.1), target)
        message = str(caught.exception)
        self.assertIn("services not ready after 0s: app,", message)
        self.assertIn("  app: running, starting\n    [warning] src: Not Found (404) #2\n", message)
        self.assertIn("[warning] src: Not Found (404) #4\n  deepstream-apps: missing", message)
        self.assertNotIn("#1", message)  # 只摘最近的几条
        self.assertNotIn("[info]", message)
        self.assertIn("camera that is not streaming", message)
        self.assertIn("the services are installed and keep running", message)
        # GStreamer 的错误详情换行：括号未闭合的行与下一行合并成一条
        target.logs["app"] = "[warning] src: Not found (gstrtspsrc.c(6736): src:\nNot Found (404))\nResetting source"
        with self.assertRaises(DeployError) as caught, redirect_stdout(io.StringIO()):
            install(Options(package, self.install_dir, interactive=False, ready_timeout=0.1), target)
        self.assertIn("    [warning] src: Not found (gstrtspsrc.c(6736): src: Not Found (404))\n",
                      str(caught.exception))

    def test_ready_service_without_input_is_noted(self):
        # 就绪不要求输入：相机没有推流时安装照常完成，但提示现场还没有收到画面。
        package = self.pack(with_base=True)
        target = FakeDocker()
        target.ready_files["app"] = "count=0 idle_s=12\n"
        output = io.StringIO()
        with redirect_stdout(output):
            install(Options(package, self.install_dir, interactive=False), target)
        self.assertIn("all services are ready\nnote: app has not received any input yet", output.getvalue())
        target.ready_files["app"] = "count=250 idle_s=0\n"
        output = io.StringIO()
        with redirect_stdout(output):
            install(Options(package, self.install_dir, interactive=False), target)
        self.assertNotIn("note:", output.getvalue())

    def test_readiness_wait_reports_progress(self):
        # 相机未推流时等待会持续到超时：期间不报进度会被当成安装卡住。
        package = self.pack(with_base=True)
        target = FakeDocker()
        target.ps_rows = [{"Service": "app", "State": "running", "Health": "starting"}]
        output = io.StringIO()
        with mock.patch.object(ready, "PROGRESS_INTERVAL", 0), redirect_stdout(output), \
                self.assertRaises(DeployError):
            install(Options(package, self.install_dir, interactive=False, ready_timeout=0.1), target)
        self.assertIn("waiting up to 0s for the services to become ready (Ctrl-C stops waiting", output.getvalue())
        self.assertIn("s: waiting for app (running, starting), deepstream-apps (missing)", output.getvalue())

    def test_interrupted_readiness_wait_says_the_services_keep_running(self):
        package = self.pack(with_base=True)
        target = FakeDocker()
        target.ps_rows = [{"Service": "app", "State": "running", "Health": "starting"}]
        errors = io.StringIO()
        with mock.patch.object(ready.time, "sleep", side_effect=KeyboardInterrupt), redirect_stdout(io.StringIO()), \
                redirect_stderr(errors), self.assertRaises(KeyboardInterrupt):
            install(Options(package, self.install_dir, interactive=False), target)
        self.assertIn("stopped waiting: the services are installed and keep running", errors.getvalue())
        self.assertNotIn("installation incomplete", errors.getvalue())


class InventoryTest(ProjectTestCase):
    def test_paths_cannot_escape(self):
        with self.assertRaises(DeployError):
            inventory.safe_path(self.root, "../x")
        (self.root / "link").symlink_to(self.tmp)
        with self.assertRaisesRegex(DeployError, "symlink"):
            inventory.safe_path(self.root, "link/x")


class ProfileChoiceTest(ProjectTestCase):
    """安装时只提供与发布包平台相符的 profile。只读文本，不需要 Docker 与 PyYAML。"""

    def setUp(self):
        super().setUp()
        profiles = self.root / "profiles"
        (profiles / "server.yaml").write_text("# 服务器\nx-platform: amd64\n")
        (profiles / "jetson.yaml").write_text("# Jetson\nx-platform: arm64\n")
        self.release = {"TARGET_PLATFORM": "arm64", "CONFIG_PROFILE": "jetson"}
        self.install_dir = self.tmp / "installed"
        self.install_dir.mkdir()

    def choose(self, profile=None, interactive=False, answer=""):
        options = mock.Mock(profile=profile, clean=False, interactive=interactive)
        output = io.StringIO()
        with redirect_stdout(output), mock.patch("builtins.input", return_value=answer):
            result = preflight._choose_profile(options, self.release, self.install_dir, self.root)
        return result, output.getvalue()

    def installed(self, profile):
        (self.install_dir / layout.ENV).write_text(f"CONFIG_PROFILE='{profile}'\n")

    def test_only_profiles_for_the_package_platform_are_offered(self):
        (_, how), menu = self.choose(interactive=True)
        self.assertEqual(how, "suggested by the package")
        self.assertIn("jetson  Jetson", menu)
        self.assertIn("test  测试方案", menu)  # 没有声明 x-platform：不限平台
        self.assertNotIn("server", menu)
        self.assertEqual(self.choose(interactive=True, answer="1")[0], ("jetson", "chosen here"))

    def test_a_profile_for_another_platform_is_rejected_before_anything_changes(self):
        with self.assertRaisesRegex(DeployError, "profile server targets another platform than this arm64 "
                                                 "package; profiles for arm64: jetson, test"):
            self.choose(profile="server")
        self.assertEqual(self.choose(profile="test")[0], ("test", "from --profile"))

    def test_installed_profile_is_kept_only_when_it_still_fits(self):
        self.installed("test")
        self.assertEqual(self.choose()[0], ("test", "kept from the current installation"))
        for retired in ("server", "dgpu-ds71"):  # 平台不符，或新包已不再提供
            self.installed(retired)
            (profile, how), output = self.choose()
            self.assertEqual((profile, how), ("jetson", "suggested by the package"))
            self.assertIn(f"the installed profile {retired} is not offered by this package for arm64", output)


if __name__ == "__main__":
    unittest.main()
