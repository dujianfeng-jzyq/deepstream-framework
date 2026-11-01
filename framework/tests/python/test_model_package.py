"""模型包：打包、停服前构建 engine、切换、只重启引用模型的服务、回滚。用假 Docker 走完整流程
（先用完整发布包安装，再装模型包）；真实的 engine 构建与重启只能在目标机验证。"""

import io
import json
import re
import shutil
import tarfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

from deploy_support import ROOT, docker_cli_available
from dsf_deploy import inventory, layout, models, modelpack
from dsf_deploy.errors import DeployError
from dsf_deploy.install.models import ModelOptions, install_models
from test_pack_install import InstallTestCase

# 挂载 models/ 的常驻服务（模型的使用者）与不挂载的服务。
UNIT_COMPOSE = """services:
  app:
    extends: {file: framework/compose/defaults.yaml, service: dsf-service}
    entrypoint: [bin/app]
    volumes:
      - ./models:${CONTAINER_MODELS_PATH}:ro
  redis:
    image: ${REDIS_IMAGE}
"""


@unittest.skipUnless(docker_cli_available(), "docker CLI not installed (needed for compose config)")
class ModelPackageTest(InstallTestCase):
    def setUp(self):
        super().setUp()
        (self.root / "services/app/compose.yaml").write_text(UNIT_COMPOSE)
        shutil.copy2(ROOT / "scripts/install-models.sh", self.root / "scripts/install-models.sh")
        self.target, _ = self.install(self.pack(with_base=True))
        self.model = self.install_dir / "models/m"

    def republish(self, release, onnx=b"onnx v2", classes=None, parser="/workspace/lib/libparser.so"):
        """模拟 train publish：新产物、新的发布标识与按发布命名的 engine 路径。"""
        directory = self.root / "models/m"
        (directory / "artifacts/model.onnx").write_bytes(onnx)
        record = json.loads((directory / "release.json").read_text())
        record["release"] = release
        if classes:
            record["classes"] = classes
            (directory / "artifacts/labels.txt").write_text("".join(f"{c}\n" for c in classes))
            spec = (directory / "model.yaml").read_text()
            (directory / "model.yaml").write_text(re.sub(r"classes: \[.*\]", f"classes: [{', '.join(classes)}]", spec))
        record["artifacts"] = {name: models.sha256(directory / name) for name in record["artifacts"]}
        (directory / "release.json").write_text(json.dumps(record))
        config = (directory / "nvinfer.txt").read_text().splitlines()
        config = [line for line in config if not line.startswith(("model-engine-file", "custom-lib-path",
                                                                  "num-detected-classes"))]
        config += [f"model-engine-file=/data/engines/m/{release}/model_b1_gpu0_fp16.engine",
                   f"num-detected-classes={len(record['classes'])}"]
        if parser:
            config.append(f"custom-lib-path={parser}")
        (directory / "nvinfer.txt").write_text("\n".join(config) + "\n")

    def pack_models(self):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            archive = modelpack.pack_models(self.root, ["m"])
        target = self.tmp / f"models-{archive.name}"
        with tarfile.open(archive) as tar:
            tar.extractall(target)
        return target / archive.name[: -len(".tar.gz")]

    def install_models(self, package, **options):
        self.target.calls.clear()
        options.setdefault("install_dir", self.install_dir)
        options.setdefault("ready_timeout", 5)
        output, errors = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            install_models(ModelOptions(package, **options), self.target)
        return output.getvalue(), errors.getvalue()

    def release(self, directory):
        return json.loads((directory / "release.json").read_text())["release"]

    def runs(self):
        return [call for call in self.target.calls if call[0] == "run"]

    def snapshot(self):
        return {p.relative_to(self.install_dir).as_posix(): p.read_bytes()
                for p in self.install_dir.rglob("*") if p.is_file()}

    def test_package_contents(self):
        self.republish("r2")
        package = self.pack_models()
        files = json.loads((package / layout.INVENTORY).read_text())
        self.assertIn("models/m/release.json", files)
        self.assertIn("framework/deploy/dsf-deploy.py", files)
        self.assertIn("install.sh", files)
        self.assertNotIn("models/m/x.engine", files)                         # engine 不进包
        self.assertFalse([name for name in files if name.startswith(("images/", "services/", "profiles/"))])
        self.assertIn("MODEL_RELEASES=m=r2", (package / layout.MODEL_PACKAGE_ENV).read_text())
        self.assertIn("-models-m-r2", package.name)

    def test_engines_are_built_before_only_the_model_users_restart(self):
        self.republish("r2")
        package = self.pack_models()
        output, _ = self.install_models(package)
        self.assertEqual(self.release(self.model), "r2")
        self.assertEqual(self.release(self.install_dir / layout.PREVIOUS_MODELS / "m"), "r1")
        self.assertFalse((self.install_dir / layout.STAGED_MODELS).exists())
        calls = self.target.calls
        build = next(i for i, c in enumerate(calls) if c[0] == "run" and "tools/prepare_model_engines.py" in c
                     and "--prune-engines" not in c)
        restart = next(i for i, c in enumerate(calls) if c[0] == "compose" and "restart" in c)
        prune = next(i for i, c in enumerate(calls) if c[0] == "run" and "--prune-engines" in c)
        self.assertLess(build, restart)                                       # 停服前构建
        self.assertLess(restart, prune)
        self.assertEqual(calls[restart][2:], ["restart", "app"])             # 只重启挂载 models/ 的常驻服务
        self.assertFalse([c for c in calls if c[0] == "compose" and ("down" in c or "up" in c)])
        builder = calls[build]
        self.assertIn(f"{self.install_dir / layout.STAGED_MODELS}:/models:ro", builder)  # 候选模型目录
        self.assertIn("nvidia", builder[builder.index("--runtime") + 1])
        self.assertIn(f"demo_app-files-{PAYLOAD}:/workspace:ro", builder)     # 已安装的载荷
        self.assertIn("--network", builder)
        self.assertIn(f"{self.install_dir / layout.PREVIOUS_MODELS}:/models-previous:ro", calls[prune])
        self.assertIn("/models-previous", calls[prune])
        check = next(c for c in calls if c[0] == "run" and "test" in c and "-f" in c)
        self.assertEqual(check[-2:], ["-f", "/workspace/lib/libparser.so"])  # 解析插件在已安装的载荷中
        self.assertLess(calls.index(check), build)
        managed = json.loads((self.install_dir / layout.MANAGED).read_text())
        self.assertEqual(managed["models/m/artifacts/model.onnx"], inventory.sha256(self.model / "artifacts/model.onnx"))
        self.assertIn("模型已更新：m r2", output)
        self.assertIn(f"在 {package} 中运行 ./install.sh --rollback", output)

        # 之后的完整安装把装入的模型视为发布文件，不当作现场修改。
        self.target, _ = self.install(self.pack(with_base=True), self.target, images=self.target.images)
        self.assertEqual(self.release(self.model), "r2")

        # 同一个包再装一次：什么都不做。
        output, _ = self.install_models(package)
        self.assertIn("m 已是 r2，跳过", output)
        self.assertEqual(self.runs(), [])

    def test_rollback_restores_the_previous_model(self):
        self.republish("r2")
        package = self.pack_models()
        self.install_models(package)
        output, _ = self.install_models(package, rollback=True)
        self.assertEqual(self.release(self.model), "r1")
        self.assertEqual(self.release(self.install_dir / layout.PREVIOUS_MODELS / "m"), "r2")
        self.assertTrue(any("restart" in c for c in self.target.calls if c[0] == "compose"))
        self.assertIn("模型已回滚：m r1", output)
        with self.assertRaisesRegex(DeployError, "回滚只撤销本包的安装"):
            self.install_models(package, rollback=True)

    def test_incompatible_or_unsupported_changes_change_nothing(self):
        before = self.snapshot()
        self.republish("r2", classes=["car", "person", "bike"])
        with self.assertRaisesRegex(DeployError, "类别与已安装的不同"):
            self.install_models(self.pack_models())
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.runs(), [])

        self.republish("r3", classes=["car", "person"])
        package = self.pack_models()
        self.target.fail_on = "-f /workspace/lib/libparser.so"
        with self.assertRaisesRegex(DeployError, "不在已安装的程序中"):
            self.install_models(package)
        self.target.fail_on = None
        self.assertEqual(self.snapshot(), before)
        self.assertFalse([c for c in self.runs() if "tools/prepare_model_engines.py" in c])

        self.target.fail_on = "tools/prepare_model_engines.py"                # engine 构建失败
        with self.assertRaises(DeployError):
            self.install_models(package)
        self.target.fail_on = None
        self.assertEqual(self.snapshot(), before)
        self.assertFalse([c for c in self.target.calls if c[0] == "compose" and "restart" in c])

        output, _ = self.install_models(package, dry_run=True)
        self.assertIn("更新模型：m r1 -> r3", output)
        self.assertIn("重启：app", output)
        self.assertEqual(self.snapshot(), before)

    def test_a_failed_restart_explains_how_to_finish(self):
        self.republish("r2")
        package = self.pack_models()
        self.target.fail_on = "restart"
        errors = io.StringIO()
        with redirect_stdout(io.StringIO()), redirect_stderr(errors), self.assertRaises(DeployError):
            install_models(ModelOptions(package, install_dir=self.install_dir), self.target)
        # 切换已完成：再次运行会跳过，所以提示直接给出重启命令。
        self.assertEqual(self.release(self.model), "r2")
        self.assertIn(f"cd {self.install_dir} && docker compose restart app", errors.getvalue())

    def test_the_package_must_match_an_installed_project_and_model(self):
        self.republish("r2")
        package = self.pack_models()
        with self.assertRaisesRegex(DeployError, "没有已安装的 demo"):
            self.install_models(package, install_dir=self.tmp / "elsewhere")
        shutil.rmtree(self.model)
        with self.assertRaisesRegex(DeployError, "已安装的项目没有模型 m"):
            self.install_models(package)
        (package / "models/m/notes.md").write_text("tampered")
        with self.assertRaisesRegex(DeployError, "发布包文件已损坏"):
            self.install_models(package)

    def test_unpublished_models_cannot_be_packed(self):
        (self.root / "models/m/release.json").unlink()
        with self.assertRaisesRegex(DeployError, "has not been published"):
            modelpack.pack_models(self.root, ["m"])
        with self.assertRaisesRegex(DeployError, "is not a model directory"):
            modelpack.pack_models(self.root, ["nope"])


PAYLOAD = "0123456789ab"


if __name__ == "__main__":
    unittest.main()
