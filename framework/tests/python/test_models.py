"""模型契约校验：规格、发布事实、产物与 nvinfer 配置的一致性。"""

import io
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from deploy_support import ROOT, FakeDocker, ProjectTestCase, docker_cli_available, write_model
from dsf_deploy import cli, models, pack
from dsf_deploy.errors import DeployError


class ModelCheckTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="dsf-models-"))
        self.model = self.tmp / "models/m"
        self.release = write_model(self.model)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def problems(self):
        return models.check_model(self.model)

    def edit(self, name, old, new):
        path = self.model / name
        text = path.read_text()
        self.assertIn(old, text)
        path.write_text(text.replace(old, new))

    def write_release(self, **changes):
        (self.model / "release.json").write_text(json.dumps({**self.release, **changes}))

    def test_a_published_model_passes(self):
        self.assertEqual(self.problems(), [])
        models.check(self.tmp)

    def test_spec_must_be_complete(self):
        self.edit("model.yaml", "id: m", "id: other")
        self.edit("model.yaml", "classes: [car, person]", "classes: [car, car]")
        self.assertEqual(self.problems(), ["model.yaml: id must equal the directory name 'm'",
                                           "model.yaml: class names must be unique"])
        (self.model / "model.yaml").unlink()
        self.assertEqual(self.problems(), ["missing model.yaml"])

    def test_spec_changes_after_publishing_need_a_new_release(self):
        # 类别号与输入尺寸是已部署业务的契约：改了规格而产物没有重新发布时拒绝。
        self.edit("model.yaml", "classes: [car, person]", "classes: [person, car]")
        self.edit("model.yaml", "width: 64", "width: 128")
        self.assertEqual(self.problems()[:2], [
            f"{key} in model.yaml differs from the published release; retrain and publish, "
            "or restore the published value" for key in ("classes", "input")])

    def test_artifacts_must_match_the_release_record(self):
        (self.model / "artifacts/model.onnx").write_bytes(b"retrained without publishing")
        (self.model / "artifacts/old.onnx").write_bytes(b"stale")
        self.assertEqual(self.problems(), ["artifacts/model.onnx does not match its sha256 in release.json",
                                           "artifacts/old.onnx is not a published artifact"])
        (self.model / "artifacts/model.onnx").unlink()
        self.assertIn("missing artifact artifacts/model.onnx", self.problems())

    def test_artifacts_cannot_point_outside_the_model(self):
        artifacts = dict(self.release["artifacts"], **{"../escape.onnx": "0" * 64})
        self.write_release(artifacts=artifacts)
        self.assertIn("release.json: artifact ../escape.onnx must be inside artifacts/", self.problems())

    def test_labels_follow_the_classes(self):
        (self.model / "artifacts/labels.txt").write_text("person\ncar\n")
        release = dict(self.release)
        release["artifacts"] = dict(release["artifacts"],
                                    **{"artifacts/labels.txt": models.sha256(self.model / "artifacts/labels.txt")})
        self.write_release(**release)
        self.assertEqual(self.problems(), ["artifacts/labels.txt must list the classes of model.yaml, "
                                           "one per line, in order"])

    def test_exported_signature_matches_the_contract(self):
        onnx = dict(self.release["onnx"], inputs=[{"name": "input", "shape": [1, 3, 64, 64]}],
                    outputs=[{"name": "output", "shape": [1, 100, 85]}])
        self.write_release(onnx=onnx)
        self.assertEqual(self.problems(), [
            "exported input shape [1, 3, 64, 64] does not match input {'width': 64, 'height': 32} (NCHW, 3 channels)",
            "xyxy-confidence-class_id needs one output whose last dimension is 6"])
        # 未知布局：由提供它的解析插件负责，框架不检查输出形状。
        self.edit("model.yaml", "output_layout: xyxy-confidence-class_id", "output_layout: custom")
        self.write_release(onnx=dict(onnx, inputs=self.release["onnx"]["inputs"]), output_layout="custom")
        self.assertEqual(self.problems(), [])

    def test_nvinfer_keys_follow_the_release(self):
        self.edit("nvinfer.txt", "onnx-file=artifacts/model.onnx", "onnx-file=../other.onnx")
        self.edit("nvinfer.txt", "num-detected-classes=2", "num-detected-classes=80")
        self.edit("nvinfer.txt", "batch-size=1", "batch-size=4")
        self.assertEqual(self.problems(), [
            "nvinfer.txt: onnx-file=../other.onnx is not a published artifact",
            "nvinfer.txt: num-detected-classes=80 but model.yaml has 2 classes",
            "nvinfer.txt: batch-size=4 but the ONNX was exported with a fixed batch of 1"])
        self.write_release(onnx=dict(self.release["onnx"], dynamic=True, inputs=[{"shape": [-1, 3, 32, 64]}]))
        self.assertNotIn("nvinfer.txt: batch-size=4 but the ONNX was exported with a fixed batch of 1",
                         self.problems())  # 动态 batch：形状由 .trtexec 给出

    def test_engines_are_written_under_the_data_directory(self):
        # 模型目录对所有容器只读；engine 是目标机的派生物，放在跨发布保留的 /data/engines/<id>/。
        for engine in ("model_b1_gpu0_fp16.engine", "/data/engines/other/x.engine", "/data/engines/m/../x.engine"):
            with self.subTest(engine=engine):
                self.edit("nvinfer.txt", "model-engine-file=/data/engines/m/model_b1_gpu0_fp16.engine",
                          f"model-engine-file={engine}")
                self.assertEqual(self.problems(), [
                    "nvinfer.txt: model-engine-file must be a file under /data/engines/m/ "
                    f"(the models directory is read-only), got {engine}"])
                self.edit("nvinfer.txt", f"model-engine-file={engine}",
                          "model-engine-file=/data/engines/m/model_b1_gpu0_fp16.engine")
        self.edit("nvinfer.txt", "model-engine-file=/data/engines/m/model_b1_gpu0_fp16.engine\n", "")
        self.assertIn("nvinfer.txt: model-engine-file must be a file under /data/engines/m/ "
                      "(the models directory is read-only), got nothing", self.problems())

    def test_unpublished_models_cannot_be_configured(self):
        (self.model / "release.json").unlink()
        shutil.rmtree(self.model / "artifacts")
        self.assertEqual(self.problems(), ["nvinfer.txt configures a model that has not been published "
                                           "(release.json is missing)"])
        (self.model / "nvinfer.txt").unlink()
        self.assertEqual(self.problems(), [])  # 只有规格：训练中的模型

    def test_problems_of_all_models_are_reported_together(self):
        write_model(self.tmp / "models/n")
        (self.model / "artifacts/model.onnx").write_bytes(b"x")
        (self.tmp / "models/n/artifacts/labels.txt").write_text("x\n")
        with self.assertRaisesRegex(DeployError, "model check failed:\n"
                                                 "  models/m: artifacts/model.onnx does not match.*\n"
                                                 "  models/n: artifacts/labels.txt does not match"):
            models.check(self.tmp)

    def test_repository_models_pass(self):
        # 构建镜像的上下文不含 models/，这里在开发机上覆盖仓库中的真实模型（打包时同样会检查）。
        models.check(ROOT)


class ServiceLockTest(unittest.TestCase):
    """服务 → 模型的类别依赖：services/<服务>/models.lock.json（docs/training.md §15.1）。"""
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="dsf-models-"))
        write_model(self.tmp / "models/m")
        (self.tmp / "services/s").mkdir(parents=True)
        (self.tmp / "services/s/compose.yaml").write_text("services: {}\n")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def deploy(self, *argv):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = cli.main([*argv, "--root", str(self.tmp)])
        return code, stdout.getvalue(), stderr.getvalue()

    def test_services_confirm_the_published_classes(self):
        models.check(self.tmp)  # 没有锁记录的服务不检查
        code, _, err = self.deploy("lock-models", "s")
        self.assertIn("services/s/models.lock.json does not exist yet: name the models s uses", err)
        code, out, err = self.deploy("lock-models", "s", "m")
        self.assertEqual(code, 0, err)
        self.assertIn("  m: recorded (2 classes)", out)
        lock = json.loads((self.tmp / "services/s/models.lock.json").read_text())
        self.assertEqual(lock, {"schema": 1, "models": {"m": ["car", "person"]}})
        models.check(self.tmp)
        shutil.rmtree(self.tmp / "models/m")
        write_model(self.tmp / "models/m", classes=("car", "bus", "person"))  # 重新发布，类别变了
        with self.assertRaisesRegex(DeployError, "services/s/models.lock.json: the published classes of m changed "
                                                 "since s confirmed them:\n    added bus \\(class 1\\)\n"
                                                 "    renumbered person 1->2(.|\n)*lock-models s"):
            models.check(self.tmp)
        code, out, err = self.deploy("lock-models", "s")
        self.assertEqual(code, 0, err)
        self.assertIn("  m:\n    added bus (class 1)", out)
        models.check(self.tmp)

    def test_models_must_be_published_and_services_must_exist(self):
        write_model(self.tmp / "models/n")
        (self.tmp / "models/n/release.json").unlink()
        shutil.rmtree(self.tmp / "models/n/artifacts")
        (self.tmp / "models/n/nvinfer.txt").unlink()
        code, _, err = self.deploy("lock-models", "s", "n")
        self.assertIn("model n has not been published", err)
        code, _, err = self.deploy("lock-models", "nope", "m")
        self.assertIn("'nope' is not a service", err)
        (self.tmp / "services/s/models.lock.json").write_text('{"schema": 1, "models": {"n": ["x"]}}')
        with self.assertRaisesRegex(DeployError, "services/s/models.lock.json: model n has not been published"):
            models.check(self.tmp)


@unittest.skipUnless(docker_cli_available(), "docker CLI not installed (needed for compose config)")
class PackModelCheckTest(ProjectTestCase):
    def test_pack_refuses_inconsistent_models(self):
        (self.root / "models/m/artifacts/model.onnx").write_bytes(b"changed")
        docker = FakeDocker()
        with self.assertRaisesRegex(DeployError, "models/m: artifacts/model.onnx does not match"), \
                redirect_stdout(io.StringIO()):
            pack.pack(self.root, "test", False, docker)
        self.assertEqual(docker.calls, [])  # 在检查镜像与保存镜像之前


if __name__ == "__main__":
    unittest.main()
