"""prepare_model_engines.py：用假 trtexec 验证扫描、指纹、复用、失败保留与并发锁。"""

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TOOL = ROOT / "framework" / "tools" / "prepare_model_engines.py"
FAKE = Path(__file__).resolve().parent / "fake_trtexec.py"

CONFIG = """[property]
gpu-id=0
onnx-file=../onnx/model.onnx
model-engine-file={engine}
network-mode=2
batch-size=1
[class-attrs-all]
pre-cluster-threshold=0.25
"""


class PrepareEnginesTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="engines-"))
        self.models = self.dir / "models"
        (self.models / "m" / "onnx").mkdir(parents=True)
        (self.models / "m" / "deployment").mkdir()
        (self.models / "m" / "onnx" / "model.onnx").write_bytes(b"onnx-v1")
        self.config = self.models / "m" / "deployment" / "config.txt"
        self.engine = self.dir / "data" / "engines" / "m.engine"
        self.write_config(str(self.engine))
        (self.models / "m" / "deployment" / "notes.txt").write_text("not a config\n")
        self.calls = self.dir / "calls.log"
        self.env = dict(os.environ, FAKE_TRTEXEC_LOG=str(self.calls))

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def write_config(self, engine: str) -> None:
        self.config.write_text(CONFIG.format(engine=engine))

    def run_tool(self, *args, **env):
        return subprocess.run([sys.executable, str(TOOL), "--trtexec", str(FAKE), "--compute-cap", "8.9",
                               *args], env={**self.env, **env}, capture_output=True, text=True, timeout=60)

    def builds(self) -> int:
        return len(self.calls.read_text().splitlines()) if self.calls.exists() else 0

    def test_first_run_builds_then_reuses(self):
        result = self.run_tool("--models-path", str(self.models))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.builds(), 1)
        self.assertIn("--fp16", self.calls.read_text())
        fingerprint = self.engine.with_name("m.engine.fingerprint").read_text()
        for line in ("tensorrt=trt100900", "compute_cap=8.9", "nvinfer.network-mode=2",
                     "nvinfer.batch-size=1", "engine_sha256="):
            self.assertIn(line, fingerprint)
        self.assertEqual(self.run_tool("--models-path", str(self.models)).returncode, 0)
        self.assertEqual(self.builds(), 1)

    def test_inputs_that_change_the_engine_trigger_rebuilds(self):
        self.run_tool(str(self.config))
        (self.models / "m" / "onnx" / "model.onnx").write_bytes(b"onnx-v2")
        self.run_tool(str(self.config))
        self.assertEqual(self.builds(), 2)
        self.run_tool(str(self.config), FAKE_TRT_VERSION="100800")
        self.assertEqual(self.builds(), 3)
        self.run_tool(str(self.config), "--trtexec-arg=--workspace=64")
        self.assertEqual(self.builds(), 4)
        self.config.write_text(self.config.read_text().replace("batch-size=1", "batch-size=2"))
        self.run_tool(str(self.config), "--trtexec-arg=--workspace=64")
        self.assertEqual(self.builds(), 5)
        # 只换构建用的卡号不重建
        self.run_tool(str(self.config), "--trtexec-arg=--workspace=64", "--trtexec-arg=--device=1")
        self.assertEqual(self.builds(), 5)

    def test_sidecar_arguments_are_used_and_fingerprinted(self):
        self.run_tool(str(self.config))
        self.config.with_suffix(".trtexec").write_text("# 动态形状\n--minShapes=input:1x3x640x640\n")
        self.run_tool(str(self.config))
        self.assertEqual(self.builds(), 2)
        self.assertIn("--minShapes=input:1x3x640x640", self.calls.read_text().splitlines()[-1])

    def test_engine_and_fingerprint_must_match(self):
        self.run_tool(str(self.config))
        self.engine.write_text("tampered")  # 指纹记录的哈希不再对应
        self.run_tool(str(self.config))
        self.assertEqual(self.builds(), 2)
        self.engine.with_name("m.engine.fingerprint").unlink()  # 没有指纹的 engine 不可信
        self.run_tool(str(self.config))
        self.assertEqual(self.builds(), 3)

    def test_failed_or_empty_build_keeps_previous_engine(self):
        self.run_tool(str(self.config))
        previous = self.engine.read_text()
        failed = self.run_tool(str(self.config), "--force", FAKE_TRTEXEC_FAIL="1")
        self.assertEqual(failed.returncode, 1)
        self.assertEqual(self.engine.read_text(), previous)
        result = self.run_tool(str(self.config), "--force", FAKE_TRTEXEC_EMPTY="1")
        self.assertEqual(result.returncode, 1)
        self.assertIn("empty engine", result.stderr)
        self.assertEqual(self.engine.read_text(), previous)
        self.assertEqual(sorted(p.name for p in self.engine.parent.iterdir()),
                         ["m.engine", "m.engine.fingerprint", "m.engine.lock"])

    def test_relative_engine_path_is_resolved_against_the_config(self):
        self.write_config("../engines/rel.engine")
        self.run_tool("--models-path", str(self.models))
        self.assertTrue((self.models / "m" / "engines" / "rel.engine").is_file())

    def test_other_sections_cannot_supply_engine_properties(self):
        self.config.write_text("[property]\nonnx-file=../onnx/model.onnx\n[other]\nmodel-engine-file=/x\n")
        result = self.run_tool(str(self.config))
        self.assertEqual(result.returncode, 1)
        self.assertIn("needs onnx-file and model-engine-file", result.stderr)
        self.assertEqual(self.run_tool("--models-path", str(self.models)).returncode, 0)  # 扫描时跳过
        self.assertEqual(self.builds(), 0)

    def test_errors_are_reported(self):
        (self.models / "m" / "onnx" / "model.onnx").unlink()
        self.assertIn("ONNX model not found", self.run_tool(str(self.config)).stderr)
        self.assertIn("models path not found", self.run_tool("--models-path", str(self.dir / "nope")).stderr)
        self.assertIn("config list not found", self.run_tool("--config-list", str(self.dir / "nope")).stderr)

    def test_config_list_selects_models(self):
        listing = self.dir / "list.txt"
        listing.write_text("# 使用中的模型\nmodels/m/deployment/config.txt\n")
        self.assertEqual(self.run_tool("--config-list", str(listing)).returncode, 0)
        self.assertEqual(self.builds(), 1)

    def test_unwritable_location_fails_cleanly(self):
        if os.geteuid() == 0:
            self.skipTest("root ignores directory permissions")
        locked = self.dir / "locked"
        locked.mkdir()
        locked.chmod(0o500)
        self.write_config(str(locked / "m.engine"))
        try:
            result = self.run_tool(str(self.config))
        finally:
            locked.chmod(0o700)
        self.assertEqual(result.returncode, 1)
        self.assertIn("cannot write next to", result.stderr)

    def test_concurrent_builders_build_once(self):
        results = []
        threads = [threading.Thread(target=lambda: results.append(self.run_tool(str(self.config))))
                   for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertTrue(all(r.returncode == 0 for r in results))
        self.assertEqual(self.builds(), 1)

    def test_no_configs_is_not_an_error(self):
        empty = self.dir / "empty"
        empty.mkdir()
        result = self.run_tool("--models-path", str(empty))
        self.assertEqual(result.returncode, 0)
        self.assertIn("nothing to do", result.stderr)


def load_tool():
    spec = importlib.util.spec_from_file_location("prepare_model_engines", TOOL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclass 在定义时按模块名查找命名空间
    spec.loader.exec_module(module)
    return module


class FakeCuda:
    """libcuda 的替身：按属性号返回计算能力的主次版本。"""

    def __init__(self, major: int, minor: int, init_result: int = 0):
        self.values = {75: major, 76: minor}
        self.init_result = init_result

    def cuInit(self, _flags):
        return self.init_result

    def cuDeviceGet(self, handle, ordinal):
        handle._obj.value = ordinal
        return 0

    def cuDeviceGetAttribute(self, value, attribute, _handle):
        value._obj.value = self.values[attribute]
        return 0


class ComputeCapabilityTest(unittest.TestCase):
    def setUp(self):
        self.tool = load_tool()

    def smi(self, stdout: str, returncode: int = 0):
        return mock.patch.object(self.tool.subprocess, "run", return_value=subprocess.CompletedProcess(
            [], returncode, stdout=stdout, stderr=""))

    def test_nvidia_smi_answer_is_used_first(self):
        with self.smi("8.9\n"), mock.patch.object(self.tool.ctypes, "CDLL", side_effect=AssertionError):
            self.assertEqual(self.tool.compute_capability(0), "8.9")

    def test_falls_back_to_libcuda_when_the_query_is_unsupported(self):
        # 不支持查询的平台可能成功退出却输出占位符；它不能进入指纹。
        for stdout, code in (("[N/A]\n", 0), ("", 6)):
            with self.subTest(stdout=stdout), self.smi(stdout, code), \
                    mock.patch.object(self.tool.ctypes, "CDLL", return_value=FakeCuda(8, 7)):
                self.assertEqual(self.tool.compute_capability(0), "8.7")
        with mock.patch.object(self.tool.subprocess, "run", side_effect=FileNotFoundError), \
                mock.patch.object(self.tool.ctypes, "CDLL", return_value=FakeCuda(8, 7)):
            self.assertEqual(self.tool.compute_capability(0), "8.7")

    def test_fails_when_neither_source_answers(self):
        for cuda in (OSError("libcuda.so.1: cannot open"), FakeCuda(8, 7, init_result=100)):
            with self.subTest(cuda=cuda), self.smi("[N/A]\n"), \
                    mock.patch.object(self.tool.ctypes, "CDLL",
                                      side_effect=cuda if isinstance(cuda, Exception) else None,
                                      return_value=cuda):
                with self.assertRaisesRegex(self.tool.Failure, "nvidia-smi or libcuda"):
                    self.tool.compute_capability(0)


if __name__ == "__main__":
    unittest.main()
