"""dsf Python 运行时：配置读取规则与服务入口约定。"""

import io
import logging
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from dsf.config import ConfigError, Reader, at_least, one_of
from dsf.service import FrameworkOptions, Service

SECTIONS = Path(__file__).resolve().parents[1] / "framework_sections.txt"


def reader(text: str) -> Reader:
    from dsf import yamlio
    return Reader.from_document(yamlio.load(text), "test.yaml")


class ConfigTest(unittest.TestCase):
    def test_types_defaults_and_constraints(self):
        root = reader("a:\n  n: 3\n  f: 2\n  b: true\n  s: x\n  empty:\n")
        a = root.section("a")
        self.assertEqual(a.get("n", 0), 3)
        self.assertEqual(a.get("f", 1.0), 2.0)
        self.assertIs(a.get("b", False), True)
        self.assertEqual(a.get("empty", "default"), "default")
        with self.assertRaisesRegex(ConfigError, "a.s: expected int"):
            a.get("s", 0)
        with self.assertRaisesRegex(ConfigError, "a.b: expected int"):
            a.get("b", 0)  # bool 不是整数
        with self.assertRaisesRegex(ConfigError, "a.n: must be at least 5"):
            a.get("n", 0, at_least(5))
        with self.assertRaisesRegex(ConfigError, "a.s: 'x' is not one of: y"):
            a.get("s", "", one_of("y"))
        with self.assertRaisesRegex(ConfigError, "a.missing: is required"):
            a.required("missing")

    def test_unknown_keys_and_lists(self):
        root = reader("items:\n  - id: 1\n  - id: 2\n    typo: 1\n")
        ids = [item.required("id", int) for item in root.items("items", 1)]
        self.assertEqual(ids, [1, 2])
        with self.assertRaisesRegex(ConfigError, r"items\[1\].typo: unknown field"):
            root.finish()

    def test_plain_scalars_follow_the_requested_type_like_cpp(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "c.yaml")
            path.write_text('password: 123456\nmode: on\nflag: Yes\nn: -3\nf: 1.5e3\n'
                            'quoted: "7"\nbad_int: 1_000\nbad_float: .inf\nnone: ~\n')
            root = Reader.load(str(path))
            self.assertEqual(root.secret("password"), "123456")
            self.assertIs(type(root.get("mode", "")), str)
            self.assertEqual(root.get("mode", ""), "on")
            self.assertIs(root.get("flag", False), True)
            self.assertEqual(root.get("n", 0), -3)
            self.assertEqual(root.get("f", 0.0), 1500.0)
            self.assertEqual(root.get("quoted", ""), "7")
            self.assertEqual(root.get("none", "default"), "default")
            with self.assertRaisesRegex(ConfigError, "quoted: expected int, got '7'"):
                root.get("quoted", 0)  # 加引号的标量只能读成字符串
            with self.assertRaisesRegex(ConfigError, "bad_int: expected int"):
                root.get("bad_int", 0)
            with self.assertRaisesRegex(ConfigError, "bad_float: expected float"):
                root.get("bad_float", 0.0)
            with self.assertRaisesRegex(ConfigError, "mode: expected int"):
                root.get("mode", 0)

    def test_warnings_are_collected_with_full_path(self):
        root = reader("a:\n  n: 3\n")
        a = root.section("a")
        a.get("n", 0)
        self.assertEqual(root.warnings(), [])
        a.warn("n", "is unusual")
        root.warn("", "check the outputs")
        root.finish()  # 警告不影响校验
        self.assertEqual(root.warnings(), ["test.yaml: a.n: is unusual", "test.yaml: (top level): check the outputs"])

    def test_duplicate_keys_and_secrets(self):
        with self.assertRaisesRegex(Exception, "duplicate key"):
            reader("a: 1\na: 2\n")
        root = reader("password: s3cret\n")
        root.secret("password")
        self.assertEqual(root.describe(), "password=***")


class ServiceTest(unittest.TestCase):
    def test_check_config_and_run(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory, "svc.yaml")
            config.write_text("logging:\n  level: warn\nvalue: 5\n")
            ready = Path(directory, "ready")
            os.environ["DSF_READY_FILE"] = str(ready)
            seen = []

            def main(value, service):
                seen.append(value)
                service.sleep(0)
                seen.append(ready.exists())
                service.stop()
                self.assertFalse(service.sleep(5))  # 停止后立即返回

            load = lambda root: root.get("value", 0)
            with redirect_stdout(io.StringIO()) as output:
                code = Service("svc").run(load, main, ["--config", str(config), "--check-config"])
            self.assertEqual((code, output.getvalue().strip()), (0, "configuration is valid"))
            self.assertEqual(seen, [])
            self.assertEqual(Service("svc").run(load, main, ["--config", str(config)]), 0)
            self.assertEqual(seen, [5, True])
            self.assertFalse(ready.exists())
            config.write_text("value: 5\nbad: 1\n")
            with redirect_stderr(io.StringIO()) as errors:
                self.assertEqual(Service("svc").run(load, main, ["--config", str(config)]), 1)
            self.assertIn("bad: unknown field", errors.getvalue())
            with redirect_stderr(io.StringIO()):
                self.assertEqual(Service("svc").run(load, main, ["--bogus"]), 2)
            del os.environ["DSF_READY_FILE"]

    def test_framework_sections_match_the_shared_table(self):
        expected = []
        for line in SECTIONS.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                path, *runtimes = line.split()
                if "python" in runtimes:
                    expected.append(path)
        root = reader("{}")
        FrameworkOptions.read(root)
        self.assertEqual(root.declared_paths(), sorted(expected))

    def test_file_logging_follows_the_cpp_rules(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory, "svc.yaml")
            logs = Path(directory, "logs")
            config.write_text("logging:\n  file: true\n")
            with redirect_stderr(io.StringIO()) as errors:
                code = Service("svc").run(lambda root: None, lambda *_: None, ["--config", str(config)])
            self.assertEqual(code, 1)
            self.assertIn("logging.directory: is required when file logging is enabled", errors.getvalue())

            config.write_text(f"logging:\n  console: false\n  file: true\n  directory: {logs}\n")
            ready = Path(directory, "ready")
            os.environ["DSF_READY_FILE"] = str(ready)
            try:
                with redirect_stdout(io.StringIO()) as output:
                    code = Service("svc").run(lambda root: None, lambda _config, service: service.log.info("hello"),
                                              ["--config", str(config)])
            finally:
                del os.environ["DSF_READY_FILE"]
                logging.shutdown()
            self.assertEqual(code, 0)
            self.assertEqual(output.getvalue(), "")  # console: false 时只写文件
            self.assertIn("[INFO] [svc] hello", Path(logs, "svc.log").read_text(encoding="utf-8"))

    def test_check_config_prints_config_warnings(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory, "svc.yaml")
            config.write_text("value: 5\n")

            def load(root):
                value = root.get("value", 0)
                root.warn("value", "is unusual")
                return value

            with redirect_stdout(io.StringIO()) as output, redirect_stderr(io.StringIO()) as errors:
                code = Service("svc").run(load, lambda *_: None, ["--config", str(config), "--check-config"])
            self.assertEqual((code, output.getvalue().strip()), (0, "configuration is valid"))
            self.assertIn(f"svc: warning: {config}: value: is unusual", errors.getvalue())

    def test_sleep_refreshes_the_ready_file_during_long_waits(self):
        with tempfile.TemporaryDirectory() as directory:
            ready = Path(directory, "ready")
            os.environ["DSF_READY_FILE"] = str(ready)
            try:
                service = Service("svc")
            finally:
                del os.environ["DSF_READY_FILE"]
            service._alive_interval = 0.05
            touches = []
            original = service.alive
            service.alive = lambda: (touches.append(time.monotonic()), original())
            self.assertTrue(service.sleep(0.3))
            self.assertGreaterEqual(len(touches), 5)
            self.assertTrue(ready.exists())
            self.assertLess(max(b - a for a, b in zip(touches, touches[1:])), 0.2)
            service.stop()
            self.assertFalse(service.sleep(10))  # 停止后立即返回，不等到时限


def run_service(directory: str, config: str, body: str, timeout: float = 30) -> subprocess.CompletedProcess:
    """在子进程中运行服务：看门狗以 os._exit 结束进程，信号处理会替换进程级处理器。"""
    Path(directory, "svc.yaml").write_text(config)
    script = Path(directory, "svc.py")
    script.write_text(textwrap.dedent("""
        import os, signal, sys, time
        from dsf.service import Service
        def main(_, service):
        """) + textwrap.indent(textwrap.dedent(body), "    ") +
        '\nsys.exit(Service("svc").run(lambda root: None, main, ["--config", sys.argv[1]]))\n')
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path),
                       DSF_READY_FILE=str(Path(directory, "ready")))
    return subprocess.run([sys.executable, str(script), str(Path(directory, "svc.yaml"))], env=environment,
                          capture_output=True, text=True, timeout=timeout)


class ServiceProcessTest(unittest.TestCase):
    def test_signal_stops_the_main_loop_and_is_logged_by_the_main_thread(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_service(directory, "", """
                os.kill(os.getpid(), signal.SIGTERM)
                while service.sleep(5):
                    pass
                """)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn(f"stopped by signal {int(signal.SIGTERM)}", result.stdout)

    def test_watchdog_exits_when_the_main_loop_stalls(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_service(directory, "watchdog:\n  stall_timeout_s: 1\n", """
                service.sleep(0)
                time.sleep(20)  # 卡住：不再调用 sleep/alive
                """)
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn("made no progress", result.stdout)
            self.assertFalse(Path(directory, "ready").exists())

    def test_long_sleeps_count_as_progress(self):
        # 时限 1 秒，单次等待 2.5 秒：等待期间分段更新，看门狗不触发。
        with tempfile.TemporaryDirectory() as directory:
            result = run_service(directory, "watchdog:\n  stall_timeout_s: 1\n", """
                assert service.sleep(2.5)
                """)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertNotIn("made no progress", result.stdout)

    def test_watchdog_stays_quiet_while_the_loop_advances(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_service(directory, "watchdog:\n  stall_timeout_s: 1\n", """
                for _ in range(10):
                    service.sleep(0.3)
                """)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
