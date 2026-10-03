"""dsf Python 运行时：配置读取规则与服务入口约定。"""

import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from dsf.config import ConfigError, Reader, at_least, one_of
from dsf.service import Service


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


if __name__ == "__main__":
    unittest.main()
