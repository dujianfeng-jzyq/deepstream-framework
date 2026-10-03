"""__NAME__：模板配置通过 --check-config。"""

import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

UNIT = Path(__file__).resolve().parents[1]


class CheckConfigTest(unittest.TestCase):
    def test_template_is_valid(self):
        text = re.sub(r"\$\{[A-Z_]+\}", "x", (UNIT / "templates/config/__NAME__.yaml").read_text())
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "__NAME__.yaml").write_text(text)
            result = subprocess.run([sys.executable, str(UNIT / "main.py"), "--check-config"],
                                    env=dict(os.environ, CONTAINER_CONFIG_PATH=directory),
                                    capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
