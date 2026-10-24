"""CTest 的 Python 测试入口：运行一个 unittest 脚本。

DSF_FORBID_SKIPS=1 时（builder 中）SKIP 计为失败：builder 带齐全部测试依赖，
出现 SKIP 说明覆盖缺失，而 CTest 只看退出码，看不到脚本内部的 SKIP。
"""

import importlib.util
import os
import sys
import unittest
from pathlib import Path


def main(argv: list[str]) -> int:
    script = Path(argv[1]).resolve()
    sys.argv = [str(script), *argv[2:]]
    sys.path.insert(0, str(script.parent))
    spec = importlib.util.spec_from_file_location(script.stem, script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[script.stem] = module
    spec.loader.exec_module(module)
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        return 1
    if result.skipped and os.environ.get("DSF_FORBID_SKIPS") == "1":
        for test, reason in result.skipped:
            print(f"[SKIP] {test.id()}: {reason}", file=sys.stderr)
        print("DSF_FORBID_SKIPS=1: skipped tests count as failures", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
