#!/usr/bin/env python3
"""部署工具入口。只依赖标准库即可启动；渲染时才需要 PyYAML。"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "python")]

from dsf_deploy.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
