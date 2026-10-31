#!/usr/bin/env python3
"""训练工具入口：python3 framework/training/dsf-train.py <命令>（通常经 scripts/train.sh 在训练镜像中运行）。"""

import sys
from pathlib import Path

FRAMEWORK = Path(__file__).resolve().parents[1]
# 训练工具复用部署工具的规格校验（dsf_deploy.models）与严格 YAML 读取（dsf.yamlio）。
sys.path[:0] = [str(FRAMEWORK / "training"), str(FRAMEWORK / "deploy"), str(FRAMEWORK / "python")]

from dsf_train.cli import main  # noqa: E402

sys.exit(main())
