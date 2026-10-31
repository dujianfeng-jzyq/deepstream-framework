#!/usr/bin/env bash
# 初始化派生项目：scripts/init-project.sh <name> [--profile <name>]
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 "${ROOT}/framework/deploy/dsf-deploy.py" init --root "${ROOT}" "$@"
