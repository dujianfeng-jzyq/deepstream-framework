#!/usr/bin/env bash
# 生成服务单元骨架：scripts/new-app.sh <name> [--kind cpp|python]
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 "${ROOT}/framework/deploy/dsf-deploy.py" new-app --root "${ROOT}" "$@"
