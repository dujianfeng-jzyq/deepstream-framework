#!/usr/bin/env bash
# 打包已发布的模型（只更新模型，不换程序）：models/<id>/ 加部署工具，不含镜像。
#   scripts/build-model-package.sh <模型> [<模型>…]
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ $# -ge 1 && "$1" != -* ]] || { sed -n '2,3p' "$0"; exit 2; }
exec python3 "${ROOT}/framework/deploy/dsf-deploy.py" pack-models --root "${ROOT}" "$@"
