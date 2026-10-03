#!/usr/bin/env bash
# 渲染配置：.env 与 generated/。只生成文件，不启停容器。
#   scripts/apply-profile.sh [profile]   省略 profile 时沿用 .env 中记录的选择
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
args=(render --root "${ROOT}")
if [[ $# -gt 0 && "$1" != -* ]]; then
  args+=(--profile "$1")
  shift
fi
exec python3 "${ROOT}/framework/deploy/dsf-deploy.py" "${args[@]}" "$@"
