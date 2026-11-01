#!/usr/bin/env bash
# 模型包安装入口（位于模型包根目录）：在已安装的项目中更新模型。只需要 Python 3.9+ 标准库与 Docker Compose 2.22+。
#   ./install.sh [--rollback] [--dry-run] [--install-dir <dir>] [--ready-timeout <s>]
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
command -v python3 >/dev/null || { echo "install.sh: python3 is required" >&2; exit 1; }
exec python3 "${HERE}/framework/deploy/dsf-deploy.py" install-models --package "${HERE}" "$@"
