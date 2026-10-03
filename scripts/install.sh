#!/usr/bin/env bash
# 发布包安装入口（位于发布包根目录）。只需要 Python 3.9+ 标准库与 Docker Compose 2.22+。
#   ./install.sh [--ask] [--clean-install] [--dry-run] [--profile <name>] [--install-dir <dir>]
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
command -v python3 >/dev/null || { echo "install.sh: python3 is required" >&2; exit 1; }
exec python3 "${HERE}/framework/deploy/dsf-deploy.py" install --package "${HERE}" "$@"
