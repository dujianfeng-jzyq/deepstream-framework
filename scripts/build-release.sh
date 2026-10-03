#!/usr/bin/env bash
# 渲染 → 构建 base 与 apps 镜像 → 打包。
#   scripts/build-release.sh [--no-build] [--with-base] [--profile <name>]
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPLOY="${ROOT}/framework/deploy/dsf-deploy.py"

build=1
pack_args=()
profile_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-build) build=0 ;;
    --with-base) pack_args+=(--with-base) ;;
    --profile)
      [[ $# -ge 2 && -n "$2" ]] || { echo "--profile requires a name" >&2; exit 2; }
      profile_args=(--profile "$2"); shift ;;
    -h|--help) sed -n '2,3p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

# 先渲染：构建与打包都使用渲染出的镜像名与版本。
python3 "${DEPLOY}" render --root "${ROOT}" ${profile_args[@]+"${profile_args[@]}"}
if [[ "${build}" -eq 1 ]]; then
  (cd "${ROOT}" && docker compose build deepstream-base deepstream-apps)
fi
exec python3 "${DEPLOY}" pack --root "${ROOT}" ${pack_args[@]+"${pack_args[@]}"}
