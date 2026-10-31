#!/usr/bin/env bash
# 在训练镜像中运行训练工具（命令见 docs/training.md）：
#   scripts/train.sh <命令> [参数…]        例如 scripts/train.sh import road-assets ~/raw --source batch-01
#   TRAIN_ROOT=<目录>                      训练根目录，默认 ~/<项目名>/training
# 镜像 tag 随 Dockerfile 与依赖文件的摘要变化，缺少时自动构建。容器以当前用户运行，挂载仓库（可写：init 与
# publish 写 models/）、训练根目录（可写）与当前目录（只读）；参数中含 / 的已有路径按真实路径只读挂载。
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
DOCKERFILE="${ROOT}/framework/docker/train.Dockerfile"
REQUIREMENTS="${ROOT}/framework/docker/train-requirements.txt"

project_value() { sed -n "s/^$1=//p" "${ROOT}/project.env" | tail -n 1; }
project="$(project_value PROJECT_NAME)"
[[ -n "${project}" ]] || { echo "PROJECT_NAME is not set in project.env" >&2; exit 2; }
image="${project}/train:$(cat "${DOCKERFILE}" "${REQUIREMENTS}" | sha256sum | cut -c1-12)"

train_root="${TRAIN_ROOT:-${HOME}/${project}/training}"
mkdir -p "${train_root}"
train_root="$(cd "${train_root}" && pwd -P)"

if ! docker image inspect "${image}" >/dev/null 2>&1; then
  echo "building ${image} (first use, or the training Dockerfile or requirements changed)" >&2
  docker build -t "${image}" --build-arg "PIP_INDEX_URL=$(project_value PIP_INDEX_URL)" \
    -f "${DOCKERFILE}" "${ROOT}/framework/docker"
fi

inside() {  # 路径是否已在可写挂载之内
  case "$1/" in "${ROOT}/"*|"${train_root}/"*) return 0 ;; *) return 1 ;; esac
}
mounts=(-v "${ROOT}:${ROOT}" -v "${train_root}:${train_root}")
workdir="$(pwd -P)"
if ! inside "${workdir}" && [[ "${workdir}" != / ]]; then
  mounts+=(-v "${workdir}:${workdir}:ro")  # 相对路径参数在容器中照常解析
fi
args=()
for arg in "$@"; do
  if [[ "${arg}" == */* && -e "${arg}" ]]; then
    path="$(realpath "${arg}")"  # 符号链接指向的目录才是要挂载的内容
    inside "${path}" || mounts+=(-v "${path}:${path}:ro")
    args+=("${path}")
  else
    args+=("${arg}")
  fi
done

gpu=()
if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'; then
  gpu=(--runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility)
fi
tty=()
[[ -t 0 && -t 1 ]] && tty=(-it)

# 家目录与 Ultralytics 的设置放在训练根目录：容器用户没有可写的家目录，下载的预训练权重也随之保留。
exec docker run --rm ${tty[@]+"${tty[@]}"} ${gpu[@]+"${gpu[@]}"} --shm-size 8g \
  --user "$(id -u):$(id -g)" -e HOME="${train_root}/.home" -e YOLO_CONFIG_DIR="${train_root}/.home/ultralytics" \
  -e TRAIN_ROOT="${train_root}" "${mounts[@]}" -w "${workdir}" \
  "${image}" python3 "${ROOT}/framework/training/dsf-train.py" ${args[@]+"${args[@]}"}
