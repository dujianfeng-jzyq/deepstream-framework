#!/bin/sh
# 向载荷卷播种：apps 镜像的 CMD，在 busybox 中运行。
#   seed-payload.sh <载荷目录> <载荷标识文件> <卷挂载点>
# 卷名由载荷标识决定（docker-compose.yaml），所以正常情况下卷要么是空的新卷，要么已装着同一份载荷。
# 卷中的标记最后写入：有标记表示播种完整；没有标记而有内容只能是上次播种中断，清空重来——
# 业务服务等本容器成功退出后才启动，播种期间没有容器在读这个卷。
set -eu
source=$1
id_file=$2
target=$3
marker="$target/.payload-id"

want=$(cat "$id_file")
if [ -f "$marker" ]; then
  have=$(cat "$marker")
  [ "$have" = "$want" ] && exit 0
  # 开发机上卷名固定（dev），重新构建后会走到这里；不能就地替换：正在运行的业务服务不会随之重启。
  echo "载荷卷中的程序（$have）与 apps 镜像（$want）不一致。" >&2
  echo "  开发机：docker compose down，删除 docker volume ls 中本项目的 *_app-files-dev 卷，再 up" >&2
  exit 1
fi

find "$target" -mindepth 1 -maxdepth 1 -exec rm -rf {} \;
cp -a "$source/." "$target/"
sync
cp "$id_file" "$marker.tmp"
mv "$marker.tmp" "$marker"
sync
echo "已向载荷卷播种 $want"
