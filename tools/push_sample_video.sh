#!/usr/bin/env bash
# 循环推送本地视频，模拟 RTSP 相机。前台运行，Ctrl-C 停止；推流断开后 5 秒重连。
#   [MEDIAMTX_PUBLISH_PASSWORD=口令] tools/push_sample_video.sh <视频文件> [rtsp-url]
# 默认推到本机 MediaMTX：rtsp://publisher:<口令>@127.0.0.1:8554/camera0（口令取自环境变量，可为空）。
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 || "$1" == -h || "$1" == --help ]]; then
  sed -n '2,4p' "$0" | sed 's/^# \{0,1\}//'
  [[ $# -ge 1 && ( "$1" == -h || "$1" == --help ) ]] && exit 0 || exit 2
fi
video="$1"

# 百分号编码：口令可含 @、# 等字符，ffmpeg 解析地址时解码。按字节处理，非 ASCII 字符也正确。
urlencode() {
  local LC_ALL=C text="$1" out="" char i
  for ((i = 0; i < ${#text}; i++)); do
    char="${text:i:1}"
    case "$char" in
      [A-Za-z0-9._~-]) out+="$char" ;;
      *) out+=$(printf '%%%02X' "'$char") ;;
    esac
  done
  printf '%s' "$out"
}
# MediaMTX 只接受以 publisher 登录的推流；口令为空时它接受任意口令，但仍要带上用户名。
url="${2:-rtsp://publisher:$(urlencode "${MEDIAMTX_PUBLISH_PASSWORD:-}")@127.0.0.1:8554/camera0}"
[[ -f "$video" ]] || { echo "video file not found: $video" >&2; exit 2; }
[[ "$url" == rtsp://* || "$url" == rtsps://* ]] || { echo "url must start with rtsp:// or rtsps://: $url" >&2; exit 2; }
command -v ffmpeg >/dev/null || { echo "ffmpeg (with libx264) is required" >&2; exit 2; }

# 停止信号只记录退出码；子进程在后台运行，wait 能被信号立即打断。
stop=0
trap 'stop=130' INT
trap 'stop=143' TERM

# 运行一个子进程直到结束或收到停止信号；停止时先 TERM，2 秒后 KILL，确保不留下孤儿进程。
run() {
  "$@" &
  local pid=$!
  while kill -0 "$pid" 2>/dev/null && [[ "$stop" -eq 0 ]]; do
    wait "$pid" 2>/dev/null || true
  done
  if [[ "$stop" -ne 0 ]]; then
    trap '' INT TERM
    kill -TERM "$pid" 2>/dev/null || true
    local deadline=$((SECONDS + 2))
    while kill -0 "$pid" 2>/dev/null && [[ "$SECONDS" -lt "$deadline" ]]; do sleep 0.1; done
    kill -KILL "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
}

# 地址中的口令不回显。
echo "streaming $video -> $(sed -E 's#^(rtsps?://[^:/@]*):[^/@]*@#\1:***@#' <<<"$url") (Ctrl-C to stop)"
while [[ "$stop" -eq 0 ]]; do
  # -re 按原始帧率读取，-stream_loop -1 无限循环；软件编码不依赖 GPU。
  run ffmpeg -loglevel warning -re -stream_loop -1 -i "$video" \
    -c:v libx264 -preset veryfast -tune zerolatency -g 30 -pix_fmt yuv420p \
    -an -f rtsp -rtsp_transport tcp "$url"
  [[ "$stop" -eq 0 ]] || break
  echo "stream ended; reconnecting in 5 seconds"
  run sleep 5
done
exit "$stop"
