# check=skip=InvalidDefaultArgInFrom
# 载荷构建：builder（本次 base + 工具链）→ runtime-check（原始 base 中自检）→ busybox。
# 载荷范围完全由 CMake 的 install 规则决定；最终镜像只用于向 app-files 卷播种。

# Compose 通过 additional_contexts 把这个名字绑定到本次 base 的构建结果。
ARG BASE_IMAGE

# ---------------- builder ----------------
FROM ${BASE_IMAGE} AS builder
ARG PIP_INDEX_URL=""
ARG APPS_VERSION=dev
ENV DEBIAN_FRONTEND=noninteractive

# 编译工具与测试依赖只在 builder：Redis 服务端与软件编解码插件让集成测试真正执行而不是 SKIP。
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update || true; \
    apt-get install -y --no-install-recommends \
      build-essential cmake ninja-build pkg-config python3-pyflakes \
      libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
      redis-server python3-redis gstreamer1.0-tools \
      gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav

# 上游镜像裁掉了软件编解码库的文件却保留了包记录，普通 apt install 不会恢复；
# 使用 SDK 自带的恢复脚本，再以实际元素检查结果为准。
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    bash /opt/nvidia/deepstream/deepstream/user_additional_install.sh; \
    apt-get install --reinstall -y --no-install-recommends libmp3lame0 libxvidcore4 || true; \
    for element in x264enc h264parse splitmuxsink mp4mux decodebin; do \
      gst-inspect-1.0 "$element" >/dev/null || { echo "missing GStreamer element: $element" >&2; exit 1; }; \
    done

WORKDIR /build
COPY ./ ./
# DSF_REQUIRE_DEEPSTREAM：SDK 不可用即失败，不会产出缺少应用的载荷。
RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
      -DDSF_REQUIRE_DEEPSTREAM=ON -DCMAKE_INSTALL_PREFIX=/workspace && \
    cmake --build build && \
    ctest --test-dir build --output-on-failure && \
    cmake --install build

# 各 Python 单元的依赖（CMake 汇总）装进载荷，服务通过 PYTHONPATH 使用。
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    set -eu; \
    if grep -qv '^#' build/python-requirements.txt; then \
      python3 -m pip install --disable-pip-version-check ${PIP_INDEX_URL:+--index-url "$PIP_INDEX_URL"} \
        --target /workspace/python --requirement build/python-requirements.txt; \
    fi; \
    printf '%s\n' "${APPS_VERSION}" > /workspace/VERSION; \
    chmod -R a+rX /workspace

# ---------------- runtime-check ----------------
# 回到原始 base：builder 追加的开发包可能掩盖载荷在运行环境中缺失的依赖。
# 以非 root、无网络、限时的方式运行自检；失败则没有最终镜像。
FROM ${BASE_IMAGE} AS runtime-check
COPY --from=builder /workspace /workspace
WORKDIR /workspace
USER 65534:65534
ENV XDG_CACHE_HOME=/tmp/cache CONTAINER_WORKSPACE=/workspace
RUN --network=none set -eu; \
    for app in bin/*; do \
      timeout 60 "$app" --version; \
      timeout 60 "$app" --selftest; \
    done; \
    PYTHONPATH=/workspace/python python3 -c 'import dsf.service, dsf.config'

# ---------------- payload ----------------
FROM busybox:1.36
COPY --from=runtime-check /workspace /workspace
CMD ["true"]
