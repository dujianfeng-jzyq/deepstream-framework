# check=skip=InvalidDefaultArgInFrom
# 载荷构建：builder（本次 base + 工具链）→ runtime-check（原始 base 中自检）→ busybox。
# 载荷范围完全由 CMake 的 install 规则决定；最终镜像只用于向 app-files 卷播种。

# Compose 通过 additional_contexts 把这个名字绑定到本次 base 的构建结果。
ARG BASE_IMAGE

# ---------------- builder ----------------
FROM ${BASE_IMAGE} AS builder
ENV DEBIAN_FRONTEND=noninteractive

# 编译工具与测试依赖只在 builder：Redis 服务端、软件编解码插件、Docker/Compose CLI 与 git 让集成测试
# 真正执行而不是 SKIP。部署测试只用 `docker compose config` 解析文件，不需要（也不启动）守护进程。
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update || true; \
    apt-get install -y --no-install-recommends \
      build-essential cmake ninja-build pkg-config python3-pyflakes git \
      libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
      redis-server python3-redis gstreamer1.0-tools docker.io docker-compose-v2 \
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

# 编译与测试以非 root 运行：业务容器也是非 root，root 会掩盖权限相关的失败（并让对应用例 SKIP）。
RUN useradd --create-home dsf-builder && install -d -o dsf-builder /build/build
WORKDIR /build
# 进入构建的路径只由这些 COPY 决定（.dockerignore 只排除目录内的杂项）。先复制源码与测试读取的文件；
# profiles/ 与项目输入在编译之后复制，改动它们只重跑 profile 校验。
COPY CMakeLists.txt docker-compose.yaml ./
COPY framework/ framework/
COPY services/ services/
COPY plugins/ plugins/
COPY scripts/ scripts/
COPY tools/ tools/
USER dsf-builder
# DSF_REQUIRE_DEEPSTREAM：SDK 不可用即失败，不会产出缺少应用的载荷。
# DSF_FORBID_SKIPS：builder 带齐全部测试依赖，任何 SKIP 都计为失败。
RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
      -DDSF_REQUIRE_DEEPSTREAM=ON -DCMAKE_INSTALL_PREFIX=/workspace && \
    cmake --build build && \
    DSF_FORBID_SKIPS=1 ctest --test-dir build --output-on-failure --label-exclude profiles
COPY profiles/ profiles/
COPY project.env site-fields.env ./
RUN DSF_FORBID_SKIPS=1 ctest --test-dir build --output-on-failure --label-regex profiles
USER root
RUN cmake --install build

# 各 Python 单元的依赖（CMake 汇总）装进载荷，服务通过 PYTHONPATH 使用。
# ARG 之后的每个 RUN 都隐式依赖它：只在这里声明，改版本号或 pip 源不会重跑 apt 与编译。
ARG PIP_INDEX_URL=""
ARG APPS_VERSION=dev
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    set -eu; \
    if grep -qv '^#' build/python-requirements.txt; then \
      python3 -m pip install --disable-pip-version-check ${PIP_INDEX_URL:+--index-url "$PIP_INDEX_URL"} \
        --target /workspace/python --requirement build/python-requirements.txt; \
    fi; \
    printf '%s\n' "${APPS_VERSION}" > /workspace/VERSION; \
    chmod -R a+rX /workspace

# ---------------- runtime-check ----------------
# 回到原始 base：builder 追加的开发包（例如测试用的 python3-redis）可能掩盖载荷在运行环境中缺失的依赖。
# 以非 root、无网络、限时的方式运行自检；失败则没有最终镜像。项目可以没有 C++ 应用或 Python 服务。
# Python 服务的 --version 执行模块顶层的导入，未写进 requirements.txt 的顶层依赖在这里失败。
FROM ${BASE_IMAGE} AS runtime-check
COPY --from=builder /workspace /workspace
WORKDIR /workspace
USER 65534:65534
ENV XDG_CACHE_HOME=/tmp/cache CONTAINER_WORKSPACE=/workspace
RUN --network=none set -eu; \
    for app in bin/*; do \
      [ -e "$app" ] || continue; \
      timeout 60 "$app" --version; \
      timeout 60 "$app" --selftest; \
    done; \
    PYTHONPATH=/workspace/python python3 -c 'import dsf.service, dsf.config'; \
    for service in services/*/main.py; do \
      [ -e "$service" ] || continue; \
      PYTHONPATH=/workspace/python timeout 60 python3 "$service" --version; \
    done

# ---------------- payload ----------------
FROM busybox:1.36
COPY --from=runtime-check /workspace /workspace
# 构建时的源码版本（git 提交号，未提交改动时带 -dirty），打包时与当前源码比对。只影响这一层。
ARG SOURCE_REVISION=""
LABEL io.deepstream.source-revision="${SOURCE_REVISION}"
CMD ["true"]
