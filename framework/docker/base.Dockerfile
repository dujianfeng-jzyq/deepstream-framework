# check=skip=InvalidDefaultArgInFrom
# base：上游 DeepStream + 框架运行依赖 + 项目额外的运行依赖（BASE_APT_PACKAGES）。
# 编译工具不进 base；builder 在此之上追加。修改本文件会改变 base 的 tag（构建输入摘要）。

# 由 Compose 传入；不写默认值，同一文件服务多个平台与 DeepStream 版本。
ARG DEEPSTREAM_IMAGE
FROM ${DEEPSTREAM_IMAGE}

ARG BASE_APT_PACKAGES=""
ENV DEBIAN_FRONTEND=noninteractive

# python3-yaml：服务配置读取与无宿主依赖的渲染；gstreamer1.0-rtsp：rtspclientsink；
# 字体：nvdsosd 绘制中文标签；公共库开发包同时提供运行库与 builder 需要的头文件。
# 上游镜像的部分软件源可能不可用，update 失败不中断；缺包仍由 install 报错。
RUN apt-get update || true; \
    apt-get install -y --no-install-recommends \
      python3 python3-pip python3-yaml \
      gstreamer1.0-rtsp \
      libyaml-cpp-dev libhiredis-dev libspdlog-dev libfmt-dev nlohmann-json3-dev \
      fontconfig fonts-wqy-microhei \
      ${BASE_APT_PACKAGES} \
    && fc-cache -f \
    && rm -rf /var/lib/apt/lists/*
