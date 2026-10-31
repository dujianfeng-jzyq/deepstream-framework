# 训练镜像：只在 x86 GPU 训练机上由 scripts/train.sh 使用，不进入 base、apps 或发布包（docs/training.md §9）。
# 训练工具本身不复制进镜像：train.sh 挂载仓库运行，改代码不需要重建镜像；镜像只提供运行环境。
# Ultralytics 以 AGPL-3.0 发布，只存在于这个镜像中。
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1
# ffmpeg：视频抽帧；libgl1、libglib2.0-0：Ultralytics 依赖的 OpenCV 运行库。
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# 依赖全部固定在 train-requirements.txt：同一个镜像 tag（Dockerfile 与依赖文件的摘要）总是同一套
# 训练环境，run 记录的后端版本才有意义。网络不稳时 pip 重试而不是整体失败。
ARG PIP_INDEX_URL=""
COPY train-requirements.txt /tmp/train-requirements.txt
RUN python3 -m pip install --no-cache-dir --retries 20 --timeout 120 ${PIP_INDEX_URL:+--index-url "$PIP_INDEX_URL"} \
      -r /tmp/train-requirements.txt && \
    python3 -c "import ultralytics, onnx, onnxruntime, onnxslim, cv2, PIL, yaml"
