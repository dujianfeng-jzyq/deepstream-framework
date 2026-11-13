# deepstream-framework

能实现任意 DeepStream 应用与服务的基础框架。产品项目从本仓库派生，在 `services/` 中增加自己的应用和服务，框架（`framework/`）负责建图、生命周期、元数据、消息、录像，以及构建、配置、打包与离线安装。

`services/example-app` 是一个完整示例：多路视频 → 推理 → 跟踪 → Redis 消息 / 分段录像 / RTSP 输出；`services/example-service` 演示消费录像事件的 Python 服务。

| 文档 | 内容 |
|---|---|
| [框架使用指南](docs/framework.md) | 架构、C++/Python 接口、块与组件、消息 schema、新增应用 |
| [构建与部署](docs/deployment.md) | 镜像、Compose、配置渲染、engine、打包与安装 |
| [设计文档](docs/design.md) | 设计目标、总体架构、核心机制与扩展边界 |
| [开发约定](AGENTS.md) | 修改代码时必须遵守的边界与验证要求 |

## 本地开发

程序在 builder 镜像中编译、测试并打进载荷，宿主不需要编译环境：`docker compose build deepstream-apps` 就是完整验证（全部 CTest 与原始 base 中的自检，见[构建与运行](#构建与运行)）。

宿主 CMake 是可选的，用于 IDE（`compile_commands.json`）和 C++ 的增量快速迭代。需要 C++17 编译器、CMake 3.22+、pkg-config，以及 GStreamer 1.20+、yaml-cpp、hiredis、spdlog/fmt、nlohmann-json 开发包：

```bash
cmake -S . -B build -G Ninja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build
(cd build && ctest --output-on-failure)
```

DeepStream SDK 自动探测（默认 `/opt/nvidia/deepstream/deepstream`，可用 `-DDEEPSTREAM_ROOT=` 指定）。SDK 不可用时只跳过需要它的应用与插件，框架、应用的配置与拓扑测试照常构建——在 macOS 上也能开发。pyflakes、redis-py、redis-server 缺失时相应用例 SKIP，builder 中都会执行。

只改 Python（部署工具、Python 服务）时不需要 CMake，Python 3.9+ 与 PyYAML 即可：

```bash
PYTHONPATH=framework/python:framework/deploy:framework/tests/python python3 -m unittest discover -s framework/tests/python
```

## 构建与运行

需要 Docker Compose 2.22+；完整推理需要 GPU、NVIDIA 驱动（宿主机 `nvidia-smi` 可用）与 NVIDIA Container Toolkit。

### NVIDIA Container Toolkit

需要 GPU 的服务（应用与 `model-engine-builder`）使用 `runtime: nvidia`，Docker 未注册该运行时会在启动时报 `unknown or invalid runtime name: nvidia`。构建机和目标机都要安装，Jetson 的 JetPack 已自带。x86_64 Ubuntu 上：

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker   # 写入 /etc/docker/daemon.json
sudo systemctl restart docker                        # 会重启正在运行的容器
```

验证：`docker info --format '{{json .Runtimes}}'` 中包含 `nvidia`，且 `docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi` 能列出 GPU。其他发行版见 [NVIDIA 安装文档](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)。

部署方案有两个：`server-ds71`（x86_64 服务器，Docker bridge 网络）与 `jetson-ds71`（Jetson，host 网络），宿主配置见[构建与部署](docs/deployment.md#jetsonhost-网络)。

### 启动

```bash
scripts/apply-profile.sh server-ds71           # 选择部署方案，渲染 .env 与 generated/
mkdir -p ~/<项目名>/data                         # 数据目录（默认值；项目名见 project.env）
docker compose build deepstream-base deepstream-apps
docker compose up -d
docker compose logs -f example-app
MEDIAMTX_PUBLISH_PASSWORD=<口令> tools/push_sample_video.sh video.mp4  # 用本地视频模拟相机（推到 MediaMTX）
```

数据目录必须在首次 `docker compose up` 前以当前用户创建：缺失的 bind 源会被 Docker 以 root 创建，服务以 `HOST_UID:HOST_GID` 运行时无法写入（如 `Permission denied: '/data/engines'`），此时执行 `sudo chown -R $(id -u):$(id -g) <数据目录>` 修复。安装器会自动创建并检查该目录。

开发机上的载荷卷名固定为 `<项目名>_app-files-dev`（卷名在构建之前确定）。重新构建程序后 `deepstream-apps` 会报告卷中的程序与镜像不一致并失败，不会悄悄运行旧程序：执行 `docker compose down && docker volume rm <项目名>_app-files-dev && docker compose up -d`（不用 `down -v`，它会连同服务的其他命名卷一起删除）。安装的项目按载荷内容命名载荷卷，不需要这一步。

## 发布与安装

```bash
scripts/build-release.sh               # 渲染、构建、打包到 dist/
scripts/build-release.sh --with-base   # 首次部署或 base 变化时带上 base 镜像
```

目标机解压后：

```bash
./install.sh             # 首次交互安装会询问 profile 与现场参数
./install.sh --dry-run   # 只预检并打印计划
./install.sh --ask       # 逐项修改现场参数（回车保留当前值）
```

安装器在停旧服务之前完成全部预检（完整性、平台、路径安全、数据目录、用新载荷执行 `--check-config`），失败时旧服务保持运行。

只更新模型时打模型包（不含镜像与程序），在解压后的模型包中安装：engine 在旧服务运行时构建，只重启引用模型的服务，见 [构建与部署](docs/deployment.md#7-只更新模型模型包)。

```bash
scripts/build-model-package.sh road-assets   # 打包已发布的模型到 dist/
./install.sh                                 # 目标机：更新模型
./install.sh --rollback                      # 换回上一份模型
```

## 新建项目与服务

```bash
scripts/init-project.sh my-product                 # 改项目名
scripts/new-app.sh my-detector --kind cpp          # 生成 C++ 应用单元
scripts/new-app.sh my-uploader --kind python       # 生成 Python 服务单元
```

## 目录

```text
framework/          框架：cmake/ cpp/ python/ deploy/ training/ docker/ compose/ tools/ templates/ tests/
services/<name>/    服务单元：代码、templates/ 配置模板、compose.yaml
plugins/<name>/     推理自定义库等 SDK 插件
models/<id>/        模型：model.yaml 规格、release.json 发布记录、artifacts/ 产物、nvinfer 配置（见 docs/training.md）
profiles/           部署方案（平台差异、Compose 覆盖、配置覆盖）
project.env         项目身份、版本与公共默认值（上游镜像写在各 profile）
site-fields.env     现场参数清单
scripts/ tools/     入口脚本与开发工具
```
