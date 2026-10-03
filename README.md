# deepstream-example

能实现任意 DeepStream 应用与服务的基础框架。产品项目从本仓库派生，在 `services/` 中增加自己的应用和服务，框架（`framework/`）负责建图、生命周期、元数据、消息、录像，以及构建、配置、打包与离线安装。

`services/example-app` 是一个完整示例：多路视频 → 推理 → 跟踪 → Redis 消息 / 分段录像 / RTSP 输出；`services/example-service` 演示消费录像事件的 Python 服务。

| 文档 | 内容 |
|---|---|
| [框架使用指南](docs/framework.md) | 架构、C++/Python 接口、块与组件、消息 schema、新增应用 |
| [构建与部署](docs/deployment.md) | 镜像、Compose、配置渲染、engine、打包与安装 |
| [设计文档](docs/design.md) | 设计目标、总体架构、核心机制与扩展边界 |
| [开发约定](AGENTS.md) | 修改代码时必须遵守的边界与验证要求 |

## 本地开发

需要 C++17 编译器、CMake 3.22+、pkg-config，以及 GStreamer 1.20+、yaml-cpp、hiredis、spdlog/fmt、nlohmann-json 开发包；Python 测试需要 Python 3.9+ 与 PyYAML（可选 pyflakes、redis-py、redis-server）。

```bash
cmake -S . -B build -G Ninja
cmake --build build
(cd build && ctest --output-on-failure)
```

DeepStream SDK 自动探测（默认 `/opt/nvidia/deepstream/deepstream`，可用 `-DDEEPSTREAM_ROOT=` 指定）。SDK 不可用时只跳过需要它的应用与插件，框架、应用的配置与拓扑测试照常构建——在 macOS 上也能开发。

## 构建与运行

需要 Docker Compose 2.22+；完整推理需要 NVIDIA 容器运行环境与 GPU。

```bash
scripts/apply-profile.sh dgpu-ds80             # 选择部署方案，渲染 .env 与 generated/
mkdir -p ~/deepstream-example/data             # 数据目录（project.env 的 HOST_DATA_PATH）
docker compose build deepstream-base deepstream-apps
docker compose up -d
docker compose logs -f example-app
tools/push_sample_video.sh video.mp4           # 用本地视频模拟相机（推到 MediaMTX）
```

更换程序时先构建成功，再 `docker compose down -v && docker compose up -d`：载荷卷只在为空时播种。

## 发布与安装

```bash
scripts/build-release.sh               # 渲染、构建、打包到 dist/
scripts/build-release.sh --with-base   # 首次部署或 base 变化时带上 base 镜像
```

目标机解压后：

```bash
./install.sh             # 首次交互安装会询问 profile 与现场参数
./install.sh --dry-run   # 只预检并打印计划
./install.sh --ask       # 重新填写现场参数
```

安装器在停旧服务之前完成全部预检（完整性、平台、路径安全、数据目录、用新载荷执行 `--check-config`），失败时旧服务保持运行。

## 新建项目与服务

```bash
scripts/init-project.sh my-product                 # 改项目名与数据目录
scripts/new-app.sh my-detector --kind cpp          # 生成 C++ 应用单元
scripts/new-app.sh my-uploader --kind python       # 生成 Python 服务单元
```

## 目录

```text
framework/          框架：cmake/ cpp/ python/ deploy/ docker/ compose/ tools/ templates/ tests/
services/<name>/    服务单元：代码、templates/ 配置模板、compose.yaml
plugins/<name>/     推理自定义库等 SDK 插件
models/<id>/        模型、标签与 nvinfer 配置
profiles/           部署方案（平台差异、Compose 覆盖、配置覆盖）
project.env         项目身份、版本、上游镜像与公共默认值
site-fields.env     现场参数清单
scripts/ tools/     入口脚本与开发工具
```
