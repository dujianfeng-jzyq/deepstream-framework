# 开发约定

人和 AI agent 修改本仓库时都按本文件执行。框架接口见 [docs/framework.md](docs/framework.md)，构建与部署见 [docs/deployment.md](docs/deployment.md)，设计依据见 [docs/design.md](docs/design.md)。

## 核心原则

本仓库是能实现任意 DeepStream 应用的基础框架。每次修改先判断它属于**机制**（放进 `framework/`）还是**策略**（放进 `services/`、`plugins/`、`models/`、`profiles/`）：

- 框架不引用任何具体产品的名字（`framework/tests/python/test_repository.py` 检查）。
- 框架不预设源的数量、拓扑或输出组合；新增块时同时提供 `properties:` 透传。
- 新增应用、服务、插件、模型只加目录，不修改框架；做不到时说明框架缺少扩展点，先补扩展点。
- 不要为了一个应用改变框架的公共语义；需要不同行为时给框架增加可选项。

## 代码

- C++17，两空格缩进，`-Wall -Wextra -Wpedantic` 无警告。类型与函数 `PascalCase`，变量 `snake_case`，常量 `kPascalCase`，成员变量以 `_` 结尾。
- Python 3.9+，部署工具（`framework/deploy`）只用标准库，渲染时才导入 PyYAML；pyflakes 无警告。
- 标识符用英文，注释与文档用中文。注释说明契约、原因和不可调换的顺序，不复述代码，不记录调试过程。
- 组件构造函数不得有副作用（`--check-config` 也会构造组件）；文件、网络、设备操作放在生命周期回调中。
- 媒体线程（pad probe、流线程信号）只复制数据和入队，不等待网络或锁住控制线程正在等待的资源。

## 构建与依赖

- 新的第三方依赖在 `framework/cmake/DsfDependencies.cmake` 中引入；运行库进入 base 时更新 `BASE_APT_PACKAGES` 或 base Dockerfile（base 的 tag 随输入摘要自动变化）。
- 载荷内容只由 CMake 的 `install` 规则决定：应用用 `dsf_add_app`，插件用 `dsf_add_plugin`，Python 服务用 `dsf_add_python_service`。
- 需要 SDK 的代码只能放在 SDK 目标中（`dsf_ds_meta`、应用可执行文件、插件）；配置与拓扑放在不依赖 SDK 的库中，以便在开发机上测试。

## 部署契约

- 业务容器使用 `${BASE_IMAGE}` + 只读载荷卷；apps 镜像最终阶段是 busybox，只复制通过原始 base 自检的载荷。
- 固定挂载点 `/workspace /config /data /models`；载荷、配置、模型只读，数据可写；业务以渲染用户的 UID/GID 运行。
- 服务通过 `extends: framework/compose/defaults.yaml#dsf-service` 获得公共定义，不写 `container_name`。
- 服务必须支持无副作用的 `--check-config`，并更新 `$DSF_READY_FILE` 表示主循环在推进。
- 安装器在停旧服务前完成全部预检，预检失败不得停服或改动安装目录；安装不自动回滚。

## 验证

- 必须：`docker compose build deepstream-apps`。builder 中的 CTest 要求 SDK，并带齐 pyflakes、redis-server 与软件编解码插件，用例真正执行而不是 SKIP；之后在原始 base 中做非 root、无网络自检。任何失败都不产出镜像。
- 可选的快速循环：宿主 `cmake --build build && (cd build && ctest --output-on-failure)` 用于框架、配置与拓扑代码的增量迭代；只改 Python 时直接运行 `python3 -m unittest`（`PYTHONPATH` 与 CTest 相同，见 `framework/cmake/DsfUnits.cmake`）。宿主缺少 SDK、pyflakes 或 redis-server 时相应用例 SKIP，不能替代镜像构建。
- 真实视频、推理、RTSP、录像收尾、engine 复用与 Linux 权限只能在目标环境验证，镜像构建与容器健康检查不能替代。

## 提交

不自动提交，只在用户明确要求时提交。格式：`<type>(<scope>): <中文主题>`，type 为 feat/fix/build/scripts/docs/test/refactor/chore，正文说明动机与影响。
