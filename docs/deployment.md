# 构建与部署

本文说明镜像、Compose、配置渲染、engine 预构建、打包与安装。框架接口见 [framework.md](framework.md)。部署工具要求 Python 3.9+，镜像构建与部署要求 Docker Compose 2.22+；真实推理需要 NVIDIA 容器运行环境。

## 1. 镜像与载荷

```text
上游 DeepStream ──继承──> base ──继承──> builder（工具链、构建、CTest、cmake --install）
                          └──继承──> runtime-check（复制载荷，非 root、无网络自检）
busybox:1.36 ──继承──> apps（只复制通过自检的载荷，向 app-files 卷播种）
```

| 位置 | 内容 |
|---|---|
| base | 上游 DeepStream、Python3/PyYAML/pip、RTSP 插件、中文字体、框架公共库（yaml-cpp、hiredis、spdlog、fmt、nlohmann-json）、`BASE_APT_PACKAGES` |
| 载荷 `/workspace` | `bin/` 应用、`lib/` 插件、`share/` 许可证、`python/` 框架运行时与服务依赖、`services/` Python 服务、`tools/` 工具、`VERSION` |
| 镜像外 | 生成配置、模型、现场答案、录像与其他持久数据 |

- 载荷范围只由 CMake 的 install 规则决定（`dsf_add_app`、`dsf_add_plugin`、`dsf_add_python_service`）；Python 依赖由 CMake 汇总各单元的 `requirements.txt` 后 `pip --target` 安装。
- 发布构建设置 `DSF_REQUIRE_DEEPSTREAM=ON`：SDK 不可用即失败，不会产出缺少应用的载荷。
- builder 安装 Redis 服务端、软件编解码插件（并用 SDK 自带脚本恢复上游裁掉的编解码库）、pyflakes 与 Docker/Compose CLI，以非 root 运行 CTest，并设置 `DSF_FORBID_SKIPS=1`：任何 SKIP 都使构建失败。
- `profiles/`、`project.env`、`site-fields.env` 在编译之后才复制，随后只运行带 `profiles` 标签的测试（每个 profile 渲染并经 `docker compose config` 解析）；改动它们不会重新编译。
- runtime-check 回到原始 base，对每个应用执行 `--version` 与 `--selftest`，并导入 Python 运行时；任何失败都不会生成 apps 镜像。
- JPEG 编码库在 `image.enabled` 的采集组件启动时才加载，构建和无 GPU 自检不加载其设备依赖。目标 Jetson 仍须使用与 JetPack 匹配的 DeepStream 镜像和 NVIDIA runtime；缺少运行库时启用图片会明确报错。
- 镜像名由渲染器派生：`<项目>/deepstream-<DS版本>-base:<BASE_VERSION>-<输入摘要>`、`<项目>/deepstream-<DS版本>-apps:<APPS_VERSION>`。base 的输入摘要覆盖 base Dockerfile、上游镜像与 `BASE_APT_PACKAGES`：输入变化一定得到新 tag，同名 tag 不会代表两种内容。

## 2. 服务与 Compose

渲染器把 Compose 文件按顺序写进 `.env` 的 `COMPOSE_FILE`：

```text
docker-compose.yaml               框架级：deepstream-base、deepstream-apps（播种）、model-engine-builder
services/<unit>/compose.yaml      各服务单元（按名称排序）
profiles/<profile>.yaml           平台差异（runtime: nvidia、设备、用户）
generated/network.compose.yaml    仅当 profile 设置 x-network-mode 时生成，排在最后
```

| 服务 | 行为 |
|---|---|
| `deepstream-base` | 构建目标与运行时父镜像，启动后立即成功退出 |
| `deepstream-apps` | 向空的 `<项目>_app-files` 卷播种后退出 |
| `model-engine-builder` | 预构建 engine，成功退出后业务服务才启动 |
| 业务服务 | `extends: {file: framework/compose/defaults.yaml, service: dsf-service}` |
| `redis`、`mediamtx` | 基础设施单元，可删除；依赖它们的服务使用 `required: false` |

公共定义 `dsf-service`：`image: ${BASE_IMAGE}`、渲染用户的 UID/GID、`restart: unless-stopped`、预检标签 `io.deepstream.config-check: "true"`、挂载点环境变量、`PYTHONPATH`、`DSF_READY_FILE`、只读载荷卷与配置目录、可写数据目录、`/etc/localtime`、就绪健康检查（就绪文件 30 秒内更新过）、等待播种完成、日志轮转。注意 `extends` 文件中的相对路径以该文件所在目录为基准，单元 compose.yaml 中的相对路径以项目根为基准。

固定挂载点：

| 变量 | 路径 | 权限 |
|---|---|---|
| `CONTAINER_WORKSPACE` | `/workspace` | 载荷，只读，工作目录 |
| `CONTAINER_CONFIG_PATH` | `/config` | `generated/config`，只读 |
| `CONTAINER_DATA_PATH` | `/data` | `HOST_DATA_PATH`，可写，跨发布保留 |
| `CONTAINER_MODELS_PATH` | `/models` | `models/`，业务只读，engine 预构建可写 |

日常命令：

```bash
scripts/apply-profile.sh <profile>                   # 渲染；省略 profile 时沿用上次选择
docker compose build deepstream-base deepstream-apps
docker compose down && docker volume rm <项目名>_app-files && docker compose up -d  # 更换载荷：卷只在为空时播种
docker compose run --rm example-app --dump-pipeline   # 一次性动作覆盖入口参数
```

## 3. 配置渲染

| 输入 | 职责 |
|---|---|
| `project.env` | 项目身份、版本、上游镜像、公共默认值、依赖镜像（`*_IMAGE`） |
| `profiles/<name>.yaml` | `x-build`（变量覆盖）、`services`（Compose 覆盖）、`x-network-mode`（全部服务的网络模式）、`x-app-<type>`（配置覆盖） |
| `site-fields.env` | 现场参数清单：`KEY=` 必填，`KEY=""` 可空，`KEY=值` 默认值；紧邻注释为提示 |
| `site.env` | 现场答案（0600，不提交，不打包）；写成 `KEY="值"`，手工编辑时 `KEY=""` 表示空值 |
| `services/*/templates/<type>/*.yaml` | 配置模板 → `generated/<type>/*.yaml` |

- 变量优先级：清单默认值 < `project.env` < `x-build` < `site.env`。显式的空答案只在可空字段上接受，且不回退到默认值。交互安装的提问默认值同样含 `project.env` 与所选 profile 的 `x-build`，回车保存的答案不会遮住 profile 的设置。
- 模板中的 `${VAR}` 必须有定义；`x-app-<type>: {<文件名>: {...}}` 在替换后深合并（映射递归，列表与标量整体替换）。
- profile 顶层只接受 `services`、`x-build`、`x-network-mode`、`x-app-*`；`PROJECT_NAME` 不能被覆盖；现场参数不能占用身份、版本、镜像、挂载点等保留键。
- 渲染器派生 `DEEPSTREAM_VERSION`、`BASE_IMAGE`、`APPS_IMAGE`、挂载点、`HOST_UID/HOST_GID`、`CONFIG_PROFILE`、`COMPOSE_FILE`；`project.env` 中带有镜像名时（发布包安装）沿用包中的镜像名。
- `x-network-mode: <mode>` 给所有未显式声明 `network_mode` 的服务（包括框架服务与以后新增的单元）设置该值，写入 `generated/network.compose.yaml`；显式声明的（如 `deepstream-apps` 的 `none`）保持不变，声明了 `networks` 的服务报错。只要有一个服务留在默认网络，Compose 就会创建 bridge 网络，所以不要在 `services` 中逐个设置。
- 先计算完整计划，再整体替换 `.env` 与 `generated/`；校验失败时旧文件保持不变。`project.env` 等 env 文件中开头的 `~` 在运行机器上展开；env 文件不执行 shell、不剥引号。例外：`site.env` 的值去掉一对外层引号（与 `site-fields.env` 中 `KEY=""` 的含义一致）；`.env` 由 Compose 按它自己的 dotenv 规则读取（未加引号时 `$` 会被插值、` #` 之后被当作注释、开头的引号必须闭合），渲染器写入时一律加引号，现场口令中的这些字符原样到达 Compose 与配置。

## 4. engine 预构建

`model-engine-builder` 运行 `tools/prepare_model_engines.py --models-path /models`：

- 扫描 `*.txt`，`[property]` 段同时含 `onnx-file` 与 `model-engine-file` 的就是待构建配置；输出路径完全遵循 `model-engine-file`（相对路径以配置目录为基准）。业务服务只读加载，nvinfer 不会把 engine 写到配置指定的路径，因此必须预构建。
- 指纹记录 ONNX 哈希、TensorRT 版本、GPU 计算能力、`network-mode`/`batch-size`/`int8-calib-file`、trtexec 参数，以及 engine 自身的哈希；engine 非空且指纹完全一致才复用，二者不配套时一定重建。
- `network-mode=2` 自动加 `--fp16`，`=1` 加 `--int8`（有校准文件时加 `--calib`）。其他参数写在配置旁的 `<配置名>.trtexec`（例如动态形状），计入指纹。
- 同一 engine 的检查与构建用文件锁串行化；构建写到同目录临时文件，成功后原子替换，失败保留旧 engine。
- 发布包不包含 engine、指纹与锁文件：engine 必须在目标 GPU 上构建。

## 5. 打包

```bash
scripts/build-release.sh [--no-build] [--with-base] [--profile <name>]
```

先渲染，再构建 base 与 apps（`--no-build` 跳过），然后打包到 `dist/<项目>-v<APPS_VERSION>-ds<DS版本>.tar.gz`：

- apps 与依赖镜像（所有非空 `*_IMAGE`，排除上游与本项目镜像）每次都带；缺失的依赖镜像按 base 的平台拉取；所有镜像必须与 base 同平台。base 只在 `--with-base` 时带。
- 部署文件：`docker-compose.yaml`、`framework/{compose,deploy,python/dsf}`、各单元的 `compose.yaml` 与 `templates/`、`profiles/`、`models/`、`site-fields.env`、`install.sh`。不含源码、Dockerfile、渲染结果、现场答案与 engine。
- apps 镜像的 tag 只含版本号，源码改了而没有重新构建时同名镜像仍是旧程序。渲染记录源码版本 `SOURCE_REVISION`（git 提交号，有未提交改动时带 `-dirty`），构建把它写进 apps 镜像的标签 `io.deepstream.source-revision`；打包时两者不一致即拒绝，提示重新构建。工作区有未提交改动时只警告：无法证明镜像包含了哪些改动。不是 git 仓库时不比对。
- 构建只由渲染出的 `.env` 决定：脚本忽略 shell 中的 `COMPOSE_FILE`、`COMPOSE_PROJECT_NAME`、`COMPOSE_PROFILES` 等变量（它们优先于 `.env`）。
- `release.env`：项目公共输入加本次发布事实（版本、镜像、平台、建议 profile、源码版本）。`package-files.json`：全部文件的 SHA-256（完整性校验，不是签名）。

## 6. 安装

```bash
./install.sh [--profile <name>] [--ask] [--clean-install] [--dry-run] [--install-dir <dir>] [--ready-timeout <s>]
```

默认安装到 `${INSTALL_ROOT:-${XDG_DATA_HOME:-~/.local/share}}/<项目名>`。顺序不可调换：

1. 读取 `release.env`，核对本机平台与包完整性（缺失、改动、多余文件或符号链接都拒绝）。
2. 检查安装目标：安装目录与包不重叠；控制文件、发布路径不能是符号链接，也不能与已有文件/目录类型冲突。之后才读取旧的选择与答案。
3. 选择 profile（`--profile` > 已安装的选择 > 交互选择 > 包内建议），收集现场答案（升级沿用；`--ask`、`--clean-install` 重新收集；非交互时只保存已有答案；提问默认值含所选 profile 的 `x-build`，宿主没有 PyYAML 时为此提前加载包内的 base）。检查清洁安装范围与退役文件冲突。
4. 加载镜像（不影响运行中的旧容器）；在临时目录渲染新配置（宿主没有 PyYAML 时用 base 镜像渲染）；按最终 Compose 检查并准备可写 bind 挂载；核对每个服务镜像的平台；从新 apps 镜像取出载荷，离线、只读地执行所有带预检标签服务的 `--check-config`。
5. `--dry-run` 到此结束并打印计划。否则重验目标，用旧定义 `docker compose down --remove-orphans`，再只删除载荷卷 `<项目名>_app-files` 使其重新播种（不用 `down -v`：它会删除项目的全部命名卷，包括服务的持久数据；首次安装改为检查遗留卷）。
6. 按需清空安装目录；删除未被修改的退役受管理文件；复制发布文件；写入 `project.env`（由 `release.env` 派生）、`site.env`、`.managed-files.json` 与预检过的配置快照（不再重新渲染）。
7. `docker compose config --quiet`、`up -d`，有界等待：常驻服务运行且健康，一次性服务成功退出。

规则：

- 可写 bind 源不得与安装目录或发布包互相包含，唯一例外是 `model-engine-builder` 写本次安装的 `models/`；`/data` 必须是可写 bind；显式 `create_host_path: false` 的源必须已存在；缺失目录由安装用户创建，失败时指出阻塞的路径层级。
- 普通升级保留现场新增的、与发布路径不冲突的文件；退役受管理文件被现场修改过时在停服前失败。
- 清洁安装拒绝根目录、家目录、与包重叠的目录及不像安装目录的目录；数据目录与镜像保留。
- 调用 shell 中的 `COMPOSE_FILE` 等选择变量被忽略，预检与执行看到同一份定义。
- 安装不是事务：停服后的失败不自动回滚，重新安装旧包即可恢复；旧镜像与外部数据保留。停服之后、启动之前的失败或中断（包括 Ctrl-C）会说明旧服务已停止、安装目录可能写了一半，以及如何完成（用同一个包重新安装）或退回（安装旧包）。
- 部署工具需要 Python 3.9+；版本过低时在启动时报错。

## 7. 平台与现场验证

profile 中写平台差异：`dgpu-ds71`、`dgpu-ds80` 为 x86_64 独立显卡，`arm64-ds71`、`arm64-ds80` 用于 Apple Silicon 上的构建与无 GPU 自检（不运行推理），`jetson-ds71-host` 为使用 host 网络的 Jetson。Jetson 部署时选择与 JetPack 匹配的上游镜像，在 `services` 中加入 `runtime: nvidia`、所需设备与 `group_add`，编码器差异写在应用配置的 `encoder.properties`。

### 不支持 bridge 的 Jetson（host 网络）

部分 Jetson 内核缺少 Docker bridge 所需的模块。先确认是否只是缺模块（`lsmod | grep -E 'br_netfilter|iptable_nat'`、`docker network create` 能否成功），能补模块时优先保持 bridge。否则 dockerd 需要关闭 bridge 与 iptables 才能启动：

```json
{
  "bridge": "none",
  "iptables": false,
  "runtimes": {"nvidia": {"path": "nvidia-container-runtime", "args": []}}
}
```

写入 `/etc/docker/daemon.json` 后 `sudo systemctl restart docker`；`runtimes` 一段保持 JetPack 原有内容。`docker build` 的 `RUN` 默认也走 bridge，因此镜像在编译服务器上构建，用 `scripts/build-release.sh --with-base --profile jetson-ds71-host` 打包，Jetson 上只安装。

`jetson-ds71-host` 用 `x-network-mode: host` 让全部服务使用宿主网络，并处理随之而来的差异：

- 服务名不再解析：`REDIS_HOST`、`RTSP_URL` 与 `rtsp_out.url` 改为 `127.0.0.1`。改成远端 Redis 或真实相机时照常填写现场参数。
- 监听直接暴露到现场网络：Redis 没有口令，只绑定 `127.0.0.1`；MediaMTX 只保留 RTSP（8554/tcp、8000-8001/udp）。`ports` 映射在 host 网络下不生效，也不再受 Docker 端口发布绕过 ufw 的影响，改由宿主防火墙控制。
- 端口与宿主共享：同一台机器不能再运行第二套实例，宿主上已有的 6379、8554 会冲突。
- RTSP over UDP、组播与 ONVIF 发现在 host 网络下可用。

镜像构建与自检不证明 GPU 推理可用。目标机仍需核对：真实 RTSP 输入、推理与跟踪结果、Redis 消息、完整 MP4 与停止时最后一段、RTSP 输出、engine 再次启动时复用、Linux 下的属主与设备权限。Docker Desktop 上的结果不能替代 Linux 验证。
