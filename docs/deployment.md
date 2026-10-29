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
- runtime-check 回到原始 base，对每个应用执行 `--version` 与 `--selftest`，导入 Python 运行时，并对每个 Python 服务执行 `services/<name>/main.py --version`（执行模块顶层的导入：未写进 `requirements.txt` 的顶层依赖在这里失败，函数内的延迟导入不在检查范围内）；没有应用或服务时相应步骤跳过；任何失败都不会生成 apps 镜像。
- JPEG 编码库在 `image.enabled` 的采集组件启动时才加载，构建和无 GPU 自检不加载其设备依赖。目标 Jetson 仍须使用与 JetPack 匹配的 DeepStream 镜像和 NVIDIA runtime；缺少运行库时启用图片会明确报错。
- 镜像名由渲染器派生：`<项目>/deepstream-<DS版本>-base:<BASE_VERSION>-<输入摘要>`、`<项目>/deepstream-<DS版本>-apps:<APPS_VERSION>`。base 的输入摘要覆盖 base Dockerfile、上游镜像与 `BASE_APT_PACKAGES`：输入变化一定得到新 tag，同名 tag 不会代表两种内容。

## 2. 服务与 Compose

渲染器把 Compose 文件按顺序写进 `.env` 的 `COMPOSE_FILE`：

```text
docker-compose.yaml               框架级：deepstream-base、deepstream-apps（播种）、model-engine-builder
services/<unit>/compose.yaml      各服务单元（按名称排序）
profiles/<profile>.yaml           平台差异（设备、用户组、网络）
generated/network.compose.yaml    仅当 profile 设置 x-network-mode 时生成，排在最后
```

| 服务 | 行为 |
|---|---|
| `deepstream-base` | 构建目标与运行时父镜像，启动后立即成功退出 |
| `deepstream-apps` | 向空的 `<项目>_app-files` 卷播种后退出 |
| `model-engine-builder` | 预构建 engine，成功退出后业务服务才启动 |
| 业务服务 | `extends: {file: framework/compose/defaults.yaml, service: dsf-service}` |
| `redis`、`mediamtx` | 基础设施单元，可删除；依赖它们的服务使用 `required: false` |

基础设施单元的访问控制：

- `redis` 只在 Compose 网络内访问。现场参数 `REDIS_PASSWORD` 非空时设为它的口令（经标准输入传给 `redis-server`，不出现在进程参数中），各服务用同一个口令连接；为空时不设口令。profile 可用 `REDIS_EXTRA_ARGS` 追加 `redis-server` 参数。
- `mediamtx` 的配置由 `services/mediamtx/templates/mediamtx/mediamtx.yaml` 渲染到 `generated/mediamtx/`。观看对现场网络开放、不需要口令：RTSP `rtsp://<主机>:8554/<路径>`，浏览器 `http://<主机>:8889/<路径>`（WebRTC，媒体走 8189/udp）。`MEDIAMTX_HOST` 是浏览器连接媒体端口用的宿主地址，见下文“浏览器查看（WebRTC）”。推流必须以 `publisher` 与 `MEDIAMTX_PUBLISH_PASSWORD` 登录，防止网络中任何人替换相机画面或应用输出；口令只能由字母、数字与 `!$()*+.;<=>[]^_-@#&` 组成（MediaMTX 的限制，example-app 的 `--check-config` 在预检中检查），为空时 `publisher` 用任意口令都能推流，只适合开发环境。模拟相机推流用 `MEDIAMTX_PUBLISH_PASSWORD=<口令> tools/push_sample_video.sh <视频>`。

### 浏览器查看（WebRTC）

WebRTC 的媒体不走页面所在的 HTTP 连接：MediaMTX 在应答中告诉浏览器去连哪些地址的 8189/udp。bridge 网络中 MediaMTX 只知道容器内的地址与 `127.0.0.1`，所以：

| 查看位置 | 需要的设置 |
|---|---|
| 本机浏览器 | 无（`MEDIAMTX_HOST` 保持默认 `127.0.0.1`） |
| 局域网其他机器 | `MEDIAMTX_HOST` 填本机的局域网 IP，且该 IP 固定（静态或 DHCP 保留）；否则页面能打开但没有画面 |

`MEDIAMTX_HOST` 不能留空：空地址会让 MediaMTX 拒绝所有 WebRTC 请求，本机也看不了。不要自动探测这个地址：多网卡或代理隧道下默认路由的源地址往往不是局域网地址。bridge 网络下端口由 Docker 发布，不受宿主防火墙（ufw）影响。

不想逐台填写 IP 时，可以用 `x-network-mode: host` 的 profile：MediaMTX 直接看到宿主网卡，自动通告局域网地址，`MEDIAMTX_HOST` 保持默认即可。代价见下文“不支持 bridge 的 Jetson”：不能关闭 `MTX_WEBRTC`，Redis 须用 `REDIS_EXTRA_ARGS: --bind 127.0.0.1` 只绑定回环地址，宿主防火墙须放行 8554/tcp、8889/tcp、8189/udp，同一台机器只能运行一套实例。不要只把 MediaMTX 单独放到 host 网络：应用经宿主网关推流，启用 ufw 的宿主通常会拦截容器到宿主的连接，RTSP 输出因此完全失效。

浏览器的 WebRTC 多数只能播放 H.264，`encoder.codec: h265` 的输出只能用 RTSP 播放器查看（example-app 在 `--check-config` 与启动时给出警告）。

公共定义 `dsf-service`：`image: ${BASE_IMAGE}`、渲染用户的 UID/GID、`restart: unless-stopped`、预检标签 `io.deepstream.config-check: "true"`、挂载点环境变量、`PYTHONPATH`、`DSF_READY_FILE`、只读载荷卷与配置目录、可写数据目录、`/etc/localtime`、就绪健康检查（`$DSF_READY_FILE` 30 秒内更新过）、等待播种完成、日志轮转。注意 `extends` 文件中的相对路径以该文件所在目录为基准，单元 compose.yaml 中的相对路径以项目根为基准。

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
| `project.env` | 项目身份、版本、公共默认值、依赖镜像（`*_IMAGE`）；本项目不在这里设上游镜像 `DEEPSTREAM_IMAGE`，由各 profile 写明 |
| `profiles/<name>.yaml` | `x-platform`（面向的平台）、`x-build`（变量覆盖）、`services`（Compose 覆盖）、`x-network-mode`（全部服务的网络模式）、`x-app-<type>`（配置覆盖） |
| `site-fields.env` | 现场参数清单：`KEY=` 必填，`KEY=""` 可空，`KEY=值` 默认值；紧邻注释为提示 |
| `site.env` | 现场答案（0600，不提交，不打包）；写成 `KEY="值"`，手工编辑时 `KEY=""` 表示空值 |
| `services/*/templates/<type>/*.yaml` | 配置模板 → `generated/<type>/*.yaml` |

- 变量优先级：清单默认值 < `project.env` < `x-build` < `site.env`。显式的空答案只在可空字段上接受，且不回退到默认值。交互安装的提问默认值同样含 `project.env` 与所选 profile 的 `x-build`，回车保存的答案不会遮住 profile 的设置。
- 模板中的 `${VAR}` 必须有定义；`x-app-<type>: {<文件名>: {...}}` 在替换后深合并（映射递归，列表与标量整体替换）。
- profile 顶层只接受 `services`、`x-platform`、`x-build`、`x-network-mode`、`x-app-*`；`PROJECT_NAME` 不能被覆盖；现场参数不能占用身份、版本、镜像、挂载点等保留键。
- `x-platform: amd64|arm64`（可选）声明 profile 面向的平台。打包时镜像必须是这个平台；安装时只列出与发布包平台相符的 profile，`--profile` 不相符时在停服前失败；已安装的 profile 不在新包中或与平台不符时重新选择（非交互时用包内建议），例如 profile 改名后的升级。profile 的 `x-build` 带有 `DEEPSTREAM_IMAGE` 时，它的 DeepStream 版本也必须与发布包一致。本项目的 `project.env` 不设 `DEEPSTREAM_IMAGE` 默认值，每个 profile 必须写明（缺少时渲染报错），所以安装时总会核对版本；框架仍接受 `project.env` 中的默认值，派生项目用它时，没有写明镜像的 profile 不核对版本。
- 渲染器派生 `DEEPSTREAM_VERSION`、`BASE_IMAGE`、`APPS_IMAGE`、挂载点、`HOST_UID/HOST_GID`、`CONFIG_PROFILE`、`COMPOSE_FILE`；`project.env` 中带有镜像名时（发布包安装）沿用包中的镜像名。
- `x-network-mode: <mode>` 给所有未显式声明 `network_mode` 的服务（包括框架服务与以后新增的单元）设置该值，写入 `generated/network.compose.yaml`；显式声明的（如 `deepstream-apps` 的 `none`）保持不变，声明了 `networks` 的服务报错。模式为 `host` 时同时给这些服务加 `extra_hosts`，把每个服务名指向 `127.0.0.1`，服务之间照常用服务名访问；声明了 `ports` 的服务用 `ports: !reset []` 清空映射（host 网络下映射无效，Docker 会告警 “Published ports are discarded”），这需要 Docker Compose 2.24.4+。只要有一个服务留在默认网络，Compose 就会创建 bridge 网络，所以不要在 `services` 中逐个设置。
- 先计算完整计划，再整体替换 `.env` 与 `generated/`；校验失败时旧文件保持不变。`project.env` 等 env 文件中开头的 `~` 在运行机器上展开；env 文件不执行 shell、不剥引号。例外：`site.env` 的值去掉一对外层引号（与 `site-fields.env` 中 `KEY=""` 的含义一致）；`.env` 由 Compose 按它自己的 dotenv 规则读取（未加引号时 `$` 会被插值、` #` 之后被当作注释、开头的引号必须闭合），渲染器写入时一律加引号，现场口令中的这些字符原样到达 Compose 与配置。

## 4. engine 预构建

`model-engine-builder` 运行 `tools/prepare_model_engines.py --models-path /models`：

- 扫描 `*.txt`，`[property]` 段同时含 `onnx-file` 与 `model-engine-file` 的就是待构建配置；输出路径完全遵循 `model-engine-file`（相对路径以配置目录为基准）。业务服务只读加载，nvinfer 不会把 engine 写到配置指定的路径，因此必须预构建。
- 计算能力先由 `nvidia-smi` 查询，不支持该查询时（例如部分 Jetson）经 `libcuda` 的驱动 API 读取，两者格式相同；都读不到时失败，`--compute-cap` 可显式指定。
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

默认安装到 `<根>/<项目名>`：根目录取 `project.env` 中的 `INSTALL_ROOT`（随发布包的 `release.env` 带到现场），为空时取环境变量 `XDG_DATA_HOME`，再为空时取 `~/.local/share`；`--install-dir` 直接指定安装目录。顺序不可调换：

1. 读取 `release.env`，核对本机平台与包完整性（缺失、改动、多余文件或符号链接都拒绝）。
2. 检查安装目标：安装目录与包不重叠；控制文件、发布路径不能是符号链接，也不能与已有文件/目录类型冲突。同名 Compose 工程已有由其他目录启动的容器（例如开发仓库的栈）时拒绝：在这里 `up` 会接管并重建它们。之后才读取旧的选择与答案。
3. 选择 profile（`--profile` > 已安装的选择 > 交互选择 > 包内建议），收集现场答案（升级沿用，新版本已删除的字段从答案中去掉并提示；`--ask` 逐项重新提问，默认值为已保存的答案，口令显示为 `***`、回车保留，可留空的字段输入 `CLEAR` 清空，没有终端时在停服前失败；`--clean-install` 重新收集；非交互时只保存已有答案；提问默认值含所选 profile 的 `x-build`，在 base 镜像中读取，需要时为此提前加载包内的 base）。检查清洁安装范围与退役文件冲突。
4. 加载镜像（不影响运行中的旧容器）；在临时目录渲染新配置。需要 PyYAML 的步骤（读取 profile 的 `x-build`、渲染）总在 base 镜像中离线运行，以安装用户身份、只挂载临时目录：目标机只需要 Python 标准库，渲染结果也不随目标机的 Python 与 PyYAML 版本变化。base 本来就是运行服务所需，包里不带 base 时它必须已在目标机上，否则在停服前失败；按最终 Compose 检查并准备可写 bind 挂载；核对每个服务镜像的平台；从新 apps 镜像取出载荷，离线、只读地执行所有带预检标签服务的 `--check-config`。
5. `--dry-run` 到此结束并打印计划。否则重验目标，用旧定义 `docker compose down --remove-orphans`，再只删除载荷卷 `<项目名>_app-files` 使其重新播种（不用 `down -v`：它会删除项目的全部命名卷，包括服务的持久数据；首次安装时删除遗留的同名载荷卷，例如删掉安装目录后重装，否则会继续运行旧载荷）。
6. 按需清空安装目录；删除未被修改的退役受管理文件；复制发布文件；写入 `project.env`（由 `release.env` 派生）、`site.env`、`.managed-files.json` 与预检过的配置快照（不再重新渲染）。
7. `docker compose config --quiet`、`up -d`，有界等待：常驻服务运行且健康，一次性服务成功退出。就绪默认不要求输入（`readiness.require_input`，见 [framework.md](framework.md)），相机没有推流时安装照常完成，并提示就绪文件中计数仍为 0 的服务（`note: <服务> has not received any input yet`）。等待期间每 30 秒报告仍在等待的服务及其状态。超时或服务提前退出时列出未就绪服务的状态与最近几条告警/错误日志（例如相机还没有推流时的 RTSP 404），以非零状态退出；此时安装已完成，服务继续运行并按重启策略恢复，安装器只是不再等待。等待中按 Ctrl-C 同样只结束等待。

规则：

- 可写 bind 源不得与安装目录或发布包互相包含，唯一例外是 `model-engine-builder` 写本次安装的 `models/`；`/data` 必须是可写 bind；显式 `create_host_path: false` 的源必须已存在；缺失目录由安装用户创建，失败时指出阻塞的路径层级。
- 普通升级保留现场新增的、与发布路径不冲突的文件；退役受管理文件被现场修改过时在停服前失败。
- 清洁安装拒绝根目录、家目录、与包重叠的目录及不像安装目录的目录；数据目录与镜像保留。
- 调用 shell 中的 `COMPOSE_FILE` 等选择变量被忽略，预检与执行看到同一份定义。
- 安装不是事务：停服后的失败不自动回滚，重新安装旧包即可恢复；旧镜像与外部数据保留。停服之后、启动之前的失败或中断（包括 Ctrl-C）会说明旧服务已停止、安装目录可能写了一半，以及如何完成（用同一个包重新安装）或退回（安装旧包）。
- 部署工具需要 Python 3.9+；版本过低时在启动时报错。

## 7. 平台与现场验证

profile 中写平台差异：`server-ds71` 为 x86_64 服务器（独立显卡，Docker bridge 网络），`jetson-ds71` 为 Jetson（host 网络）。两者都是 DeepStream 7.1。Apple Silicon 开发机用 `jetson-ds71` 构建并自检（同为 arm64 镜像，不运行推理）。新增平台时复制一份 profile：写明 `x-platform` 与上游镜像，在 `services` 中加入所需设备与 `group_add`（`runtime: nvidia` 由需要 GPU 的单元自己声明，不写在 profile），编码器差异写在应用配置的 `encoder.properties`。

### Jetson（host 网络）

`jetson-ds71` 让全部服务使用宿主网络，因此不依赖 Docker bridge：部分 Jetson 内核缺少 bridge 所需的模块（`lsmod | grep -E 'br_netfilter|iptable_nat'`、`docker network create` 可以确认）。这样的内核上 dockerd 需要关闭 bridge 与 iptables 才能启动：

```json
{
  "bridge": "none",
  "iptables": false,
  "runtimes": {"nvidia": {"path": "nvidia-container-runtime", "args": []}}
}
```

写入 `/etc/docker/daemon.json` 后 `sudo systemctl restart docker`；`runtimes` 一段保持 JetPack 原有内容。`docker build` 的 `RUN` 默认也走 bridge，因此镜像在 arm64 编译机上构建，用 `scripts/build-release.sh --with-base --profile jetson-ds71` 打包，Jetson 上只安装。x86 机器默认构建出 amd64 镜像，打包时因 `x-platform: arm64` 而拒绝。

`jetson-ds71` 用 `x-network-mode: host` 让全部服务使用宿主网络，并处理随之而来的差异：

- 服务名照常可用：Compose 的 DNS 在 host 网络下不生效，渲染器给每个服务加 `extra_hosts`，把全部服务名指向 `127.0.0.1`，所以配置中的 `redis`、`mediamtx` 与 `RTSP_URL` 的默认值不用按网络模式覆盖。真实相机照常填写 `RTSP_URL`。
- 监听直接暴露到现场网络：Redis 默认没有口令，只绑定 `127.0.0.1`（`REDIS_EXTRA_ARGS`）；MediaMTX 只保留 RTSP（8554/tcp、8000-8001/udp），推流仍需口令。`ports` 映射在 host 网络下不生效，也不再受 Docker 端口发布绕过 ufw 的影响，改由宿主防火墙控制。
- 端口与宿主共享：同一台机器不能再运行第二套实例，宿主上已有的 6379、8554 会冲突。
- RTSP over UDP、组播与 ONVIF 发现在 host 网络下可用。

镜像构建与自检不证明 GPU 推理可用。目标机仍需核对：真实 RTSP 输入、推理与跟踪结果、Redis 消息、完整 MP4 与停止时最后一段、RTSP 输出、engine 再次启动时复用、Linux 下的属主与设备权限（Jetson 上 `group_add: [video]` 按容器内的组名解析：确认编解码器等设备节点的属组与容器内 `video` 的 GID 一致，属于 `render` 等其他组的节点需要按 GID 加入）。Docker Desktop 上的结果不能替代 Linux 验证。
