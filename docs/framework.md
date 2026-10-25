# 框架使用指南

本文说明如何在框架上开发 DeepStream 应用与 Python 服务。构建、配置渲染与安装见 [deployment.md](deployment.md)，设计依据见 [design.md](design.md)。

## 1. 架构

```text
services/<应用>（策略：拓扑、业务逻辑、消息内容）
        │ 只依赖下面的公共接口
dsf::app      入口（CLI、日志、运行流程）与标准组件（就绪心跳、统计）
dsf::record   分段录像
dsf::publish  发布接口 + Redis Stream 实现
dsf::ds       块库、元数据模型、消息编码
dsf::ds_meta  元数据采集（需要 SDK）
dsf::gst      内省、图模型、Pipeline、Component
dsf::config   配置读取
```

| CMake 目标 | 依赖 SDK | 说明 |
|---|---|---|
| `dsf::framework` | 否 | 聚合 `dsf_app dsf_ds dsf_record dsf_publish dsf_gst dsf_config` |
| `dsf::ds_meta` | 是 | `MetaTap`、`DeepStreamVersion()` |
| `dsf::deepstream` | 是 | SDK 头文件与元数据库（插件使用） |

单元登记函数：`dsf_add_app`（默认需要 SDK，`NO_DEEPSTREAM` 用于纯 GStreamer 应用）、`dsf_add_plugin`、`dsf_add_python_service`、`dsf_add_test`、`dsf_add_python_test`。应用的配置与拓扑应放在不依赖 SDK 的库中，开发机上即可测试（参见 example-app 的 `example_app_core`）。

## 2. 应用入口

```cpp
struct MyApp {
  using Config = my::Config;
  static dsf::app::Info Describe();                        // 名称、说明、附加版本信息
  static Config Load(dsf::config::Reader& root);           // 读取并校验配置
  static void Build(const Config&, dsf::app::Assembly&);   // 组图、登记组件与发布器
};
int main(int argc, char** argv) { return dsf::app::Main<MyApp>(argc, argv); }
```

`Load` 与 `Build` 都不得有副作用：`--check-config` 与 `--dump-pipeline` 也会执行它们。框架负责：

| 选项 | 行为 |
|---|---|
| `--config PATH` | 默认 `$CONTAINER_CONFIG_PATH/<name>.yaml` |
| `--check-config` | 读取、校验、组图；不连接设备与服务，不写文件 |
| `--check-config --plugins` | 另外按实际插件校验元件、属性与取值（需要能加载插件的环境） |
| `--dump-pipeline` | 输出图（敏感值打码）与插件需求 |
| `--selftest` | 30 帧软件管线，不需要 GPU、相机与配置 |
| `--version` | 载荷版本（`/workspace/VERSION`）、框架版本、GStreamer 与 `Info::versions` |

退出码：0 成功，1 失败，2 用法错误。运行模式下的顺序：配置日志 → 插件能力检查 → 启动发布器 → 运行管线 → 停止发布器（排空）→ 汇总组件健康状态。组件 `Healthy()` 为 false（例如录像没有完整收尾）时以 1 退出。

配置中的 `logging` 段由框架读取：`level`（trace/debug/info/warn/error/critical/off）、`console`、`file`、`directory`、`stats_interval_s`（组件统计日志周期，0 关闭）。

`watchdog` 段也由框架读取：`stall_timeout_s`（默认 0 关闭）。为正时，就绪计数（`ReadyWhen` / `ReadyWhenFramesReach`）超过该时长没有增长，应用记录原因、排空停止并以 1 退出，由容器的重启策略恢复。Docker 只在进程退出时重启容器，`unhealthy` 不会触发重启：输入长时间断流、管线卡住时，没有看门狗的容器会一直停在 unhealthy。时限应明显长于源的正常重连时间；没有登记计数来源时主循环运行即算推进，看门狗不会触发。

## 3. 配置读取

读取代码本身就是 schema：每次读取登记一个路径，`Finish()` 拒绝没有被读取的键。

```cpp
auto sources = dsf::ds::ReadSources(root);                        // 列表
dsf::config::Reader redis = root.Section("redis");                // 映射
int port = redis.Get("port", 6379, dsf::config::Range(1, 65535)); // 类型由默认值决定
std::string password = redis.Secret("password");                  // 摘要中打码
auto props = root.Props("encoder.properties");                    // 自由映射，透传给插件
root.Fail("tracker.enabled", "requires inference");               // 跨字段规则
```

规则：拒绝重复键、带点号的键、错误类型（引号中的 `"123"` 不是整数）、非有限浮点；null 视为缺失。约束：`OneOf`、`NonEmpty`、`PathComponent`、`AtLeast`、`Range`；列表用 `List(key, min_items)`。错误消息为 `<文件>: <完整路径>: <原因>`。

测试中可用 `Document::LeafPaths()` 与 `Reader::DeclaredPaths()` 验证模板与读取代码一致（见 example-app 的测试）。

## 4. 组图

### 4.1 图模型

```cpp
dsf::gst::Graph& g = assembly.graph();
auto src = g.Add("videotestsrc", "src").Set("is-live", true);
auto caps = g.Caps("video/x-raw,width=640,height=480");
g.Chain({src, caps, g.Add("fakesink", "sink")});
g.Link(out, mux.Pad("sink_0"));           // 指定 pad：静态、request（sink_0）或模板（sink_%u）
auto tee = g.Tee(out);  g.Branch(tee);    // tee 分支自动带 queue
element.SetSecret("uri", url);            // 诊断输出中打码
element.Apply(spec.properties);           // 配置透传，覆盖块设置的同名属性
```

sometimes pad（nvurisrcbin、decodebin）在 pad 出现时自动连接。属性值以文本保存，实例化时按插件的 `GParamSpec` 转换（数值、布尔、字符串、枚举名或值、flags、caps）并校验范围；错误消息带配置路径。同一份图同时产生 `Requirements()`（启动前的插件能力检查）与 `Describe()`（`--dump-pipeline`）。需要完全手写时可用 `Pipeline(name, builder)` 或 `Pipeline::Launch(description)`。

### 4.2 块库（`dsf/ds/blocks.h`）

| 块 | 配置读取 | 说明 |
|---|---|---|
| `AddSource` | `SourceSpec::Read` / `ReadSources` | `rtsp` `file` `uri` `v4l2` `test`，输出 NVMM NV12，可缩放 |
| `AddMux` / `LinkToMux` | `MuxSpec::Read` | nvstreammux；第 i 路接 `sink_i`，i 即元数据中的 `pad_index` |
| `AddInfer` | `InferSpec::Read` / `ReadInferences` | nvinfer 或 nvinferserver，可串接多级 |
| `AddTracker` | `TrackerSpec::Read` | nvtracker |
| `AddAnalytics` / `AddPreprocess` | `ConfigFileSpec::Read` | nvdsanalytics / nvdspreprocess |
| `AddMetaTap` | `ImageSpec::Read` | 元数据采集点（identity）；启用图片编码时前面加 queue |
| `AddOsd` | `OsdSpec::Read` | nvdsosd（前后各一次格式转换） |
| `AddTiler` / `AddDemux` | `TilerSpec::Read` | 多路拼接 / 按源拆分（`demux.Pad("src_i")`） |
| `AddEncoder` / `AddParser` | `EncoderSpec::Read` | 硬件编码；编码器属性在 Jetson 与 dGPU 不同，用 `properties` |
| `AddRtspOut` / `AddFakeSink` | — | rtspclientsink / fakesink |
| `record::AddSegmentSink` | — | parser + splitmuxsink(mp4mux) |

每个块的配置都可以加 `properties:`，设置对应插件的任意属性；不需要时省略即可（缺省等于空）。源配置示例：

```yaml
sources:
  - id: gate-1                 # 合法路径段；消息 source_id 与录像目录名
    type: rtsp                 # rtsp | file | uri | v4l2 | test
    uri: rtsp://admin@10.0.0.5/stream1
    rtsp: {password: "...", latency_ms: 200, tcp: true, reconnect_interval_s: 5}
    file: {loop: true}         # type: file
    capture: {device: /dev/video0, format: UYVY, width: 1920, height: 1080, framerate: 30}  # v4l2/test
    output: {width: 1920, height: 1080}   # 规范化尺寸，0 保持原尺寸
    properties: {drop-on-latency: true}   # 可选，透传给 nvurisrcbin / v4l2src / videotestsrc
```

`rtsp`/`file`/`uri` 源默认设置 nvurisrcbin 的 `num-extra-surfaces: 8`：解码器的输出 surface 池很小，源侧转换走零拷贝直通时，下游硬件编码器持有的就是解码器的 surface，池耗尽后管线静默卡住。加 queue 解决不了（queue 只会多持有 surface）。下游持有帧更多的拓扑（多路编码、深队列）可在 `properties` 中调大。

新版 nvstreammux（`USE_NEW_NVSTREAMMUX=yes`）没有 width/height 属性，`MuxSpec` 的尺寸会在能力检查时报错；使用新版时请用 `properties` 设置 `config-file-path` 并自行处理尺寸。

## 5. 组件与生命周期

```cpp
class Component {
  virtual std::string_view Name() const = 0;
  virtual void Require(Requirements&) const;           // 图之外的能力需求（例如信号）
  virtual void OnCreated(Pipeline&, GstElement* bin);  // 取元件、挂探针、连信号
  virtual void OnPlaying(Pipeline&);                   // 真正进入 PLAYING 后一次
  virtual void OnMessage(GstMessage*);                 // 每条总线消息
  virtual void OnStopping();                           // 首次停止请求、发送 EOS 之前
  virtual void OnReleased();                           // 已置 NULL、流线程已停：拆探针
  virtual bool Healthy() const;                        // false 时应用以失败退出
  virtual void Report(Stats&) const;                   // 周期统计，可能在进入 PLAYING 前调用
};
```

- 全部回调在控制线程；OnCreated 按登记顺序，OnReleased 逆序且总会调用。某个组件抛异常不影响其他组件清理，并使结果失败。
- 主循环默认运行在 GLib 默认 context 上：插件的内部定时器（例如 nvurisrcbin 的断流重连）依赖它。在多个线程中并行运行多条管线时，次要管线设置 `Options::private_context`。
- `Pipeline::Post` 把任务排队到控制线程；`Every` 返回 `Timer`（析构即取消）；`RequestStop` 可在任意线程调用。
- 停止：首次请求通知 OnStopping 后发送 EOS 排空（默认 5 秒）。排空完成的判据是管线 EOS，或者 EOS 已到达每个 sink 的输入且所有组件 `Drained()`——网络 sink（如 rtspclientsink）收到 EOS 后不一定结束，不会因此拖住停止，而录像器会等最后一个分段封装完成。超时或 `kImmediate` 强制结束；排空中再收到 SIGINT/SIGTERM 立即结束。排空时限可在 `assembly.options()` 调整，须小于容器的停止宽限期。
- 运行期错误：默认立即以 `run-error` 结束。`assembly.options().drain_on_error = true` 时，PLAYING 或排空中的总线错误先排空再结束，结果仍是 `run-error`（第一条错误，排空未完成时附上原因），让其余分支收尾——例如 rtspclientsink 断线（它报错且不重连）时录像器仍封装当前分段，进程退出后由重启策略恢复推流。出错元件若向上游返回流错误，共用 tee 的其他分支也会停住，只能依靠组件自己的兜底（录像器的 `eos_fallback`）或在 `stop_timeout` 后强制结束。启动阶段的错误仍立即结束。
- 启动：`assembly.options().startup_timeout` 默认 30 秒，设为 0 禁用启动超时，负数视为启动失败。异步切换状态的 sink 可能等待首帧，使按需产帧的管线长期无法进入 PLAYING；是否等待取决于元件及其配置。状态切换返回失败或总线报告错误时仍以 `start-error` / `run-error` 结束，`OnPlaying` 只在真正进入 PLAYING 后调用。计时在 `set_state(PLAYING)` 返回异步状态切换结果后开始，不覆盖同步初始化和阻塞调用；定时任务须等主循环运行才能执行。
- 结果原因：`eos`、`stopped`（成功）；`forced`、`build-error`、`start-error`、`run-error`、`startup-timeout`（失败）。

标准组件由框架自动加入：`ReadinessHeartbeat` 与 `StatsReporter`，两者在 `OnCreated` 注册定时器，由主循环按各自周期调度，不等 PLAYING。指定进度指标时，就绪文件 `$DSF_READY_FILE` 只在处理计数增长时更新：DeepStream 管线用 `assembly.ReadyWhen([&tap] { return tap.frames(); })` 按批次元数据中的真实帧计数——mux 之后一个 buffer 是一个批次，多路时某一路断流批次仍会推送，按 buffer 计数不能反映输入是否在供帧；不经过 nvstreammux 的普通管线可用 `ReadyWhenFramesReach(元件)`；都不指定时只要主循环在运行就每秒更新，按需产帧的源配合 `startup_timeout = 0` 使用，静止期间仍保持就绪。组件的 `Report()` 须能在进入 PLAYING 前报告统计。

## 6. 元数据（需要 SDK）

```cpp
auto& tap = assembly.Add<dsf::ds::MetaTap>(options, sources);   // options.element = AddMetaTap 的名字
tap.OnBatch([&](const dsf::ds::BatchRecord& batch) { ... });    // 媒体线程，必须快
tap.OnBatchAsync([&](const dsf::ds::BatchRecord& batch) { ... }, 64);  // 工作线程，队列满时丢弃并计数
tap.AddObjectExtractor([](const NvDsObjectMeta& meta, dsf::ds::ObjectRecord& out) {
  out.extra["my_field"] = ...;                                  // 自定义 user meta
});
```

图片编码（可选，默认关闭）：

```cpp
options.image = dsf::ds::ImageSpec::Read(reader.Section("image"));   // enabled quality width height properties
options.gpu_id = mux.gpu_id;
dsf::ds::AddMetaTap(graph, tail, "meta", options.image);             // 同一个 ImageSpec：在采集点前插入 queue
tap.SetImageFilter([](const dsf::ds::FrameRecord& f) { return !f.objects.empty(); });  // 可选，默认即此规则
```

选中的帧在转换之后、处理函数之前用 `nvds_obj_enc` 整帧编码为 JPEG（Jetson 为 NVJPG 硬件，dGPU 为 nvJPEG/CUDA），直接读取 GPU 上的批次 surface，结果复制进 `FrameRecord::image`（`format` `width` `height` `bytes`）。采集点位于 OSD 之前时图片不含框，尺寸默认等于 mux 输出尺寸，与 bbox 坐标一致；`width/height` 设置后缩放，bbox 需按比例换算。编码在采集点的流线程中同步等待 GPU 完成，queue 使它与上游推理并行：吞吐要求每批编码时间小于帧间隔，延时增加一次编码时间。统计 `<element>_images`、`_image_failures`、`_encode_us_avg`；编码失败的帧照常交给处理函数，`image` 为空。

`BatchRecord → FrameRecord → ObjectRecord` 是纯数据：`source_id`（由 `SourceRegistry` 按 `pad_index` 映射）、`frame_num`、`pts_ns`、`ntp_ms`、`wall_ms`、帧尺寸（mux 输出尺寸，即框坐标空间）、目标的类别、标签、置信度、框、`track_id`（仅被跟踪时有值）、二级分类结果 `classes`、`extra`。`options.analytics = true` 时采集 nvdsanalytics 的帧级计数与目标状态。

## 7. 消息

`dsf/ds/message.h` 提供 schema v1 编码，应用也可以完全使用自己的格式。共同字段：`schema_version`、`event_type`、`source_id`、`timestamp_ms`。

| event_type | 字段 |
|---|---|
| `detection` | `frame_num` `pts_ns` `ntp_ms`(有值时) `frame_width` `frame_height` `objects[]`：`class_id` `label` `confidence`(三位小数) `bbox{x,y,width,height}`（裁剪到帧内后取整，零面积丢弃）`track_id`(被跟踪时) `classes[]`(有值时) `extra`(有值时)；帧级 `extra`；`image{format,width,height,size}`(帧带图片时，只是描述) |
| `recording_started` | `branch` `segment_id` `start_ms` |
| `recording_finished` | 另加 `end_ms` `duration_ms`（主机墙钟差）`detected` `path`（已封装完成、可直接读取的文件） |

发布：

```cpp
auto& redis = assembly.AddPublisher(dsf::publish::MakeRedisStreamPublisher(options));
dsf::publish::Channel& channel = redis.Open({"project:detections", 256, 10000});
channel.TryPublish(dsf::ds::Serialize(dsf::ds::DetectionMessage(frame)));   // 不阻塞，单字段 payload
channel.TryPublish(dsf::ds::DetectionEntry(frame));   // payload + 帧带图片时的 image 字段（JPEG 原始字节）
```

一个连接、一个后台线程、每个通道独立的有界队列；消息入队后立即发送 `XADD <stream> MAXLEN ~ <n> * <field> <value> ...`（字段值二进制安全，图片不做 base64），有积压时每轮最多 `batch_size` 条、`batch_bytes` 字节（至少一条）以 pipelining 合并发送。`ChannelSpec::max_bytes` 限制队列积压的总字节数（0 不限，空队列总能放下一条），带图片的通道应设置。`MAXLEN ~` 按 Stream 内部节点（默认 100 条）裁剪，实际长度可能略多于设定值。

带图片的检测记录读取方式（redis-py，`decode_responses=False`）：

```python
event = json.loads(fields[b"payload"])
jpeg = fields.get(b"image")        # bytes；该帧没有图片时为 None
```

图片随消息走内存：Redis 应与应用在同一主机（Compose 网络内），Stream 的 `MAXLEN` 只用来吸收消费端的短暂抖动（例如 100），不作为存储；消费端读取后自行存盘或上传，慢操作放到读取循环之外。消费端应使用 `XREAD`/`XREADGROUP` 的 `BLOCK` 阻塞读取。语义是至多一次：队列满、断线超过缓冲能力、停止时排空超时都会丢弃并计数；不落盘、不补发。Redis 不可用时每秒重连，消息先在队列中缓冲。需要可靠投递时实现 `Publisher` 接口或在图内使用 nvmsgbroker。

## 8. 分段录像

```cpp
auto sink = dsf::record::AddSegmentSink(graph, encoded, "h264", 60, "rec_0");
dsf::record::RecorderOptions options{root, source_id, "osd", "rec_0"};
options.retention = {1440, 60, 5ull << 30};  // max_files keep_files min_free_bytes；默认全 0 不清理
auto& recorder = assembly.Add<dsf::record::SegmentRecorder>(options, events);
recorder.MarkDetected();   // 任意线程：当前分段出现过目标
```

文件为 `<root>/<source_id>/<branch>/<source_id>-<branch>-<起始本地时间>-<序号>[-detected].mp4`，起始本地时间形如 `20261005T163012.345`（本地时间与毫秒，不带时区标记，时区取自 `/etc/localtime`），程序应使用事件中的 `start_ms` 与 `path`，不解析文件名。写入期间是 `.mp4.tmp`，收到 `splitmuxsink-fragment-closed` 并改名成功后才触发 `finished`；分段内调用过 `MarkDetected()` 时最终文件名带 `-detected`（`segment_id` 不变，`path` 为实际路径）。启动时删除本目录遗留的 `.tmp`。

保留策略 `RecorderOptions::retention` 按录像器（每个源的每个分支）独立计数，在启动时与每个分段收尾后执行，均在控制线程：已收尾分段按修改时间从新到旧，最新 `keep_files` 个始终保留；总数超过 `max_files` 时在其余分段中先删最旧的无目标分段，再删最旧的有目标分段；文件系统剩余空间低于 `min_free_bytes` 时按同样顺序继续删除，并可删入保护区（记 warning）。只处理符合本录像器命名的 `.mp4`，不动 `.tmp` 与其他文件。`max_files` 非 0 时须不小于 `keep_files`，否则构造时抛异常。统计 `<element>_pruned`；删除失败计入 `_failures` 并使 `Healthy()` 为 false。重启后有无目标从文件名还原。

停止时的顺序很重要：`OnStopping` **不**直接向录像入口发送 EOS，而是让管线注入的 EOS 跟在最后一帧后面流经编码器与队列，全部到达后才封装（判据是 `fragment-closed`）。提前在录像入口发 EOS 会让分段立即封装，编码器内部与分支队列中仍在处理的帧随后到达时被丢弃。若 EOS 在 `RecorderOptions::eos_fallback`（默认 3 秒，须小于管线的 `stop_timeout`）内仍未到达录像入口（某个上游元件不传递 EOS），才会直接向录像入口发送 EOS 作为兜底。

## 9. Python 服务

```python
from dsf.config import Reader, at_least
from dsf.service import Service

def load(root: Reader):                      # 读取并校验；不得有副作用
    return root.section("redis").get("port", 6379, at_least(1))

def main(port, service: Service):
    while service.running:
        ...                                  # 业务处理
        service.sleep(1)                     # 可被停止信号打断；同时更新就绪文件

sys.exit(Service("my-service", "说明").run(load, main))
```

`Service` 负责 `--config`、`--check-config`、`--version`、`logging.level`、`watchdog.stall_timeout_s`、SIGINT/SIGTERM 与就绪文件。看门狗为正时，主循环超过时限没有调用 `sleep()` / `alive()`，进程记录原因后直接以 1 退出（卡住的主循环无法自行排空）。`Reader` 的规则与 C++ 一致（`get` / `required` / `secret` / `section` / `items` / `finish`）：未加引号的标量按读取时要求的类型解析，`password: 123456` 读成字符串即为 `"123456"`；加引号的标量只能读成字符串。第三方依赖写进单元的 `requirements.txt`。

## 10. 新增应用

1. `scripts/new-app.sh <name> --kind cpp|python` 生成 `services/<name>/`。
2. 在 `src/app.cpp`（或 `main.py`）中读取配置、组图、写业务逻辑；配置模板放 `templates/config/<name>.yaml`，现场参数加到 `site-fields.env`。
3. 平台差异（`runtime: nvidia`、设备、`group_add`）写在 `profiles/*.yaml` 的 `services.<name>`。
4. 需要自定义推理解析库时在 `plugins/` 下用 `dsf_add_plugin`；模型放 `models/<id>/`，`model-engine-builder` 会自动构建其 engine。
5. `scripts/apply-profile.sh && docker compose config` 检查部署定义，`docker compose build deepstream-apps` 完成编译、CTest 与自检（宿主 `cmake --build build && ctest` 可用于更快的迭代）。

## 11. 测试分层

| 层 | 内容 | 环境 |
|---|---|---|
| 框架单元/集成 | 配置、图、内省、Pipeline、块结构、消息、软件录像、Redis 投递、入口 | 开发机（GStreamer 软件插件、redis-server 可选） |
| 应用核心 | 模板与读取代码一致、拓扑结构、跨字段规则 | 开发机 |
| SDK 层 | 应用与插件编译链接、CLI 冒烟测试 | builder 镜像 |
| 部署 | 渲染、打包、安装（假 Docker）、engine 工具（假 trtexec）、仓库约定 | 开发机（docker CLI 可选） |
| 镜像 | builder 中的 CTest（以上各层全部执行，`DSF_FORBID_SKIPS=1` 使 SKIP 计为失败），每个 profile 的渲染与 Compose 解析，原始 base 中的非 root 无网络自检 | 有 Docker 的开发机或编译服务器 |
| 目标机 | 真实视频、推理与跟踪、Redis 消息、MP4 收尾、RTSP 输出、engine 复用、Linux 权限 | 目标设备 |

宿主缺少 redis-server、软件编码插件或 docker CLI 时相关用例显示 SKIP 并成功退出，查看详细输出确认实际覆盖范围；镜像构建是必须通过的完整验证。C++ 的 `dsf_test.h` 与 Python 的 `run_unittest.py`（`dsf_add_python_test` 的入口）都读取 `DSF_FORBID_SKIPS`。
