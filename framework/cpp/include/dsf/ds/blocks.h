// DeepStream 块库：可复用的子图构造器。
//
// 每个块由一个 Spec（配置读取 + 校验）和一个 Add 函数组成：Add 把块接在 input 之后，
// 返回块的输出元件，下一个块接在它后面。
// 块给出经过验证的默认值，配置中的 properties: 可以覆盖或补充任意插件属性。
// 块只引用工厂名与属性名，不包含 SDK 头文件，组图可在没有 SDK 的开发机上测试。
//
// 各 Spec::Read 约定的配置结构见 docs/framework.md。

#ifndef DSF_DS_BLOCKS_H
#define DSF_DS_BLOCKS_H

#include <string>
#include <vector>

#include "dsf/config/config.h"
#include "dsf/gst/graph.h"

namespace dsf::ds {

using config::Properties;
using config::Reader;
using gst::Element;
using gst::Graph;

// ---- 视频源 ----

enum class SourceType { kUri, kRtsp, kFile, kV4l2, kTest };

// 配置中的类型名：uri | rtsp | file | v4l2 | test。
const char* ToString(SourceType type);

// 一路源。子段与配置一一对应，只读取与 type 相关的段：rtsp 只用于 rtsp，file 只用于 file，
// capture 只用于 v4l2 与 test（其中 device 只用于 v4l2），uri 不用于 v4l2 与 test；
// 写了无关的段说明配置与类型不符，读取时报错而不是悄悄忽略。
struct SourceSpec {
  struct Rtsp {
    std::string password;  // 单独配置，编码后注入到 uri 的用户名之后，不出现在日志与诊断中
    int latency_ms = 200;
    bool tcp = true;
    int reconnect_interval_s = 5;
  };
  struct File {
    bool loop = true;
  };
  // 原始画面参数；0 或空表示由协商决定（test 读取时默认 1280x720@25）。
  struct Capture {
    std::string device = "/dev/video0";  // 只用于 v4l2
    std::string format;
    int width = 0;
    int height = 0;
    int framerate = 0;
  };

  std::string id;  // 源标识：消息、录像目录与日志中使用，必须是合法路径段
  SourceType type = SourceType::kUri;
  std::string uri;   // uri/rtsp/file：nvurisrcbin 的 uri；file 也可写绝对路径
  bool live = true;  // 由 type 推导：file 与 uri(file://) 为 false
  Rtsp rtsp;
  File file;
  Capture capture;
  // 规范化输出尺寸；0 表示保持源尺寸
  int output_width = 0;
  int output_height = 0;
  Properties properties;  // 透传给源元件（nvurisrcbin / v4l2src / videotestsrc）

  static SourceSpec Read(Reader reader);
};

// 读取 sources 列表，检查 id 唯一。
std::vector<SourceSpec> ReadSources(Reader reader, const std::string& key = "sources");

// 返回 rtsp://user:password@host/... 形式的地址。口令按百分号编码（GStreamer 解析地址时解码），
// 可以含任意字符；字母、数字与 -._~ 保持原样。password 为空时原样返回；write_empty 为 true 时
// 空口令也写出 "user:"（向要求带凭据、但不校验口令的服务端推流时需要）。
// 口令非空而地址没有用户名或已含口令时抛 std::invalid_argument。
std::string InjectPassword(const std::string& uri, const std::string& password, bool write_empty = false);

// 添加一路源并规范为 NVMM NV12（可选缩放）。name 是源元件名（例如 "src0"），块内其余元件以它为前缀。
Element AddSource(Graph& graph, const SourceSpec& spec, const std::string& name);

// ---- 批处理 ----

struct MuxSpec {
  int width = 1920;   // 输出尺寸：元数据坐标空间
  int height = 1080;
  int batched_push_timeout_us = 40000;
  int gpu_id = 0;
  Properties properties;

  static MuxSpec Read(Reader reader);
};

// nvstreammux，batch-size 等于 batch。live 时设置 live-source，避免某路断流时整条管线等待。
Element AddMux(Graph& graph, const MuxSpec& spec, int batch, bool live,
               const std::string& name = "mux");

// 第 index 路接入 mux 的 sink_<index>；index 即元数据中的 pad_index / source_id 编号。
void LinkToMux(Graph& graph, const gst::Endpoint& from, const Element& mux, int index);

// ---- 推理与分析 ----

struct InferSpec {
  std::string name;                // 元件名，也是日志与消息中的推理单元标识
  std::string plugin = "nvinfer";  // nvinfer | nvinferserver
  std::string config_file;
  Properties properties;

  static InferSpec Read(Reader reader);
};

std::vector<InferSpec> ReadInferences(Reader reader, const std::string& key = "inference");
Element AddInfer(Graph& graph, const gst::Endpoint& input, const InferSpec& spec);

struct TrackerSpec {
  bool enabled = false;
  std::string ll_lib_file;
  std::string ll_config_file;
  int width = 640;
  int height = 384;
  Properties properties;

  static TrackerSpec Read(Reader reader);
};

Element AddTracker(Graph& graph, const gst::Endpoint& input, const TrackerSpec& spec,
                   const std::string& name = "tracker");

// nvdsanalytics、nvdspreprocess 等只需配置文件的插件。
struct ConfigFileSpec {
  bool enabled = false;
  std::string config_file;
  Properties properties;

  static ConfigFileSpec Read(Reader reader);
};

Element AddAnalytics(Graph& graph, const gst::Endpoint& input, const ConfigFileSpec& spec,
                     const std::string& name = "analytics");
Element AddPreprocess(Graph& graph, const gst::Endpoint& input, const ConfigFileSpec& spec,
                      const std::string& name = "preprocess");

// 采集点上的整帧图片编码（MetaTap::Options::image）。编码数据随 FrameRecord::image 交给处理函数。
struct ImageSpec {
  bool enabled = false;
  int quality = 85;  // JPEG 质量 1–100
  int width = 0;     // 编码输出尺寸，0 保持帧尺寸（mux 输出尺寸，即框坐标空间）
  int height = 0;
  Properties properties;  // 透传给采集点前的 queue

  static ImageSpec Read(Reader reader);
};

// 元数据采集点：identity，探针挂在它的 src pad 上。image.enabled 时在前面插入 queue（<name>_queue）：
// 编码在采集点的流线程中同步完成，queue 让它与上游的推理并行，而不是串行累加到每个批次上。
Element AddMetaTap(Graph& graph, const gst::Endpoint& input, const std::string& name,
                   const ImageSpec& image = {});

// ---- 画面输出 ----

struct OsdSpec {
  bool enabled = true;
  Properties properties;
  static OsdSpec Read(Reader reader);
};

// nvvideoconvert ! nvdsosd ! nvvideoconvert，返回最后一个转换器。
Element AddOsd(Graph& graph, const gst::Endpoint& input, const OsdSpec& spec,
               const std::string& name = "osd");

struct TilerSpec {
  int rows = 0;     // 0 表示按源数自动计算
  int columns = 0;
  int width = 1920;
  int height = 1080;
  Properties properties;
  static TilerSpec Read(Reader reader);
};

// sources 为输入的源数，必须为正；rows/columns 为 0 时按它计算。
Element AddTiler(Graph& graph, const gst::Endpoint& input, const TilerSpec& spec, int sources,
                 const std::string& name = "tiler");

// nvstreamdemux；第 i 路输出用 demux.Pad("src_<i>")。
Element AddDemux(Graph& graph, const gst::Endpoint& input, const std::string& name = "demux");

struct EncoderSpec {
  std::string codec = "h264";  // h264 | h265
  int bitrate = 4000000;
  int iframe_interval = 30;  // 关键帧间隔（帧）；关键帧都是 IDR，每个都能独立起播
  // 输入时间戳相邻两帧相差超过该值（直播源断流恢复）时请求关键帧，由应用为每个编码器登记
  // app::KeyframeOnGap 实现；块本身只组图。0 表示不请求。
  int keyframe_on_gap_ms = 1000;
  Properties properties;  // 编码器属性在 Jetson 与 dGPU 上不同，平台差异写在这里
  static EncoderSpec Read(Reader reader);
};

// nvvideoconvert ! nvv4l2h26xenc，返回编码器。
Element AddEncoder(Graph& graph, const gst::Endpoint& input, const EncoderSpec& spec,
                   const std::string& name);

// h264parse / h265parse；config_interval=-1 时在每个 IDR 前补参数集（RTSP 需要）。
Element AddParser(Graph& graph, const gst::Endpoint& input, const std::string& codec,
                  const std::string& name, bool repeat_parameter_sets);

// parser ! rtspclientsink（rtspclientsink 自行选择 payloader）。默认以 TCP 推流：
// 容器网络中 UDP 推流会丢包；properties 透传给 rtspclientsink。
Element AddRtspOut(Graph& graph, const gst::Endpoint& encoded, const std::string& codec,
                   const std::string& url, const std::string& name,
                   const Properties& properties = {});

Element AddFakeSink(Graph& graph, const gst::Endpoint& input, const std::string& name);

}  // namespace dsf::ds

#endif  // DSF_DS_BLOCKS_H
