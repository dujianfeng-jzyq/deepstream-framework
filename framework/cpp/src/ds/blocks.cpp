// 块库的实现。

#include "dsf/ds/blocks.h"

#include <cmath>
#include <set>
#include <stdexcept>

namespace dsf::ds {
namespace {

bool StartsWith(const std::string& text, const std::string& prefix) {
  return text.compare(0, prefix.size(), prefix) == 0;
}

void ReadOutputSize(Reader reader, int* width, int* height) {
  *width = reader.Get("width", 0, config::AtLeast(0));
  *height = reader.Get("height", 0, config::AtLeast(0));
  if ((*width == 0) != (*height == 0)) {
    reader.Fail("", "width and height must be set together (0 keeps the source size)");
  }
}

std::string NvmmCaps(int width, int height) {
  std::string caps = "video/x-raw(memory:NVMM),format=NV12";
  if (width > 0 && height > 0) {
    caps += ",width=" + std::to_string(width) + ",height=" + std::to_string(height);
  }
  return caps;
}

std::string RawCaps(const std::string& format, int width, int height, int framerate) {
  std::string caps = "video/x-raw";
  if (!format.empty()) caps += ",format=" + format;
  if (width > 0 && height > 0) {
    caps += ",width=" + std::to_string(width) + ",height=" + std::to_string(height);
  }
  if (framerate > 0) caps += ",framerate=" + std::to_string(framerate) + "/1";
  return caps;
}

}  // namespace

// ---- 源 ----

SourceSpec SourceSpec::Read(Reader reader) {
  SourceSpec spec;
  spec.id = reader.Required<std::string>("id", config::PathComponent());
  spec.type = reader.Get<std::string>("type", "uri",
                                      config::OneOf({"uri", "rtsp", "file", "v4l2", "test"}));
  spec.uri = reader.Get<std::string>("uri", "");
  if ((spec.type == "uri" || spec.type == "rtsp" || spec.type == "file") && spec.uri.empty()) {
    reader.Fail("uri", "is required for source type " + spec.type);
  }

  Reader rtsp = reader.Section("rtsp");
  spec.password = rtsp.Secret("password");
  spec.latency_ms = rtsp.Get("latency_ms", 200, config::AtLeast(0));
  spec.tcp = rtsp.Get("tcp", true);
  spec.reconnect_interval_s = rtsp.Get("reconnect_interval_s", 5, config::AtLeast(0));
  if (spec.type == "rtsp") {
    if (!StartsWith(spec.uri, "rtsp://") && !StartsWith(spec.uri, "rtsps://")) {
      reader.Fail("uri", "must start with rtsp:// or rtsps://");
    }
    try {
      (void)InjectPassword(spec.uri, spec.password);
    } catch (const std::invalid_argument& error) {
      reader.Fail("rtsp.password", error.what());
    }
  } else if (!spec.password.empty()) {
    reader.Fail("rtsp.password", "is only used by rtsp sources");
  }

  spec.loop = reader.Section("file").Get("loop", true);

  Reader capture = reader.Section("capture");  // v4l2 与 test 的原始画面参数
  spec.device = capture.Get<std::string>("device", "/dev/video0");
  spec.raw_format = capture.Get<std::string>("format", "");
  spec.width = capture.Get("width", spec.type == "test" ? 1280 : 0, config::AtLeast(0));
  spec.height = capture.Get("height", spec.type == "test" ? 720 : 0, config::AtLeast(0));
  spec.framerate = capture.Get("framerate", spec.type == "test" ? 25 : 0, config::AtLeast(0));

  ReadOutputSize(reader.Section("output"), &spec.output_width, &spec.output_height);
  spec.properties = reader.Props("properties");

  spec.live = !(spec.type == "file" || (spec.type == "uri" && StartsWith(spec.uri, "file://")));
  return spec;
}

std::vector<SourceSpec> ReadSources(Reader& reader, const std::string& key) {
  std::vector<SourceSpec> sources;
  std::set<std::string> ids;
  for (Reader item : reader.List(key, 1)) {
    SourceSpec spec = SourceSpec::Read(item);
    if (!ids.insert(spec.id).second) item.Fail("id", "duplicate source id '" + spec.id + "'");
    sources.push_back(std::move(spec));
  }
  return sources;
}

std::string InjectPassword(const std::string& uri, const std::string& password) {
  if (password.empty()) return uri;
  if (password.find_first_of("@/") != std::string::npos) {
    throw std::invalid_argument("must not contain '@' or '/'");
  }
  const std::size_t scheme = uri.find("://");
  const std::size_t at = scheme == std::string::npos ? scheme : uri.find('@', scheme + 3);
  const std::size_t slash = scheme == std::string::npos ? scheme : uri.find('/', scheme + 3);
  if (at == std::string::npos || (slash != std::string::npos && slash < at)) {
    throw std::invalid_argument("is set but the uri has no user name (expected rtsp://user@host/...)");
  }
  if (uri.substr(scheme + 3, at - scheme - 3).find(':') != std::string::npos) {
    throw std::invalid_argument("is set but the uri already contains a password");
  }
  return uri.substr(0, at) + ":" + password + uri.substr(at);
}

Element AddSource(Graph& graph, const SourceSpec& spec, const std::string& prefix) {
  Element head;
  Element tail;
  if (spec.type == "v4l2" || spec.type == "test") {
    if (spec.type == "v4l2") {
      head = graph.Add("v4l2src", prefix);
      head.Set("device", spec.device);
    } else {
      head = graph.Add("videotestsrc", prefix);
      head.Set("is-live", true);
    }
    head.Apply(spec.properties);
    tail = graph.Caps(RawCaps(spec.raw_format, spec.width, spec.height, spec.framerate),
                      prefix + "_raw");
    graph.Link(head, tail);
  } else {
    head = graph.Add("nvurisrcbin", prefix);
    // 解码器的输出 surface 池很小（最少解码所需 + 1），硬件编码器会持有输入帧直到编码完成；
    // 当源侧转换走零拷贝直通时，编码器持有的就是解码器的 surface，池被耗尽后管线停在第 3 帧左右
    // 且不报错（无 B 帧码流必现）。加 queue 无济于事：queue 只会多持有 surface。留出余量。
    head.Set("num-extra-surfaces", 8);
    std::string uri = spec.uri;
    if (spec.type == "file" && StartsWith(uri, "/")) uri = "file://" + uri;
    if (spec.type == "rtsp") {
      // 口令只在实例化时注入；诊断输出中整个地址打码。
      head.SetSecret("uri", InjectPassword(uri, spec.password));
      head.Set("latency", spec.latency_ms);
      head.Set("rtsp-reconnect-interval", spec.reconnect_interval_s);
      head.Set("rtsp-reconnect-attempts", -1);  // 现场断流是常态：一直重连，由监控发现长时间断流
      if (spec.tcp) head.Set("select-rtp-protocol", 4);  // GstRTSPLowerTrans: TCP
    } else {
      head.Set("uri", uri);
      if (spec.type == "file") head.Set("file-loop", spec.loop);
    }
    head.Apply(spec.properties);
    tail = head;
  }
  // 下游统一为 NVMM NV12：mux 之后的处理不再关心输入是哪种源。
  Element convert = graph.Add("nvvideoconvert", prefix + "_conv");
  Element caps = graph.Caps(NvmmCaps(spec.output_width, spec.output_height), prefix + "_nvmm");
  graph.Link(tail, convert);
  graph.Link(convert, caps);
  return caps;
}

// ---- 批处理 ----

MuxSpec MuxSpec::Read(Reader reader) {
  MuxSpec spec;
  spec.width = reader.Get("width", 1920, config::AtLeast(1));
  spec.height = reader.Get("height", 1080, config::AtLeast(1));
  spec.batched_push_timeout_us = reader.Get("batched_push_timeout_us", 40000, config::AtLeast(1));
  spec.gpu_id = reader.Get("gpu_id", 0, config::AtLeast(0));
  spec.properties = reader.Props("properties");
  return spec;
}

Element AddMux(Graph& graph, const MuxSpec& spec, int batch, bool live, const std::string& name) {
  if (batch <= 0) throw std::invalid_argument("mux batch size must be positive");
  Element mux = graph.Add("nvstreammux", name);
  mux.Set("batch-size", batch)
      .Set("width", spec.width)
      .Set("height", spec.height)
      .Set("gpu-id", spec.gpu_id)
      .Set("batched-push-timeout", spec.batched_push_timeout_us);
  if (live) mux.Set("live-source", true);
  mux.Apply(spec.properties);
  return mux;
}

void LinkToMux(Graph& graph, const gst::Endpoint& from, const Element& mux, int index) {
  graph.Link(from, mux.Pad("sink_" + std::to_string(index)));
}

// ---- 推理与分析 ----

InferSpec InferSpec::Read(Reader reader) {
  InferSpec spec;
  spec.name = reader.Required<std::string>("name", config::PathComponent());
  spec.plugin = reader.Get<std::string>("plugin", "nvinfer", config::OneOf({"nvinfer", "nvinferserver"}));
  spec.config_path = reader.Required<std::string>("config_path", config::NonEmpty());
  spec.properties = reader.Props("properties");
  return spec;
}

std::vector<InferSpec> ReadInferences(Reader& reader, const std::string& key) {
  std::vector<InferSpec> specs;
  std::set<std::string> names;
  for (Reader item : reader.List(key)) {
    InferSpec spec = InferSpec::Read(item);
    if (!names.insert(spec.name).second) item.Fail("name", "duplicate inference name");
    specs.push_back(std::move(spec));
  }
  return specs;
}

Element AddInfer(Graph& graph, const gst::Endpoint& input, const InferSpec& spec) {
  Element infer = graph.Add(spec.plugin, spec.name);
  infer.Set("config-file-path", spec.config_path).Apply(spec.properties);
  graph.Link(input, infer);
  return infer;
}

TrackerSpec TrackerSpec::Read(Reader reader) {
  TrackerSpec spec;
  spec.enabled = reader.Get("enabled", false);
  spec.ll_lib_file = reader.Get<std::string>("ll_lib_file", "");
  spec.ll_config_file = reader.Get<std::string>("ll_config_file", "");
  spec.width = reader.Get("width", 640, config::AtLeast(1));
  spec.height = reader.Get("height", 384, config::AtLeast(1));
  spec.properties = reader.Props("properties");
  if (spec.enabled && (spec.ll_lib_file.empty() || spec.ll_config_file.empty())) {
    reader.Fail("", "ll_lib_file and ll_config_file are required when the tracker is enabled");
  }
  return spec;
}

Element AddTracker(Graph& graph, const gst::Endpoint& input, const TrackerSpec& spec,
                   const std::string& name) {
  Element tracker = graph.Add("nvtracker", name);
  tracker.Set("ll-lib-file", spec.ll_lib_file)
      .Set("ll-config-file", spec.ll_config_file)
      .Set("tracker-width", spec.width)
      .Set("tracker-height", spec.height)
      .Apply(spec.properties);
  graph.Link(input, tracker);
  return tracker;
}

ConfigFileSpec ConfigFileSpec::Read(Reader reader) {
  ConfigFileSpec spec;
  spec.enabled = reader.Get("enabled", false);
  spec.config_file = reader.Get<std::string>("config_file", "");
  spec.properties = reader.Props("properties");
  if (spec.enabled && spec.config_file.empty()) reader.Fail("config_file", "is required when enabled");
  return spec;
}

Element AddAnalytics(Graph& graph, const gst::Endpoint& input, const ConfigFileSpec& spec,
                     const std::string& name) {
  Element element = graph.Add("nvdsanalytics", name);
  element.Set("config-file", spec.config_file).Apply(spec.properties);
  graph.Link(input, element);
  return element;
}

Element AddPreprocess(Graph& graph, const gst::Endpoint& input, const ConfigFileSpec& spec,
                      const std::string& name) {
  Element element = graph.Add("nvdspreprocess", name);
  element.Set("config-file", spec.config_file).Apply(spec.properties);
  graph.Link(input, element);
  return element;
}

ImageSpec ImageSpec::Read(Reader reader) {
  ImageSpec spec;
  spec.enabled = reader.Get("enabled", false);
  spec.quality = reader.Get("quality", 85, config::Range(1, 100));
  ReadOutputSize(reader, &spec.width, &spec.height);
  spec.properties = reader.Props("properties");
  return spec;
}

Element AddMetaTap(Graph& graph, const gst::Endpoint& input, const std::string& name,
                   const ImageSpec& image) {
  Element tap = graph.Add("identity", name);
  if (image.enabled) {
    Element queue = graph.Add("queue", name + "_queue");
    queue.Apply(image.properties);
    graph.Chain({queue, tap});
    graph.Link(input, queue);
  } else {
    graph.Link(input, tap);
  }
  return tap;
}

// ---- 画面输出 ----

OsdSpec OsdSpec::Read(Reader reader) {
  OsdSpec spec;
  spec.enabled = reader.Get("enabled", true);
  spec.properties = reader.Props("properties");
  return spec;
}

Element AddOsd(Graph& graph, const gst::Endpoint& input, const OsdSpec& spec,
               const std::string& name) {
  // nvdsosd 在部分平台只接受 RGBA，前后各转换一次，下游编码器仍拿到 NV12。
  Element before = graph.Add("nvvideoconvert", name + "_in");
  Element osd = graph.Add("nvdsosd", name);
  osd.Apply(spec.properties);
  Element after = graph.Add("nvvideoconvert", name + "_out");
  graph.Link(input, before);
  return graph.Chain({before, osd, after});
}

TilerSpec TilerSpec::Read(Reader reader) {
  TilerSpec spec;
  spec.rows = reader.Get("rows", 0, config::AtLeast(0));
  spec.columns = reader.Get("columns", 0, config::AtLeast(0));
  spec.width = reader.Get("width", 1920, config::AtLeast(1));
  spec.height = reader.Get("height", 1080, config::AtLeast(1));
  spec.properties = reader.Props("properties");
  return spec;
}

Element AddTiler(Graph& graph, const gst::Endpoint& input, const TilerSpec& spec, int sources,
                 const std::string& name) {
  int rows = spec.rows;
  int columns = spec.columns;
  if (rows == 0) rows = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(sources))));
  if (columns == 0) columns = (sources + rows - 1) / rows;
  Element tiler = graph.Add("nvmultistreamtiler", name);
  tiler.Set("rows", rows)
      .Set("columns", columns)
      .Set("width", spec.width)
      .Set("height", spec.height)
      .Apply(spec.properties);
  graph.Link(input, tiler);
  return tiler;
}

Element AddDemux(Graph& graph, const gst::Endpoint& input, const std::string& name) {
  Element demux = graph.Add("nvstreamdemux", name);
  graph.Link(input, demux);
  return demux;
}

EncoderSpec EncoderSpec::Read(Reader reader) {
  EncoderSpec spec;
  spec.codec = reader.Get<std::string>("codec", "h264", config::OneOf({"h264", "h265"}));
  spec.bitrate = reader.Get("bitrate", 4000000, config::AtLeast(1));
  spec.iframe_interval = reader.Get("iframe_interval", 30, config::AtLeast(1));
  spec.properties = reader.Props("properties");
  return spec;
}

Element AddEncoder(Graph& graph, const gst::Endpoint& input, const EncoderSpec& spec,
                   const std::string& name) {
  Element convert = graph.Add("nvvideoconvert", name + "_conv");
  Element encoder = graph.Add(spec.codec == "h265" ? "nvv4l2h265enc" : "nvv4l2h264enc", name);
  encoder.Set("bitrate", spec.bitrate).Set("iframeinterval", spec.iframe_interval).Apply(spec.properties);
  graph.Link(input, convert);
  graph.Link(convert, encoder);
  return encoder;
}

Element AddParser(Graph& graph, const gst::Endpoint& input, const std::string& codec,
                  const std::string& name, bool repeat_parameter_sets) {
  if (codec != "h264" && codec != "h265") throw std::invalid_argument("codec must be h264 or h265");
  Element parser = graph.Add(codec + "parse", name);
  if (repeat_parameter_sets) parser.Set("config-interval", -1);
  graph.Link(input, parser);
  return parser;
}

Element AddRtspOut(Graph& graph, const gst::Endpoint& encoded, const std::string& codec,
                   const std::string& url, const std::string& name, const Properties& properties) {
  Element parser = AddParser(graph, encoded, codec, name + "_parse", true);
  Element sink = graph.Add("rtspclientsink", name);
  // 地址可能带推流口令：诊断输出中整个地址打码，与 RTSP 源一致。
  sink.SetSecret("location", url).Set("protocols", "tcp").Apply(properties);
  graph.Link(parser, sink);
  return sink;
}

Element AddFakeSink(Graph& graph, const gst::Endpoint& input, const std::string& name) {
  Element sink = graph.Add("fakesink", name);
  sink.Set("sync", false);
  graph.Link(input, sink);
  return sink;
}

}  // namespace dsf::ds
