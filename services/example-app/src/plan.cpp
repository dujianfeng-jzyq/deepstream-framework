// example-app 拓扑的实现，形状见 plan.h。

#include "plan.h"

#include "dsf/record/segment_recorder.h"

namespace example_app {

using dsf::gst::Element;
namespace ds = dsf::ds;

Plan BuildGraph(const Config& config, dsf::gst::Graph& graph) {
  Plan plan;
  const int count = static_cast<int>(config.sources.size());
  const std::string& codec = config.encoder.codec;
  const auto record = [&](const dsf::gst::Endpoint& encoded, int source, const std::string& branch) {
    const std::string name = "rec_" + branch + "_" + std::to_string(source);
    dsf::record::AddSegmentSink(graph, encoded, codec, name, config.recording.properties);
    plan.recorders.push_back({source, branch, name});
  };
  const auto encode = [&](const dsf::gst::Endpoint& input, const std::string& name) {
    plan.encoders.push_back(name);
    return ds::AddEncoder(graph, input, config.encoder, name);
  };

  Element mux = ds::AddMux(graph, config.mux, count, config.live());
  for (int i = 0; i < count; ++i) {
    const std::string prefix = "src" + std::to_string(i);
    Element source = ds::AddSource(graph, config.sources[static_cast<std::size_t>(i)], prefix);
    if (!config.recording.raw) {
      ds::LinkToMux(graph, source, mux, i);
      continue;
    }
    // raw 录像在规范化之后、推理之前分出，与推理链互不影响。
    Element tee = graph.Tee(source, prefix + "_tee");
    record(encode(graph.Branch(tee), prefix + "_raw_enc"), i, "raw");
    ds::LinkToMux(graph, graph.Branch(tee), mux, i);
  }

  Element tail = mux;
  for (const auto& infer : config.inference) tail = ds::AddInfer(graph, tail, infer);
  if (config.tracker.enabled) tail = ds::AddTracker(graph, tail, config.tracker);
  if (config.analytics.enabled) tail = ds::AddAnalytics(graph, tail, config.analytics);
  tail = ds::AddMetaTap(graph, tail, kMetaTap, config.redis.image);

  const bool rtsp = config.rtsp_out.enabled;
  const std::string rtsp_url = rtsp ? PublishUrl(config.rtsp_out.url, config.rtsp_out.password) : "";
  const bool osd_recording = config.recording.osd;
  if (!rtsp && !osd_recording) {
    ds::AddFakeSink(graph, tail, "sink");  // 只发消息时不编码
    return plan;
  }
  if (config.osd.enabled) tail = ds::AddOsd(graph, tail, config.osd);

  if (count == 1) {
    Element encoded = encode(tail, "enc");
    if (rtsp && osd_recording) {
      Element tee = graph.Tee(encoded, "enc_tee");
      ds::AddRtspOut(graph, graph.Branch(tee), codec, rtsp_url, "rtsp_out",
                     config.rtsp_out.properties);
      record(graph.Branch(tee), 0, "osd");
    } else if (rtsp) {
      ds::AddRtspOut(graph, encoded, codec, rtsp_url, "rtsp_out", config.rtsp_out.properties);
    } else {
      record(encoded, 0, "osd");
    }
    return plan;
  }

  Element rtsp_input = tail;
  Element record_input = tail;
  if (rtsp && osd_recording) {
    Element tee = graph.Tee(tail, "out_tee");
    rtsp_input = graph.Branch(tee);
    record_input = graph.Branch(tee);
  }
  if (rtsp) {
    Element tiled = ds::AddTiler(graph, rtsp_input, config.rtsp_out.tiler, count);
    ds::AddRtspOut(graph, encode(tiled, "rtsp_enc"), codec,
                   rtsp_url, "rtsp_out", config.rtsp_out.properties);
  }
  if (osd_recording) {
    Element demux = ds::AddDemux(graph, record_input);
    for (int i = 0; i < count; ++i) {
      const std::string prefix = "out" + std::to_string(i);
      Element queue = graph.Add("queue", prefix + "_queue");
      graph.Link(demux.Pad("src_" + std::to_string(i)), queue);
      record(encode(queue, prefix + "_enc"), i, "osd");
    }
  }
  return plan;
}

}  // namespace example_app
