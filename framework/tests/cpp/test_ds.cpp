// dsf::ds：块组图（结构与属性，不需要 DeepStream 插件）与消息编码。

#include <nlohmann/json.hpp>

#include "dsf/ds/blocks.h"
#include "dsf/ds/message.h"
#include "dsf_test.h"

using namespace dsf::ds;
using dsf::config::Document;

namespace {

std::string Property(const Graph& graph, const std::string& element, const std::string& name) {
  for (const auto& node : graph.nodes()) {
    if (node.name != element) continue;
    for (const auto& property : node.properties) {
      if (property.name == name) return property.value;
    }
  }
  return "<unset>";
}

std::vector<SourceSpec> Sources(const std::string& yaml) {
  const Document document = Document::LoadText(yaml, "test.yaml");
  dsf::config::Reader root = document.Root();
  auto sources = ReadSources(root);
  root.Finish();
  return sources;
}

}  // namespace

TEST(rtsp_source_injects_password_and_redacts_it) {
  const auto sources = Sources(
      "sources:\n  - id: cam\n    type: rtsp\n    uri: rtsp://admin@10.0.0.1/s\n"
      "    rtsp: {password: pw, latency_ms: 100}\n    output: {width: 1280, height: 720}\n");
  CHECK(sources[0].live);
  Graph graph;
  Element out = AddSource(graph, sources[0], "src0");
  CHECK_EQ(out.name(), std::string("src0_nvmm"));
  CHECK_EQ(Property(graph, "src0", "uri"), std::string("rtsp://admin:pw@10.0.0.1/s"));
  CHECK_EQ(Property(graph, "src0", "latency"), std::string("100"));
  CHECK_EQ(Property(graph, "src0", "select-rtp-protocol"), std::string("4"));
  CHECK_EQ(Property(graph, "src0", "num-extra-surfaces"), std::string("8"));
  CHECK_EQ(Property(graph, "src0_nvmm", "caps"),
           std::string("video/x-raw(memory:NVMM),format=NV12,width=1280,height=720"));
  CHECK(graph.Describe().find("pw") == std::string::npos);
}

TEST(source_validation) {
  CHECK_THROWS(Sources("sources:\n  - id: a\n    type: rtsp\n    uri: http://x\n"), "rtsp://");
  CHECK_THROWS(Sources("sources:\n  - id: a\n    type: rtsp\n    uri: rtsp://host/s\n    rtsp: {password: p}\n"),
               "no user name");
  CHECK_THROWS(Sources("sources:\n  - id: a\n    type: file\n"), "uri: is required");
  CHECK_THROWS(Sources("sources:\n  - id: a\n    type: test\n  - id: a\n    type: test\n"), "duplicate source id");
  CHECK_THROWS(Sources("sources:\n  - id: ../x\n    type: test\n"), "single path component");
  CHECK_THROWS(Sources("sources: []\n"), "at least 1");
  CHECK_THROWS(Sources("sources:\n  - id: a\n    type: test\n    output: {width: 10}\n"), "set together");
  CHECK_EQ(InjectPassword("rtsp://h/s", ""), std::string("rtsp://h/s"));
  CHECK_THROWS(InjectPassword("rtsp://u:x@h/s", "p"), "already contains");
  CHECK_THROWS(InjectPassword("rtsp://u@h/s", "a@b"), "must not contain");
}

TEST(file_and_test_sources) {
  const auto sources = Sources(
      "sources:\n  - id: f\n    type: file\n    uri: /videos/a.mp4\n    file: {loop: false}\n"
      "  - id: t\n    type: test\n    properties: {pattern: ball}\n");
  CHECK(!sources[0].live);
  CHECK(sources[1].live);
  Graph graph;
  AddSource(graph, sources[0], "src0");
  AddSource(graph, sources[1], "src1");
  CHECK_EQ(Property(graph, "src0", "uri"), std::string("file:///videos/a.mp4"));
  CHECK_EQ(Property(graph, "src0", "file-loop"), std::string("false"));
  CHECK_EQ(Property(graph, "src1", "pattern"), std::string("ball"));
  CHECK_EQ(Property(graph, "src1_raw", "caps"), std::string("video/x-raw,width=1280,height=720,framerate=25/1"));
}

TEST(mux_infer_and_outputs_build_expected_graph) {
  Graph graph;
  SourceSpec test;
  test.id = "t";
  test.type = "test";
  test.width = 320;
  test.height = 240;
  test.framerate = 25;
  Element mux = AddMux(graph, MuxSpec{}, 2, true);
  LinkToMux(graph, AddSource(graph, test, "src0"), mux, 0);
  LinkToMux(graph, AddSource(graph, test, "src1"), mux, 1);
  InferSpec pgie{"pgie", "nvinfer", "/models/a.txt", {}};
  InferSpec sgie{"sgie", "nvinferserver", "/models/b.txt", {{"interval", "2", "inference[1].properties.interval"}}};
  Element tail = AddInfer(graph, AddInfer(graph, mux, pgie), sgie);
  tail = AddMetaTap(graph, tail, "meta");
  Element tiler = AddTiler(graph, tail, TilerSpec{}, 3);
  EncoderSpec encoder;
  encoder.codec = "h265";
  AddRtspOut(graph, AddEncoder(graph, tiler, encoder, "enc"), "h265", "rtsp://out/x", "rtsp");

  CHECK_EQ(Property(graph, "mux", "batch-size"), std::string("2"));
  CHECK_EQ(Property(graph, "mux", "live-source"), std::string("true"));
  CHECK_EQ(Property(graph, "sgie", "interval"), std::string("2"));
  CHECK_EQ(Property(graph, "tiler", "rows"), std::string("2"));
  CHECK_EQ(Property(graph, "tiler", "columns"), std::string("2"));
  CHECK_EQ(Property(graph, "rtsp_parse", "config-interval"), std::string("-1"));
  CHECK(graph.Contains("enc"));
  CHECK_EQ(graph.nodes()[static_cast<std::size_t>(graph.Find("enc").Pad("").element)].factory,
           std::string("nvv4l2h265enc"));
  const std::string text = graph.Describe();
  CHECK(text.find("src1_nvmm -> mux.sink_1") != std::string::npos);
  CHECK(text.find("pgie -> sgie") != std::string::npos);
  const std::string requirements = graph.Requirements().Describe();
  CHECK(requirements.find("value    nvinferserver.interval=2") != std::string::npos);
}

TEST(detection_message_schema) {
  FrameRecord frame;
  frame.source_id = "cam";
  frame.frame_num = 7;
  frame.wall_ms = 1000;
  frame.width = 100;
  frame.height = 50;
  ObjectRecord object;
  object.class_id = 2;
  object.label = "车辆\"x\"";
  object.confidence = 0.98765;
  object.box = {-5.5, 10.9, 50, 100};  // 裁剪到帧内
  object.track_id = 0;                   // 0 是合法编号
  object.classes.push_back({2, 1, "red", 0.5});
  frame.objects.push_back(object);
  ObjectRecord tiny;
  tiny.box = {10.2, 10.2, 0.5, 0.5};  // 取整后零宽，被丢弃
  frame.objects.push_back(tiny);

  const auto message = nlohmann::json::parse(Serialize(DetectionMessage(frame)));
  CHECK_EQ(message["schema_version"].get<int>(), 1);
  CHECK_EQ(message["event_type"].get<std::string>(), std::string("detection"));
  CHECK_EQ(message["source_id"].get<std::string>(), std::string("cam"));
  CHECK_EQ(message["objects"].size(), 1u);
  const auto& item = message["objects"][0];
  CHECK_EQ(item["label"].get<std::string>(), std::string("车辆\"x\""));
  CHECK_EQ(item["confidence"].get<double>(), 0.988);
  CHECK_EQ(item["track_id"].get<int>(), 0);
  CHECK_EQ(item["bbox"]["x"].get<int>(), 0);
  CHECK_EQ(item["bbox"]["y"].get<int>(), 10);
  CHECK_EQ(item["bbox"]["width"].get<int>(), 44);
  CHECK_EQ(item["bbox"]["height"].get<int>(), 40);
  CHECK_EQ(item["classes"][0]["label"].get<std::string>(), std::string("red"));
  CHECK(!message.contains("extra"));
  CHECK(!message.contains("ntp_ms"));

  ObjectRecord untracked;
  untracked.box = {0, 0, 10, 10};
  frame.objects = {untracked};
  CHECK(!DetectionMessage(frame)["objects"][0].contains("track_id"));
}

TEST(recording_messages) {
  RecordingEvent event;
  event.source_id = "cam";
  event.branch = "osd";
  event.segment_id = "cam-osd-1-0";
  event.start_ms = 1000;
  const auto started = RecordingMessage(event);
  CHECK_EQ(started["event_type"].get<std::string>(), std::string("recording_started"));
  CHECK(!started.contains("path"));
  event.type = RecordingEvent::Type::kFinished;
  event.end_ms = 500;  // 时钟回拨
  event.path = "/data/x.mp4";
  const auto finished = RecordingMessage(event);
  CHECK_EQ(finished["duration_ms"].get<int>(), 0);
  CHECK_EQ(finished["path"].get<std::string>(), std::string("/data/x.mp4"));
  CHECK_EQ(finished["timestamp_ms"].get<int>(), 500);
}

TEST(serialize_replaces_invalid_utf8) {
  const std::string text = Serialize(nlohmann::json{{"label", std::string("a\xff")}});
  CHECK(text.find("\xEF\xBF\xBD") != std::string::npos);
}

TEST(source_registry) {
  SourceRegistry registry;
  registry.Add("a");
  registry.Add("b");
  CHECK_EQ(registry.IdOf(1), std::string("b"));
  CHECK_EQ(registry.IdOf(5), std::string("source5"));
  CHECK(registry.IndexOf("b").value() == 1u);
}

DSF_TEST_MAIN()
