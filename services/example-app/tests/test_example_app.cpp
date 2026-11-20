// example-app 的配置与拓扑测试，不需要 DeepStream SDK。

#include <fstream>
#include <regex>
#include <sstream>

#include "config.h"
#include "dsf/app/app.h"
#include "dsf_test.h"
#include "plan.h"

using dsf::config::Document;

namespace {

// 模板中的 ${VAR} 用占位值替换，模拟渲染后的配置。
std::string Template() {
  std::ifstream stream(EXAMPLE_APP_TEMPLATE);
  std::stringstream buffer;
  buffer << stream.rdbuf();
  std::string text = buffer.str();
  text = std::regex_replace(text, std::regex(R"(\$\{RTSP_URL\})"), "rtsp://cam@host/stream");
  text = std::regex_replace(text, std::regex(R"(\$\{[A-Z_]+\})"), "x");
  return text;
}

example_app::Config Load(const std::string& text) {
  const Document document = Document::LoadText(text, "example-app.yaml");
  dsf::config::Reader root = document.Root();
  (void)root.Section("logging");  // 由框架读取；这里只验证应用自己的字段
  auto config = example_app::Load(root);
  return config;
}

// 去掉整个推理链（一级与二级），换成空列表。
std::string WithoutInference(const std::string& text) {
  const std::string result =
      std::regex_replace(text, std::regex(R"(inference:\n((  - |    |  #)[^\n]*\n)+)"), "inference: []\n");
  if (result == text) throw std::runtime_error("template has no inference list");
  return result;
}

std::vector<std::string> WarningsOf(const std::string& text) {
  const Document document = Document::LoadText(text, "example-app.yaml");
  dsf::config::Reader root = document.Root();
  (void)example_app::Load(root);
  return root.Warnings();
}

std::string Replace(std::string text, const std::string& from, const std::string& to) {
  const auto position = text.find(from);
  if (position == std::string::npos) throw std::runtime_error("template has no " + from);
  return text.replace(position, from.size(), to);
}

std::string Describe(const example_app::Config& config, example_app::Plan* plan = nullptr) {
  dsf::gst::Graph graph;
  auto result = example_app::BuildGraph(config, graph);
  if (plan != nullptr) *plan = result;
  return graph.Describe();
}

const char* kSecondSource =
    "  - id: cam2\n    type: file\n    uri: /videos/a.mp4\n\n# mux";

}  // namespace

TEST(template_matches_the_reader) {
  const std::string text = Template();
  const Document document = Document::LoadText(text, "example-app.yaml");
  dsf::config::Reader root = document.Root();
  (void)dsf::app::FrameworkOptions::Read(root);  // 日志、看门狗与就绪段由框架读取
  (void)example_app::Load(root);
  root.Finish();  // 模板没有读取代码不认识的键
  const auto declared = root.DeclaredPaths();
  for (const auto& leaf : document.LeafPaths()) {
    if (std::find(declared.begin(), declared.end(), leaf) == declared.end()) {
      throw dsf_test::Failure("template field not read by the app: " + leaf);
    }
  }
}

TEST(single_source_shares_one_encoder) {
  example_app::Plan plan;
  const std::string text = Describe(Load(Template()), &plan);
  CHECK(text.find("src0_nvmm -> mux.sink_0") != std::string::npos);
  CHECK(text.find("mux -> pgie") != std::string::npos);
  CHECK(text.find("pgie -> sgie") != std::string::npos);  // 二级分类紧接一级检测，在跟踪之前
  CHECK(text.find("sgie -> tracker") != std::string::npos);
  CHECK(text.find("tracker -> meta") != std::string::npos);
  CHECK(text.find("enc -> enc_tee") != std::string::npos);
  CHECK(text.find("rtsp_out = rtspclientsink location=***") != std::string::npos);
  CHECK(text.find("uri=***") != std::string::npos);
  CHECK_EQ(plan.recorders.size(), 1u);
  CHECK_EQ(plan.recorders[0].element, std::string("rec_osd_0"));
}

TEST(multiple_sources_tile_rtsp_and_record_per_source) {
  const std::string text = Replace(Template(), "\n# mux", kSecondSource);
  const auto config = Load(text);
  CHECK_EQ(config.sources.size(), 2u);
  example_app::Plan plan;
  const std::string graph = Describe(config, &plan);
  CHECK(graph.find("mux = nvstreammux batch-size=2") != std::string::npos);
  CHECK(graph.find("src1_nvmm -> mux.sink_1") != std::string::npos);
  CHECK(graph.find("tiler = nvmultistreamtiler") != std::string::npos);
  CHECK(graph.find("demux.src_1 -> out1_queue") != std::string::npos);
  CHECK_EQ(plan.recorders.size(), 2u);
  CHECK_EQ(plan.recorders[1].element, std::string("rec_osd_1"));
}

TEST(raw_recording_branches_before_inference) {
  const std::string text = Replace(Template(), "  raw: false", "  raw: true");
  example_app::Plan plan;
  const std::string graph = Describe(Load(text), &plan);
  CHECK(graph.find("src0_nvmm -> src0_tee") != std::string::npos);
  CHECK_EQ(plan.recorders.size(), 2u);
  CHECK_EQ(plan.recorders[0].branch, std::string("raw"));
}

TEST(message_only_deployment_has_no_encoder) {
  std::string text = Replace(Template(), "  osd: true\n\nrtsp_out:\n  enabled: true",
                             "  osd: false\n\nrtsp_out:\n  enabled: false");
  const std::string graph = Describe(Load(text));
  CHECK(graph.find("meta -> sink") != std::string::npos);
  CHECK(graph.find("nvv4l2h264enc") == std::string::npos);
}

TEST(image_encoding_adds_a_queue_before_the_tap) {
  CHECK(Describe(Load(Template())).find("tracker -> meta") != std::string::npos);
  const auto config = Load(Replace(Template(), "  image:\n    enabled: false", "  image:\n    enabled: true"));
  CHECK(config.redis.image.enabled);
  CHECK_EQ(config.redis.max_length, 100);
  const std::string graph = Describe(config);
  CHECK(graph.find("tracker -> meta_queue") != std::string::npos);
  CHECK(graph.find("meta_queue -> meta") != std::string::npos);
}

TEST(publish_url_encodes_the_password_after_the_user) {
  using example_app::PublishUrl;
  CHECK_EQ(PublishUrl("rtsp://publisher@m:8554/a", "p$w#x@&"),
           std::string("rtsp://publisher:p%24w%23x%40%26@m:8554/a"));
  // 口令为空时也带上 "user:"：MediaMTX 只认带凭据的推流客户端。
  CHECK_EQ(PublishUrl("rtsp://publisher@m:8554/a", ""), std::string("rtsp://publisher:@m:8554/a"));
  CHECK_EQ(PublishUrl("rtsp://m:8554/a", ""), std::string("rtsp://m:8554/a"));
  CHECK_EQ(PublishUrl("rtsp://u:p@m/a", ""), std::string("rtsp://u:p@m/a"));
  CHECK_THROWS(PublishUrl("rtsp://m:8554/a@b", "x"), "has no user name");
  CHECK_THROWS(PublishUrl("rtsp://u:p@m/a", "x"), "already contains a password");
}

TEST(rtsp_output_password_is_checked_and_hidden) {
  const std::string base = Template();
  const auto config = Load(Replace(base, "  password: x\n  tiler:", "  password: \"a$b\"\n  tiler:"));
  CHECK_EQ(config.rtsp_out.password, std::string("a$b"));
  CHECK(Describe(config).find("a%24b") == std::string::npos);  // location 整体打码
  CHECK_THROWS(Load(Replace(base, "  password: x\n  tiler:", "  password: \"a,b\"\n  tiler:")),
               "rtsp_out.password: MediaMTX accepts only");
  CHECK_THROWS(Load(Replace(base, "url: rtsp://publisher@mediamtx", "url: rtsp://mediamtx")),
               "rtsp_out.password: is set but the uri has no user name");
}

TEST(rtsp_output_path_must_be_a_mediamtx_path) {
  const std::string base = Template();  // 模板的推流路径是 ${CAMERA_ID}-out
  const auto with_path = [&](const std::string& path) {
    return Replace(base, "mediamtx:8554/x-out", "mediamtx:8554/" + path);
  };
  CHECK(Load(with_path("gate:1-out?tcp")).rtsp_out.enabled);  // 查询串不属于路径
  CHECK(Load(with_path("site/cam_1.~-out")).rtsp_out.enabled);
  // MediaMTX 拒绝的路径在预检中报错，而不是在现场反复启动失败。
  for (const char* bad : {"东门-out", "gate 1-out", "a#b", ""}) {
    CHECK_THROWS(Load(with_path(bad)), "rtsp_out.url: path");
  }
  // 不推流时路径不检查：相机编号仍可用于录像目录与消息。
  CHECK(!Load(Replace(with_path("东门-out"), "rtsp_out:\n  enabled: true", "rtsp_out:\n  enabled: false"))
             .rtsp_out.enabled);
}

TEST(h265_rtsp_output_warns_about_browser_playback) {
  CHECK(WarningsOf(Template()).empty());
  const auto warnings = WarningsOf(Replace(Template(), "  codec: h264", "  codec: h265"));
  CHECK_EQ(warnings.size(), 1u);
  CHECK(warnings[0].find("example-app.yaml: encoder.codec: ") == 0);
  CHECK(warnings[0].find("WebRTC") != std::string::npos);
}

TEST(cross_field_rules) {
  const std::string base = Template();
  CHECK_THROWS(Load(WithoutInference(base)),
               "tracker.enabled: requires at least one inference");
  std::string no_output = Replace(base, "  osd: true\n\nrtsp_out:\n  enabled: true",
                                  "  osd: false\n\nrtsp_out:\n  enabled: false");
  no_output = Replace(no_output, "redis:\n  enabled: true", "redis:\n  enabled: false");
  CHECK_THROWS(Load(no_output), "enable at least one output");
  // 没有推理时 Redis 收不到检测消息，不算输出。
  std::string redis_only = Replace(base, "  osd: true\n\nrtsp_out:\n  enabled: true",
                                   "  osd: false\n\nrtsp_out:\n  enabled: false");
  redis_only = Replace(redis_only, "tracker:\n  enabled: true", "tracker:\n  enabled: false");
  CHECK(Load(redis_only).redis.enabled);
  CHECK_THROWS(Load(WithoutInference(redis_only)),
               "or redis with at least one inference entry");
  CHECK_THROWS(Load(Replace(base, "  root: x/recordings", "  root: \"\"")), "recording.root: is required");
  // 只替换保留策略的数值，不依赖模板中的具体值（现场可能为验证而调小）。
  const std::regex retention(R"(    max_files: \d+\n    keep_files: \d+\n    min_free_mb: \d+)");
  const auto with_retention = [&](const std::string& values) {
    const std::string text = std::regex_replace(base, retention, values);
    if (text == base) throw std::runtime_error("template has no recording.retention");
    return text;
  };
  CHECK_THROWS(Load(with_retention("    max_files: 2\n    keep_files: 3\n    min_free_mb: 0")),
               "recording.retention.keep_files: must not exceed max_files");
  const auto config = Load(with_retention("    max_files: 0\n    keep_files: 3\n    min_free_mb: 7"));
  CHECK_EQ(config.recording.retention.max_files, std::size_t{0});  // 0 表示不限数量，不检查 keep_files
  CHECK_EQ(config.recording.retention.min_free_bytes, std::uint64_t{7} << 20);
  CHECK_THROWS(Load(Replace(base, "  codec: h264", "  codec: vp9")), "encoder.codec");
  const std::string images = Replace(base, "  image:\n    enabled: false", "  image:\n    enabled: true");
  CHECK_THROWS(Load(Replace(images, "redis:\n  enabled: true", "redis:\n  enabled: false")),
               "redis.image.enabled: requires redis.enabled");
}

DSF_TEST_MAIN()
