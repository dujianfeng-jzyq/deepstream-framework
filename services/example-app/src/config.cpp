// example-app 配置的读取与跨字段校验。

#include "config.h"

#include <stdexcept>

namespace example_app {

using dsf::config::AtLeast;
using dsf::config::NonEmpty;
using dsf::config::Reader;

namespace {

// MediaMTX 只接受这些字符组成的口令，其他字符会让它拒绝启动。
constexpr const char* kMediaMtxCredential =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!$()*+.;<=>[]^_-@#&";

}  // namespace

std::string PublishUrl(const std::string& url, const std::string& password) {
  return dsf::ds::InjectPassword(url, password, true);
}

bool Config::live() const {
  for (const auto& source : sources) {
    if (source.live) return true;
  }
  return false;
}

Config Load(Reader& root) {
  Config config;
  config.sources = dsf::ds::ReadSources(root);
  config.mux = dsf::ds::MuxSpec::Read(root.Section("mux"));
  config.inference = dsf::ds::ReadInferences(root);
  config.tracker = dsf::ds::TrackerSpec::Read(root.Section("tracker"));
  config.analytics = dsf::ds::ConfigFileSpec::Read(root.Section("analytics"));
  config.osd = dsf::ds::OsdSpec::Read(root.Section("osd"));
  config.encoder = dsf::ds::EncoderSpec::Read(root.Section("encoder"));

  Reader recording = root.Section("recording");
  config.recording.root = recording.Get<std::string>("root", "");
  config.recording.segment_seconds = recording.Get("segment_seconds", 60, AtLeast(1));
  config.recording.raw = recording.Get("raw", false);
  config.recording.osd = recording.Get("osd", true);
  config.recording.properties = recording.Props("properties");
  Reader retention = recording.Section("retention");
  config.recording.retention.max_files =
      static_cast<std::size_t>(retention.Get("max_files", 0, AtLeast(0)));
  config.recording.retention.keep_files =
      static_cast<std::size_t>(retention.Get("keep_files", 0, AtLeast(0)));
  config.recording.retention.min_free_bytes =
      static_cast<std::uint64_t>(retention.Get("min_free_mb", 0, AtLeast(0))) << 20;

  Reader rtsp = root.Section("rtsp_out");
  config.rtsp_out.enabled = rtsp.Get("enabled", false);
  config.rtsp_out.url = rtsp.Get<std::string>("url", "");
  config.rtsp_out.password = rtsp.Secret("password");
  config.rtsp_out.tiler = dsf::ds::TilerSpec::Read(rtsp.Section("tiler"));
  config.rtsp_out.properties = rtsp.Props("properties");

  Reader redis = root.Section("redis");
  config.redis.enabled = redis.Get("enabled", false);
  config.redis.options = dsf::publish::RedisOptions::Read(redis);
  config.redis.detection_stream = redis.Get<std::string>("detection_stream", "detections", NonEmpty());
  config.redis.recording_stream = redis.Get<std::string>("recording_stream", "recordings", NonEmpty());
  config.redis.only_on_detection = redis.Get("only_on_detection", true);
  config.redis.queue_capacity = redis.Get("queue_capacity", 256, AtLeast(1));
  config.redis.max_length = redis.Get("max_length", 100, AtLeast(1));
  config.redis.queue_bytes =
      static_cast<std::size_t>(redis.Get("queue_bytes_mb", 64, AtLeast(1))) << 20;
  config.redis.image = dsf::ds::ImageSpec::Read(redis.Section("image"));

  // ---- 跨字段规则 ----
  if (config.tracker.enabled && config.inference.empty()) {
    root.Fail("tracker.enabled", "requires at least one inference entry (nothing to track)");
  }
  if (config.analytics.enabled && config.inference.empty()) {
    root.Fail("analytics.enabled", "requires at least one inference entry");
  }
  if (config.records() && config.recording.root.empty()) {
    root.Fail("recording.root", "is required when raw or osd recording is enabled");
  }
  if (config.recording.retention.max_files > 0 &&
      config.recording.retention.keep_files > config.recording.retention.max_files) {
    root.Fail("recording.retention.keep_files", "must not exceed max_files");
  }
  if (config.redis.image.enabled && !config.redis.enabled) {
    root.Fail("redis.image.enabled", "requires redis.enabled (images are sent with detection messages)");
  }
  if (config.redis.image.enabled && config.inference.empty()) {
    root.Fail("redis.image.enabled", "requires at least one inference entry (images are taken on detections)");
  }
  if (config.rtsp_out.enabled && config.rtsp_out.url.empty()) {
    root.Fail("rtsp_out.url", "is required when rtsp_out is enabled");
  }
  // 口令来自现场参数 MEDIAMTX_PUBLISH_PASSWORD，与 MediaMTX 共用：在预检中拦下它不接受的字符，
  // 而不是停服之后 MediaMTX 无法启动。
  if (config.rtsp_out.password.find_first_not_of(kMediaMtxCredential) != std::string::npos) {
    root.Fail("rtsp_out.password", "MediaMTX accepts only letters, digits and !$()*+.;<=>[]^_-@#&");
  }
  if (config.rtsp_out.enabled) {
    try {
      (void)PublishUrl(config.rtsp_out.url, config.rtsp_out.password);
    } catch (const std::invalid_argument& error) {
      root.Fail("rtsp_out.password", error.what());
    }
  }
  // 只进不出的管线占用相机与 GPU 却没有任何产出。没有推理时 Redis 不会收到检测消息，
  // 录像事件又只在录像开启时产生，所以 Redis 只有配合推理才算输出。
  const bool publishes = config.redis.enabled && !config.inference.empty();
  if (!config.records() && !config.rtsp_out.enabled && !publishes) {
    root.Fail("", "enable at least one output: recording.raw, recording.osd, rtsp_out, "
                  "or redis with at least one inference entry");
  }
  return config;
}

}  // namespace example_app
