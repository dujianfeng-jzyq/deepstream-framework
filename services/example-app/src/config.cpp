// example-app 配置的读取与跨字段校验。

#include "config.h"

namespace example_app {

using dsf::config::AtLeast;
using dsf::config::NonEmpty;
using dsf::config::Reader;

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

  Reader rtsp = root.Section("rtsp_out");
  config.rtsp_out.enabled = rtsp.Get("enabled", false);
  config.rtsp_out.url = rtsp.Get<std::string>("url", "");
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
  if (config.redis.image.enabled && !config.redis.enabled) {
    root.Fail("redis.image.enabled", "requires redis.enabled (images are sent with detection messages)");
  }
  if (config.redis.image.enabled && config.inference.empty()) {
    root.Fail("redis.image.enabled", "requires at least one inference entry (images are taken on detections)");
  }
  if (config.rtsp_out.enabled && config.rtsp_out.url.empty()) {
    root.Fail("rtsp_out.url", "is required when rtsp_out is enabled");
  }
  // 只进不出的管线占用相机与 GPU 却没有任何产出。
  if (!config.records() && !config.rtsp_out.enabled && !config.redis.enabled) {
    root.Fail("", "enable at least one output: recording.raw, recording.osd, rtsp_out or redis");
  }
  return config;
}

}  // namespace example_app
