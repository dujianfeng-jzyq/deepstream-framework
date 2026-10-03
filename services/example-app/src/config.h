// example-app 的配置。读取代码即 schema：字段、默认值与约束都写在 Load 中，
// 模板 templates/config/example-app.yaml 与它一一对应（由测试保证）。

#ifndef EXAMPLE_APP_CONFIG_H
#define EXAMPLE_APP_CONFIG_H

#include <string>
#include <vector>

#include "dsf/config/config.h"
#include "dsf/ds/blocks.h"
#include "dsf/publish/publisher.h"

namespace example_app {

struct RecordingConfig {
  std::string root;
  int segment_seconds = 60;
  bool raw = false;  // 每路源在推理前的画面
  bool osd = true;   // 推理、跟踪、画框之后的画面
  dsf::config::Properties properties;  // 透传给 splitmuxsink
};

struct RtspOutConfig {
  bool enabled = false;
  std::string url;
  dsf::ds::TilerSpec tiler;  // 多路时拼接成一路输出
  dsf::config::Properties properties;  // 透传给 rtspclientsink
};

struct RedisConfig {
  bool enabled = false;
  dsf::publish::RedisOptions options;
  std::string detection_stream;
  std::string recording_stream;
  bool only_on_detection = true;  // 没有目标的帧不发消息：现场绝大多数帧是空的
  int queue_capacity = 256;
  int max_length = 10000;
};

struct Config {
  std::vector<dsf::ds::SourceSpec> sources;
  dsf::ds::MuxSpec mux;
  std::vector<dsf::ds::InferSpec> inference;
  dsf::ds::TrackerSpec tracker;
  dsf::ds::ConfigFileSpec analytics;
  dsf::ds::OsdSpec osd;
  dsf::ds::EncoderSpec encoder;
  RecordingConfig recording;
  RtspOutConfig rtsp_out;
  RedisConfig redis;

  bool live() const;
  bool records() const { return recording.raw || recording.osd; }
};

// 读取并校验跨字段规则；失败抛 dsf::config::Error。
Config Load(dsf::config::Reader& root);

}  // namespace example_app

#endif  // EXAMPLE_APP_CONFIG_H
