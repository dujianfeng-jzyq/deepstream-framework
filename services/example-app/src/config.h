// example-app 的配置。读取代码即 schema：字段、默认值与约束都写在 Load 中，
// 模板 templates/config/example-app.yaml 与它一一对应（由测试保证）。

#ifndef EXAMPLE_APP_CONFIG_H
#define EXAMPLE_APP_CONFIG_H

#include <cstddef>
#include <string>
#include <vector>

#include "dsf/config/config.h"
#include "dsf/ds/blocks.h"
#include "dsf/publish/publisher.h"
#include "dsf/record/segment_recorder.h"

namespace example_app {

struct RecordingConfig {
  std::string root;
  int segment_seconds = 60;  // 分段内画面的时长，不含断流（RecorderOptions::segment_length）
  bool raw = false;  // 每路源在推理前的画面
  bool osd = true;   // 推理、跟踪、画框之后的画面
  int split_on_gap_ms = 2000;  // 输入断流超过该时长时收尾当前分段、恢复后开始新分段，0 关闭（RecorderOptions::split_on_gap）
  dsf::config::Properties properties;  // 透传给 splitmuxsink
  dsf::record::Retention retention;    // 每个录像器（每路源的每个分支）独立计数
};

struct RtspOutConfig {
  bool enabled = false;
  std::string url;
  std::string password;  // 单独配置，推流时写进 url 的用户名之后（见 PublishUrl），日志中打码
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
  int max_length = 100;            // 只为吸收消费端的短暂抖动，不是存储
  std::size_t queue_bytes = 64u << 20;  // 发布队列的字节上限（带图片时一条约几百 KB）
  dsf::ds::ImageSpec image;        // 有目标的帧编码 JPEG，随检测消息发送
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

// 读取并校验跨字段规则；失败抛 dsf::config::Error。合法但多半不是想要的组合
// （例如 h265 的 RTSP 输出在浏览器 WebRTC 中无法播放）登记为 root.Warn，由框架输出。
Config Load(dsf::config::Reader& root);

// 推流地址：口令编码后写在用户名之后（dsf::ds::InjectPassword）。口令为空时也写出 "user:"：
// MediaMTX 的推流用户口令为空时接受任意口令，但只认带凭据的客户端。
std::string PublishUrl(const std::string& url, const std::string& password);

}  // namespace example_app

#endif  // EXAMPLE_APP_CONFIG_H
