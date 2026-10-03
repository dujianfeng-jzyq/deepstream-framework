// 框架内置的消息编码（schema v1）。应用可以直接使用，也可以完全使用自己的编码；
// 字段定义见 docs/framework.md 的“消息”一节。
//
// 共同字段：schema_version、event_type、source_id、timestamp_ms。
// 可选字段只在有值时出现（例如 track_id），不用 0 或空串冒充“没有”。

#ifndef DSF_DS_MESSAGE_H
#define DSF_DS_MESSAGE_H

#include <cstdint>
#include <string>

#include <nlohmann/json.hpp>

#include "dsf/ds/meta.h"

namespace dsf::ds {

inline constexpr int kSchemaVersion = 1;

// event_type=detection。框选裁剪到帧内后取整，取整后零宽或零高的目标被丢弃；
// 置信度保留三位小数，非有限值记为 0。
nlohmann::json DetectionMessage(const FrameRecord& frame);

struct RecordingEvent {
  enum class Type { kStarted, kFinished };
  Type type = Type::kStarted;
  std::string source_id;
  std::string branch;
  std::string segment_id;
  std::string path;  // 只有 finished 才有：已完成封装、可直接读取的文件
  std::uint64_t start_ms = 0;
  std::uint64_t end_ms = 0;
  bool detected = false;
};

// event_type=recording_started / recording_finished。
nlohmann::json RecordingMessage(const RecordingEvent& event);

// 紧凑 JSON；非法 UTF-8 字节替换为 U+FFFD，不抛异常。
std::string Serialize(const nlohmann::json& message);

}  // namespace dsf::ds

#endif  // DSF_DS_MESSAGE_H
