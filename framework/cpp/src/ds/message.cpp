// 消息编码的实现。

#include "dsf/ds/message.h"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace dsf::ds {
namespace {

double Confidence(double value) {
  return std::isfinite(value) ? std::round(value * 1000.0) / 1000.0 : 0.0;
}

// 裁剪到 [0, limit] 后向零取整。
int Clamp(double value, int limit) {
  if (!std::isfinite(value)) return 0;
  return static_cast<int>(std::clamp(value, 0.0, static_cast<double>(limit)));
}

nlohmann::json Envelope(const char* type, const std::string& source_id, std::uint64_t timestamp_ms) {
  return {{"schema_version", kSchemaVersion},
          {"event_type", type},
          {"source_id", source_id},
          {"timestamp_ms", timestamp_ms}};
}

}  // namespace

std::uint64_t WallMilliseconds() {
  return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                        std::chrono::system_clock::now().time_since_epoch())
                                        .count());
}

nlohmann::json DetectionMessage(const FrameRecord& frame) {
  nlohmann::json message = Envelope("detection", frame.source_id, frame.wall_ms);
  message["frame_num"] = frame.frame_num;
  message["pts_ns"] = frame.pts_ns;
  if (frame.ntp_ms != 0) message["ntp_ms"] = frame.ntp_ms;
  message["frame_width"] = frame.width;
  message["frame_height"] = frame.height;
  nlohmann::json objects = nlohmann::json::array();
  for (const ObjectRecord& object : frame.objects) {
    const int left = Clamp(object.box.left, frame.width);
    const int top = Clamp(object.box.top, frame.height);
    const int right = Clamp(object.box.left + object.box.width, frame.width);
    const int bottom = Clamp(object.box.top + object.box.height, frame.height);
    if (right <= left || bottom <= top) continue;
    nlohmann::json item = {
        {"class_id", object.class_id},
        {"label", object.label},
        {"confidence", Confidence(object.confidence)},
        {"bbox", {{"x", left}, {"y", top}, {"width", right - left}, {"height", bottom - top}}},
    };
    if (object.track_id) item["track_id"] = *object.track_id;
    if (!object.classes.empty()) {
      nlohmann::json classes = nlohmann::json::array();
      for (const Classification& c : object.classes) {
        classes.push_back({{"component_id", c.component_id},
                           {"class_id", c.class_id},
                           {"label", c.label},
                           {"confidence", Confidence(c.confidence)}});
      }
      item["classes"] = std::move(classes);
    }
    if (!object.extra.empty()) item["extra"] = object.extra;
    objects.push_back(std::move(item));
  }
  message["objects"] = std::move(objects);
  if (!frame.extra.empty()) message["extra"] = frame.extra;
  return message;
}

nlohmann::json RecordingMessage(const RecordingEvent& event) {
  const bool finished = event.type == RecordingEvent::Type::kFinished;
  nlohmann::json message =
      Envelope(finished ? "recording_finished" : "recording_started", event.source_id,
               finished ? event.end_ms : event.start_ms);
  message["branch"] = event.branch;
  message["segment_id"] = event.segment_id;
  message["start_ms"] = event.start_ms;
  if (finished) {
    message["end_ms"] = event.end_ms;
    // 主机墙钟之差，不是媒体时长；时钟回拨时记为 0。
    message["duration_ms"] = event.end_ms > event.start_ms ? event.end_ms - event.start_ms : 0;
    message["detected"] = event.detected;
    message["path"] = event.path;
  }
  return message;
}

std::string Serialize(const nlohmann::json& message) {
  return message.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
}

}  // namespace dsf::ds
