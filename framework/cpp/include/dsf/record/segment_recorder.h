// 分段录像：splitmuxsink 按时长切分 MP4，框架负责命名、收尾确认与事件通知。
//
// 文件布局：<root>/<source_id>/<branch>/<source_id>-<branch>-<起始本地时间>-<序号>.mp4
// 起始本地时间形如 20261005T163012.345（本地时间与毫秒，不带时区标记），时区取自 /etc/localtime。
// 写入期间使用 .mp4.tmp 后缀；splitmuxsink 报告 fragment-closed 后改名为 .mp4，
// 改名成功才通知 finished，消费者因此不会读到未写完的文件。

#ifndef DSF_RECORD_SEGMENT_RECORDER_H
#define DSF_RECORD_SEGMENT_RECORDER_H

#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <string>

#include <gst/gst.h>

#include "dsf/gst/component.h"
#include "dsf/gst/graph.h"
#include "dsf/gst/pipeline.h"

namespace dsf::record {

struct Segment {
  std::string id;
  std::string temp_path;
  std::string final_path;
  std::uint64_t start_ms = 0;
  bool detected = false;
};

// 分段登记的纯逻辑（不依赖 GStreamer），单独可测。非线程安全，由 SegmentRecorder 加锁。
class SegmentTracker {
 public:
  // source_id 与 branch 必须是合法路径段，否则抛 std::invalid_argument。
  SegmentTracker(std::string root, std::string source_id, std::string branch);

  std::string Directory() const;
  // 分配新分段；创建目录；跳过已存在的文件名（重启或时钟回拨不会覆盖旧录像）。
  Segment Open(std::uint64_t start_ms);
  // 按临时路径关闭；不是本登记表分配的路径返回 false。
  bool Close(const std::string& temp_path, Segment* closed);
  // .tmp → .mp4；失败返回 false 并写入原因。
  static bool Finalize(const Segment& segment, std::string* error);
  // 删除本目录遗留的 .tmp（上次运行未收尾的分段），返回删除数。只能在开始录像前调用。
  int RemoveStale() const;
  void MarkDetected();

  std::size_t open_count() const { return open_.size(); }
  std::uint64_t closed_count() const { return closed_; }

 private:
  std::string root_;
  std::string source_id_;
  std::string branch_;
  std::map<std::string, Segment> open_;  // temp_path → 分段
  std::uint64_t sequence_ = 0;
  std::uint64_t closed_ = 0;
};

struct RecorderOptions {
  std::string root;       // 可写目录，例如 /data/recordings
  std::string source_id;
  std::string branch;     // 例如 raw / osd
  std::string element;    // AddSegmentSink 创建的 splitmuxsink 名
  // 停止时，管线注入的 EOS 在这段时间内仍未到达录像入口，就直接向录像入口发送 EOS。
  // 必须小于管线的 stop_timeout，否则兜底来不及生效。0 表示不兜底。
  std::chrono::milliseconds eos_fallback{3000};
};

class SegmentRecorder : public gst::Component {
 public:
  struct Events {
    std::function<void(const Segment&)> started;                       // 媒体线程
    std::function<void(const Segment&, std::uint64_t end_ms)> finished;  // 控制线程
  };

  SegmentRecorder(RecorderOptions options, Events events);
  ~SegmentRecorder() override;

  // 当前打开的分段标记为“出现过目标”；任意线程。
  void MarkDetected();

  std::string_view Name() const override { return name_; }
  void Require(gst::Requirements& requirements) const override;
  void OnCreated(gst::Pipeline& pipeline, GstElement* bin) override;
  void OnMessage(GstMessage* message) override;
  void OnStopping() override;
  void OnReleased() override;
  bool Drained() const override { return open_count() == 0; }
  bool Healthy() const override;
  void Report(gst::Stats& stats) const override;

  std::uint64_t failures() const;
  std::size_t open_count() const;

 private:
  static gchar* OnFormatLocation(GstElement* sink, guint fragment, gpointer self);

  RecorderOptions options_;
  Events events_;
  std::string name_;
  mutable std::mutex mutex_;  // format-location 在流线程，fragment-closed 在控制线程
  SegmentTracker tracker_;
  std::uint64_t failures_ = 0;
  GstElement* sink_ = nullptr;
  gulong handler_ = 0;
  gst::Pipeline* pipeline_ = nullptr;  // OnCreated 到 OnReleased 之间有效
  gst::Timer fallback_;
};

// encoded（已编码的 H.264/H.265 流）→ parser → splitmuxsink(mp4mux)。
// 返回 splitmuxsink；properties 透传给 splitmuxsink。
gst::Element AddSegmentSink(gst::Graph& graph, const gst::Endpoint& encoded,
                            const std::string& codec, int segment_seconds, const std::string& name,
                            const config::Properties& properties = {});

}  // namespace dsf::record

#endif  // DSF_RECORD_SEGMENT_RECORDER_H
