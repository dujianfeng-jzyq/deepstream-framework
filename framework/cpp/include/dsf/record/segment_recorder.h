// 分段录像：录像器按分段内画面的时长请求 splitmuxsink 在关键帧处切分 MP4，并负责命名、收尾确认与事件通知。
// 分段时长不交给 splitmuxsink 的 max-size-time：它按时间戳计时，直播源断流恢复后会把断流时长算进
// 断流前最后一个 GOP，提前切分，断流前的分段变短（见 RecorderOptions::segment_length）。
//
// 文件布局：<root>/<source_id>/<branch>/<source_id>-<branch>-<起始本地时间>-<序号>[-detected].mp4
// 起始本地时间形如 20261005T163012.345（本地时间与毫秒，不带时区标记），时区取自 /etc/localtime。
// 写入期间使用 .mp4.tmp 后缀；splitmuxsink 报告 fragment-closed 后改名为 .mp4，
// 分段内出现过目标时最终文件名带 -detected；改名成功才通知 finished，消费者因此不会读到未写完的文件。
// 可选的保留策略在启动时与每个分段收尾后清理本目录中的旧分段，见 Retention。
// 录像入口一帧都没有收到时（直播源离线时停止）不把 EOS 交给 splitmuxsink：GStreamer 1.20 的
// splitmuxsink 在没有数据时收到 EOS 会断言失败并终止进程，而此时也没有需要封装的分段。

#ifndef DSF_RECORD_SEGMENT_RECORDER_H
#define DSF_RECORD_SEGMENT_RECORDER_H

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include <gst/gst.h>

#include "dsf/ds/message.h"
#include "dsf/gst/component.h"
#include "dsf/gst/graph.h"
#include "dsf/gst/pipeline.h"
#include "dsf/gst/probe.h"

namespace dsf::record {

struct Segment {
  std::string id;
  std::string temp_path;
  std::string final_path;  // 收尾前不带检测标记；Close 后为实际的最终路径
  std::uint64_t start_ms = 0;
  bool detected = false;
};

// 每个录像器（<root>/<source_id>/<branch>）独立计数的保留策略；全部为 0 表示不清理。
// 已收尾分段从新到旧排列：最新 keep_files 个为保护区；总数超过 max_files 时，在保护区之外
// 先删最旧的无目标分段，再删最旧的有目标分段。剩余空间低于 min_free_bytes 时按同样顺序继续删，
// 并可以进入保护区（写满磁盘会使全部录像失败）。.tmp 与不符合本录像器命名的文件不计数也不删除。
struct Retention {
  std::size_t max_files = 0;        // 0 表示不限数量；非 0 时须 ≥ keep_files
  std::size_t keep_files = 0;
  std::uint64_t min_free_bytes = 0;  // 0 表示不检查剩余空间
};

struct PruneResult {
  int removed = 0;
  int removed_protected = 0;  // 其中因剩余空间不足而删除的保护区分段
  int failed = 0;
  bool low_space = false;      // 已无可删分段，剩余空间仍不足
  std::vector<std::string> errors;
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
  // 按保留策略删除本目录的已收尾分段。available 返回剩余字节数，为空时查询文件系统（测试可替换）。
  // 只读取不可变成员，调用方不必持有登记表的锁。
  PruneResult Prune(const Retention& retention,
                    const std::function<std::uint64_t()>& available = {}) const;
  void MarkDetected();

  std::size_t open_count() const { return open_.size(); }
  std::uint64_t closed_count() const { return closed_; }

 private:
  std::string root_;
  std::string source_id_;
  std::string branch_;
  std::string FinalPath(const std::string& id, bool detected) const;

  std::map<std::string, Segment> open_;  // temp_path → 分段
  std::uint64_t sequence_ = 0;
  std::uint64_t closed_ = 0;
};

struct RecorderOptions {
  std::string root;       // 可写目录，例如 /data/recordings
  std::string source_id;
  std::string branch;     // 例如 raw / osd
  std::string element;    // AddSegmentSink 创建的 splitmuxsink 名
  // 分段内画面的时长达到这个值后，在下一个关键帧处开始新分段；分段因此比它长不到一个关键帧间隔。
  // 断流切分（split_on_gap）后的分段从恢复后的第一个关键帧重新计时，断流时长不计入。必须为正。
  std::chrono::milliseconds segment_length{60000};
  // 停止时，管线注入的 EOS 在这段时间内仍未到达录像入口，就直接向录像入口发送 EOS。
  // 必须小于管线的 stop_timeout，否则兜底来不及生效。0 表示不兜底。
  std::chrono::milliseconds eos_fallback{3000};
  Retention retention{};  // 默认不清理
  // 录像入口相邻两帧的时间戳相差超过这个值（直播源断流后恢复）时，从恢复后的第一个关键帧开始
  // 新分段：断流前的分段只含断流前的画面，不把断流时长计入分段，也不在文件中留下长时间的静止。
  // 恢复后、第一个关键帧之前的帧不写入录像。编码器在断流处给出关键帧（app::KeyframeOnGap）时
  // 不丢或只丢一帧（NVIDIA 编码器晚一帧生效），否则最多丢一个关键帧间隔。0 表示不切分。
  std::chrono::milliseconds split_on_gap{2000};
};

class SegmentRecorder : public gst::Component {
 public:
  struct Events {
    std::function<void(const Segment&)> started;  // 媒体线程
    // end_ms = start_ms + 分段内画面的时长（splitmuxsink 报告的起止 running-time 之差），
    // 不是收尾时的墙钟：分段可能在断流恢复后才收尾。得不到时长时退回收尾时的墙钟。控制线程。
    std::function<void(const Segment&, std::uint64_t end_ms)> finished;
  };

  // segment_length 不为正，或 retention.max_files 非 0 且小于 keep_files 时抛 std::invalid_argument。
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
  static GstPadProbeReturn OnInput(GstPad* pad, GstPadProbeInfo* info, gpointer self);
  static GstPadProbeReturn OnSinkData(GstPad* pad, GstPadProbeInfo* info, gpointer self);
  void Prune();  // 控制线程

  RecorderOptions options_;
  Events events_;
  std::string name_;
  mutable std::mutex mutex_;  // format-location 在流线程，fragment-closed 在控制线程
  SegmentTracker tracker_;
  std::uint64_t failures_ = 0;
  std::uint64_t pruned_ = 0;
  std::uint64_t gap_splits_ = 0;
  std::map<std::string, GstClockTime> opened_at_;  // 临时路径 → 分段起点的 running-time
  // 断流前最后一帧结束时的 running-time。fragment-closed 报告的是下一分段的起点（含断流），
  // 落在分段起止之间的断流以这里为分段的结束。
  std::vector<GstClockTime> gap_ends_;
  GstElement* sink_ = nullptr;
  gulong handler_ = 0;
  gst::PadProbe input_;  // 录像入口：按时长与断流请求切分
  GstClockTime last_input_ = GST_CLOCK_TIME_NONE;  // 以下五项只在流线程访问
  GstClockTime last_duration_ = GST_CLOCK_TIME_NONE;
  GstClockTime segment_begin_ = GST_CLOCK_TIME_NONE;  // 当前分段第一个关键帧的 running-time
  bool awaiting_keyframe_ = false;  // 断流后丢弃关键帧之前的帧
  std::uint64_t skipped_ = 0;
  // 挂在录像入口（ghost pad）的内部 pad 上：管线在 ghost pad 上记录 EOS 到达，之后才轮到这里，
  // 丢弃无数据的 EOS 不会让排空等到超时。received_ 由流线程与兜底 EOS 的控制线程访问。
  gst::PadProbe sink_data_;
  std::atomic<bool> received_{false};
  gst::Pipeline* pipeline_ = nullptr;  // OnCreated 到 OnReleased 之间有效
  gst::Timer fallback_;
};

// 分段事件对应的消息事件（dsf::ds::RecordingMessage 的输入）；source_id 与 branch 取自 options。
ds::RecordingEvent RecordingStarted(const RecorderOptions& options, const Segment& segment);
ds::RecordingEvent RecordingFinished(const RecorderOptions& options, const Segment& segment, std::uint64_t end_ms);

// encoded（已编码的 H.264/H.265 流）→ parser → splitmuxsink(mp4mux)。
// 返回 splitmuxsink；properties 透传给 splitmuxsink。块本身不设分段时长：切分由登记在它上面的
// SegmentRecorder 发出（RecorderOptions::segment_length），没有录像器时整段录成一个文件。
gst::Element AddSegmentSink(gst::Graph& graph, const gst::Endpoint& encoded, const std::string& codec,
                            const std::string& name, const config::Properties& properties = {});

}  // namespace dsf::record

#endif  // DSF_RECORD_SEGMENT_RECORDER_H
