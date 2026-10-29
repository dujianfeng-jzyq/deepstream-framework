// 框架提供的标准组件。

#ifndef DSF_APP_COMPONENTS_H
#define DSF_APP_COMPONENTS_H

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "dsf/gst/component.h"
#include "dsf/gst/pipeline.h"
#include "dsf/gst/probe.h"
#include "dsf/publish/publisher.h"

namespace dsf::app {

// 默认就绪文件，Compose 的公共 healthcheck 检查它的更新时间。可用 DSF_READY_FILE 覆盖。
std::string ReadyFilePath();

// 就绪心跳：按 interval 更新就绪文件，不等 PLAYING（按需产帧的管线可能等待首帧才能完成状态切换）。
// progress 是单调递增的处理计数（例如 MetaTap::frames()、BufferCounter::count()），为空表示没有计数。
// require_input 为 true 时只在计数增长时更新（就绪 = 输入真的在供帧）；为 false 时主循环运行即更新，
// 没有输入也算就绪（线阵、外触发等按需产帧的源）。有计数时文件内容为 "count=<计数> idle_s=<距上次
// 增长的秒数>"，否则为 "alive"；健康检查只看更新时间。结束时删除文件。
// stall_timeout 为正且有计数时兼作看门狗：计数在时限内没有增长就请求排空停止、不再更新文件，
// Healthy() 为 false。与 require_input 无关：按需产帧的源应关闭看门狗或给出足够长的时限。
// 默认值与配置段 readiness / watchdog 的默认值一致。
class ReadinessHeartbeat : public gst::Component {
 public:
  ReadinessHeartbeat(std::function<std::uint64_t()> progress, std::string path,
                     std::chrono::milliseconds interval = std::chrono::seconds(1),
                     std::chrono::milliseconds stall_timeout = std::chrono::milliseconds(0),
                     bool require_input = false);
  std::string_view Name() const override { return "readiness"; }
  void OnCreated(gst::Pipeline& pipeline, GstElement* bin) override;
  void OnReleased() override;
  bool Healthy() const override { return !stalled_; }

 private:
  void Touch(std::uint64_t count, std::chrono::steady_clock::time_point now);
  std::function<std::uint64_t()> progress_;
  std::string path_;
  std::chrono::milliseconds interval_;
  std::chrono::milliseconds stall_timeout_;
  bool require_input_;
  std::chrono::steady_clock::time_point last_progress_;
  bool stalled_ = false;
  std::uint64_t last_ = 0;
  gst::Timer timer_;
};

// 经过 element 的 src pad 的 buffer 数（Assembly::ReadyWhenFramesReach 的计数来源），统计 <element>_buffers。
class BufferCounter : public gst::Component {
 public:
  explicit BufferCounter(std::string element);
  std::string_view Name() const override { return name_; }
  void OnCreated(gst::Pipeline& pipeline, GstElement* bin) override;
  void OnReleased() override { probe_.Remove(); }
  void Report(gst::Stats& stats) const override;

  std::uint64_t count() const { return count_.load(std::memory_order_relaxed); }

 private:
  std::string element_;
  std::string name_;
  std::atomic<std::uint64_t> count_{0};
  gst::PadProbe probe_;
};

// 断流后的关键帧：直播源断流重连后，下一帧的时间戳比上一帧晚整个断流时长，编码器却照常延续当前
// GOP。挂在编码器（element）的输入上，相邻两帧的时间戳相差超过 gap 时，在这一帧之前向编码器发送
// 关键帧请求，使它从关键帧开始：录像器据此在断流处干净地切分分段（见 RecorderOptions），
// RTSP 观看端也不必等到下一个周期关键帧才能恢复画面。请求同时以三种方式发出，编码器响应其中之一即可：
// 标准的 GstForceKeyUnit 事件、DeepStream 的 nv-enc-force-idr 事件（dGPU 的 nvv4l2h26xenc），以及
// 编码器的 force-IDR 动作信号（Jetson 的 nvv4l2h26xenc，它不响应前两种事件；存在时才发出）。
class KeyframeOnGap : public gst::Component {
 public:
  KeyframeOnGap(std::string element, std::chrono::milliseconds gap);
  std::string_view Name() const override { return name_; }
  void OnCreated(gst::Pipeline& pipeline, GstElement* bin) override;
  void OnReleased() override { probe_.Remove(); }
  void Report(gst::Stats& stats) const override;

 private:
  static GstPadProbeReturn OnBuffer(GstPad* pad, GstPadProbeInfo* info, gpointer self);
  std::string element_;
  std::string name_;
  GstClockTime gap_;
  GstClockTime last_ = GST_CLOCK_TIME_NONE;  // 只在流线程访问
  std::atomic<std::uint64_t> forced_{0};
  bool force_idr_signal_ = false;  // OnCreated 中确定（流线程尚未启动），之后只读
  gst::PadProbe probe_;
};

// 周期统计：汇总全部组件的 Report() 与各发布通道的计数写一行日志。主循环运行即开始，不等 PLAYING。
class StatsReporter : public gst::Component {
 public:
  StatsReporter(std::chrono::seconds interval, std::vector<const gst::Component*> components,
                std::vector<const publish::Channel*> channels);
  std::string_view Name() const override { return "stats"; }
  void OnCreated(gst::Pipeline& pipeline, GstElement* bin) override;
  void OnReleased() override { timer_.Cancel(); }
  std::string Line() const;

 private:
  std::chrono::seconds interval_;
  std::vector<const gst::Component*> components_;
  std::vector<const publish::Channel*> channels_;
  gst::Timer timer_;
};

}  // namespace dsf::app

#endif  // DSF_APP_COMPONENTS_H
