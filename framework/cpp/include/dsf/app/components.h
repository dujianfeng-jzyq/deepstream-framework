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
#include "dsf/publish/publisher.h"

namespace dsf::app {

// 默认就绪文件，Compose 的公共 healthcheck 检查它的更新时间。可用 DSF_READY_FILE 覆盖。
std::string ReadyFilePath();

// 就绪心跳：按 interval 更新就绪文件，不等 PLAYING（按需产帧的管线可能等待首帧才能完成状态切换）。
// 计数来源二选一：progress（例如 MetaTap::frames()），或经过 element 的 buffer 数。
// require_input 为 true 时只在计数增长时更新（就绪 = 输入真的在供帧）；为 false 时主循环运行即更新，
// 没有输入也算就绪（线阵、外触发等按需产帧的源）。有计数来源时文件内容为 "count=<计数> idle_s=<距上次
// 增长的秒数>"，否则为 "alive"；健康检查只看更新时间。结束时删除文件。
// stall_timeout 为正且有计数来源时兼作看门狗：计数在时限内没有增长就请求排空停止、不再更新文件，
// Healthy() 为 false。与 require_input 无关：按需产帧的源应关闭看门狗或给出足够长的时限。
class ReadinessHeartbeat : public gst::Component {
 public:
  ReadinessHeartbeat(std::string element, std::function<std::uint64_t()> progress, std::string path,
                     std::chrono::milliseconds interval = std::chrono::seconds(1),
                     std::chrono::milliseconds stall_timeout = std::chrono::milliseconds(0),
                     bool require_input = true);
  std::string_view Name() const override { return "readiness"; }
  void OnCreated(gst::Pipeline& pipeline, GstElement* bin) override;
  void OnReleased() override;
  bool Healthy() const override { return !stalled_; }
  void Report(gst::Stats& stats) const override;

 private:
  void Touch(bool counting, std::uint64_t count, std::chrono::steady_clock::time_point now);
  std::uint64_t Count() const;
  std::string element_;
  std::function<std::uint64_t()> progress_;
  std::string path_;
  std::chrono::milliseconds interval_;
  std::chrono::milliseconds stall_timeout_;
  bool require_input_;
  std::chrono::steady_clock::time_point last_progress_;
  bool stalled_ = false;
  GstPad* pad_ = nullptr;
  gulong probe_ = 0;
  std::atomic<std::uint64_t> buffers_{0};
  std::uint64_t last_ = 0;
  gst::Timer timer_;
};

// 断流后的关键帧：直播源断流重连后，下一帧的时间戳比上一帧晚整个断流时长，编码器却照常延续当前
// GOP。挂在编码器（element）的输入上，相邻两帧的时间戳相差超过 gap 时，在这一帧之前向编码器发送
// GstForceKeyUnit，使它从关键帧开始：录像器据此在断流处干净地切分分段（见 RecorderOptions），
// RTSP 观看端也不必等到下一个周期关键帧才能恢复画面。编码器须支持该事件（GstVideoEncoder 子类都支持）。
class KeyframeOnGap : public gst::Component {
 public:
  KeyframeOnGap(std::string element, std::chrono::milliseconds gap);
  std::string_view Name() const override { return name_; }
  void OnCreated(gst::Pipeline& pipeline, GstElement* bin) override;
  void OnReleased() override;
  void Report(gst::Stats& stats) const override;

 private:
  static GstPadProbeReturn OnBuffer(GstPad* pad, GstPadProbeInfo* info, gpointer self);
  std::string element_;
  std::string name_;
  GstClockTime gap_;
  GstClockTime last_ = GST_CLOCK_TIME_NONE;  // 只在流线程访问
  std::atomic<std::uint64_t> forced_{0};
  GstPad* pad_ = nullptr;
  gulong probe_ = 0;
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
