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

// 就绪心跳：计数增长时才更新就绪文件（证明输入真的在供帧）。计数来源二选一：
// progress（例如 MetaTap::frames()），或经过 element 的 buffer 数；都为空时只要主循环在运行就每秒
// 更新，不等 PLAYING（按需产帧的管线可能等待首帧才能完成状态切换）。结束时删除文件。
// stall_timeout 为正且有计数来源时兼作看门狗：计数在时限内没有增长就请求排空停止，Healthy() 为 false。
class ReadinessHeartbeat : public gst::Component {
 public:
  ReadinessHeartbeat(std::string element, std::function<std::uint64_t()> progress, std::string path,
                     std::chrono::milliseconds interval = std::chrono::seconds(1),
                     std::chrono::milliseconds stall_timeout = std::chrono::milliseconds(0));
  std::string_view Name() const override { return "readiness"; }
  void OnCreated(gst::Pipeline& pipeline, GstElement* bin) override;
  void OnReleased() override;
  bool Healthy() const override { return !stalled_; }
  void Report(gst::Stats& stats) const override;

 private:
  void Touch(std::uint64_t count);
  std::uint64_t Count() const;
  std::string element_;
  std::function<std::uint64_t()> progress_;
  std::string path_;
  std::chrono::milliseconds interval_;
  std::chrono::milliseconds stall_timeout_;
  std::chrono::steady_clock::time_point last_progress_;
  bool stalled_ = false;
  GstPad* pad_ = nullptr;
  gulong probe_ = 0;
  std::atomic<std::uint64_t> buffers_{0};
  std::uint64_t last_ = 0;
  gst::Timer timer_;
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
