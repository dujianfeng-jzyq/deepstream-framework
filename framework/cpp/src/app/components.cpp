// 标准组件的实现。

#include "dsf/app/components.h"

#include <spdlog/spdlog.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace dsf::app {

std::string ReadyFilePath() {
  const char* value = std::getenv("DSF_READY_FILE");
  return value != nullptr && *value != '\0' ? value : "/tmp/dsf.ready";
}

ReadinessHeartbeat::ReadinessHeartbeat(std::string element, std::function<std::uint64_t()> progress,
                                       std::string path, std::chrono::milliseconds interval)
    : element_(std::move(element)), progress_(std::move(progress)), path_(std::move(path)),
      interval_(interval) {}

std::uint64_t ReadinessHeartbeat::Count() const {
  return progress_ ? progress_() : buffers_.load(std::memory_order_relaxed);
}

void ReadinessHeartbeat::OnCreated(gst::Pipeline& pipeline, GstElement* bin) {
  std::error_code ignored;
  std::filesystem::remove(path_, ignored);  // 上次运行留下的文件不能代表本次就绪
  // 定时器由主循环驱动，不等 PLAYING：按需产帧的管线可能等待首帧才能完成状态切换。
  const bool counting = progress_ || !element_.empty();
  timer_ = pipeline.Every(interval_, [this, counting] {
    const std::uint64_t count = Count();
    if (!counting || count != last_) {
      last_ = count;
      Touch(count);
    }
  });
  if (element_.empty() || progress_) return;
  GstElement* element = gst_bin_get_by_name(GST_BIN(bin), element_.c_str());
  if (element == nullptr) throw std::runtime_error("readiness element '" + element_ + "' is missing");
  pad_ = gst_element_get_static_pad(element, "src");
  gst_object_unref(element);
  if (pad_ == nullptr) throw std::runtime_error("readiness element '" + element_ + "' has no src pad");
  probe_ = gst_pad_add_probe(
      pad_, GST_PAD_PROBE_TYPE_BUFFER,
      +[](GstPad*, GstPadProbeInfo*, gpointer data) -> GstPadProbeReturn {
        static_cast<std::atomic<std::uint64_t>*>(data)->fetch_add(1, std::memory_order_relaxed);
        return GST_PAD_PROBE_OK;
      },
      &buffers_, nullptr);
}

void ReadinessHeartbeat::Touch(std::uint64_t count) {
  std::ofstream stream(path_, std::ios::trunc);
  stream << count << "\n";
}

void ReadinessHeartbeat::OnReleased() {
  timer_.Cancel();
  if (pad_ != nullptr) {
    if (probe_ != 0) gst_pad_remove_probe(pad_, probe_);
    probe_ = 0;
    gst_object_unref(pad_);
    pad_ = nullptr;
  }
  std::error_code ignored;
  std::filesystem::remove(path_, ignored);
}

void ReadinessHeartbeat::Report(gst::Stats& stats) const {
  if (!progress_ && !element_.empty()) stats.Add(element_ + "_buffers", Count());
}

StatsReporter::StatsReporter(std::chrono::seconds interval,
                             std::vector<const gst::Component*> components,
                             std::vector<const publish::Channel*> channels)
    : interval_(interval), components_(std::move(components)), channels_(std::move(channels)) {}

void StatsReporter::OnCreated(gst::Pipeline& pipeline, GstElement* /*bin*/) {
  if (interval_.count() <= 0) return;
  timer_ = pipeline.Every(interval_, [this] { spdlog::info("stats: {}", Line()); });
}

std::string StatsReporter::Line() const {
  gst::Stats stats;
  for (const gst::Component* component : components_) component->Report(stats);
  for (const publish::Channel* channel : channels_) {
    const publish::ChannelStats counts = channel->Stats();
    const std::string prefix = channel->target() + ":";
    stats.Add(prefix + "published", counts.published);
    stats.Add(prefix + "dropped", counts.dropped);
    stats.Add(prefix + "failed", counts.failed);
  }
  return stats.Text();
}

}  // namespace dsf::app
