// 标准组件的实现。

#include "dsf/app/components.h"

#include <spdlog/spdlog.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace dsf::app {

namespace {

// gst-nvcustomevent.h 的 GST_NVEVENT_ENC_FORCE_IDR。
constexpr auto kNvEncForceIdr = static_cast<GstEventType>(
    GST_EVENT_MAKE_TYPE(504, GST_EVENT_TYPE_DOWNSTREAM | GST_EVENT_TYPE_SERIALIZED));

}  // namespace

std::string ReadyFilePath() {
  const char* value = std::getenv("DSF_READY_FILE");
  return value != nullptr && *value != '\0' ? value : "/tmp/dsf.ready";
}

ReadinessHeartbeat::ReadinessHeartbeat(std::string element, std::function<std::uint64_t()> progress,
                                       std::string path, std::chrono::milliseconds interval,
                                       std::chrono::milliseconds stall_timeout, bool require_input)
    : element_(std::move(element)), progress_(std::move(progress)), path_(std::move(path)),
      interval_(interval), stall_timeout_(stall_timeout), require_input_(require_input) {}

std::uint64_t ReadinessHeartbeat::Count() const {
  return progress_ ? progress_() : buffers_.load(std::memory_order_relaxed);
}

void ReadinessHeartbeat::OnCreated(gst::Pipeline& pipeline, GstElement* bin) {
  std::error_code ignored;
  std::filesystem::remove(path_, ignored);  // 上次运行留下的文件不能代表本次就绪
  // 定时器由主循环驱动，不等 PLAYING：按需产帧的管线可能等待首帧才能完成状态切换。
  const bool counting = progress_ || !element_.empty();
  last_progress_ = std::chrono::steady_clock::now();
  timer_ = pipeline.Every(interval_, [this, counting, &pipeline] {
    const std::uint64_t count = Count();
    const auto now = std::chrono::steady_clock::now();
    const bool advanced = !counting || count != last_;
    if (advanced) {
      last_ = count;
      last_progress_ = now;
    } else if (stall_timeout_.count() > 0 && !stalled_ && now - last_progress_ >= stall_timeout_) {
      // 排空而不是立即结束：录像等组件仍能收尾；排空超时由管线的 stop_timeout 兜底。
      stalled_ = true;
      spdlog::error("no progress for {} s (count={}); stopping so the container restarts",
                    std::chrono::duration_cast<std::chrono::seconds>(stall_timeout_).count(), count);
      pipeline.RequestStop();
    }
    if (!stalled_ && (advanced || !require_input_)) Touch(counting, count, now);
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

void ReadinessHeartbeat::Touch(bool counting, std::uint64_t count, std::chrono::steady_clock::time_point now) {
  std::ofstream stream(path_, std::ios::trunc);
  if (!counting) {
    stream << "alive\n";
    return;
  }
  stream << "count=" << count
         << " idle_s=" << std::chrono::duration_cast<std::chrono::seconds>(now - last_progress_).count() << "\n";
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

KeyframeOnGap::KeyframeOnGap(std::string element, std::chrono::milliseconds gap)
    : element_(std::move(element)),
      name_("keyframe-on-gap:" + element_),
      gap_(static_cast<GstClockTime>(std::chrono::nanoseconds(gap).count())) {
  if (gap.count() <= 0) throw std::invalid_argument("keyframe gap must be positive");
}

void KeyframeOnGap::OnCreated(gst::Pipeline& /*pipeline*/, GstElement* bin) {
  GstElement* element = gst_bin_get_by_name(GST_BIN(bin), element_.c_str());
  if (element == nullptr) throw std::runtime_error("encoder '" + element_ + "' is not in the pipeline");
  pad_ = gst_element_get_static_pad(element, "sink");
  gst_object_unref(element);
  if (pad_ == nullptr) throw std::runtime_error("encoder '" + element_ + "' has no sink pad");
  probe_ = gst_pad_add_probe(pad_, GST_PAD_PROBE_TYPE_BUFFER, &KeyframeOnGap::OnBuffer, this, nullptr);
}

GstPadProbeReturn KeyframeOnGap::OnBuffer(GstPad* pad, GstPadProbeInfo* info, gpointer self) {
  auto* component = static_cast<KeyframeOnGap*>(self);
  const GstClockTime pts = GST_BUFFER_PTS(GST_PAD_PROBE_INFO_BUFFER(info));
  if (!GST_CLOCK_TIME_IS_VALID(pts)) return GST_PAD_PROBE_OK;
  const GstClockTime last = component->last_;
  component->last_ = pts;
  if (!GST_CLOCK_TIME_IS_VALID(last) || pts <= last || pts - last <= component->gap_) {
    return GST_PAD_PROBE_OK;
  }
  // 编码器按 running-time 匹配请求：取这一帧的 running-time，使它本身成为关键帧。
  GstClockTime running = pts;
  GstClockTime stream = pts;
  if (GstEvent* event = gst_pad_get_sticky_event(pad, GST_EVENT_SEGMENT, 0)) {
    const GstSegment* segment = nullptr;
    gst_event_parse_segment(event, &segment);
    running = gst_segment_to_running_time(segment, GST_FORMAT_TIME, pts);
    stream = gst_segment_to_stream_time(segment, GST_FORMAT_TIME, pts);
    gst_event_unref(event);
  }
  const std::uint64_t count = component->forced_.fetch_add(1, std::memory_order_relaxed) + 1;
  // 与 gst_video_event_new_downstream_force_key_unit 构造的事件相同，不必为此链接 gstreamer-video。
  GstStructure* request = gst_structure_new(
      "GstForceKeyUnit", "timestamp", G_TYPE_UINT64, pts, "stream-time", G_TYPE_UINT64, stream,
      "running-time", G_TYPE_UINT64, running, "all-headers", G_TYPE_BOOLEAN, TRUE, "count", G_TYPE_UINT,
      static_cast<guint>(count), nullptr);
  // 在流线程中、这一帧进入编码器之前送达：编码器先登记请求，再处理这一帧。
  gst_pad_send_event(pad, gst_event_new_custom(GST_EVENT_CUSTOM_DOWNSTREAM, request));
  // NVIDIA 的 nvv4l2h26xenc 不响应上面的标准请求（dGPU 上实测），只认自定义事件 nv-enc-force-idr。
  // 事件类型与结构照 DeepStream 的 gst-nvcustomevent.h 构造，不必为此链接 SDK；其他编码器把它
  // 当作未知的串行事件向下游转发，不受影响。
  gchar* stream_id = gst_pad_get_stream_id(pad);
  GstStructure* idr = gst_structure_new("nv-enc-force-idr", "stream_id", G_TYPE_STRING,
                                        stream_id != nullptr ? stream_id : "", "force", G_TYPE_INT, 1,
                                        nullptr);
  g_free(stream_id);
  gst_pad_send_event(pad, gst_event_new_custom(kNvEncForceIdr, idr));
  spdlog::info("{}: input resumed after {:.1f} s; requested a keyframe", component->element_,
               static_cast<double>(pts - last) / GST_SECOND);
  return GST_PAD_PROBE_OK;
}

void KeyframeOnGap::OnReleased() {
  if (pad_ == nullptr) return;
  if (probe_ != 0) gst_pad_remove_probe(pad_, probe_);
  probe_ = 0;
  gst_object_unref(pad_);
  pad_ = nullptr;
}

void KeyframeOnGap::Report(gst::Stats& stats) const {
  stats.Add(element_ + "_gap_keyframes", forced_.load(std::memory_order_relaxed));
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
