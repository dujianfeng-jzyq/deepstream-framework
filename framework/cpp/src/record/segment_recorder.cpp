// 分段录像的实现。

#include "dsf/record/segment_recorder.h"

#include <spdlog/spdlog.h>

#include <filesystem>
#include <stdexcept>
#include <utility>

#include "dsf/ds/meta.h"
#include "dsf/ds/blocks.h"

namespace dsf::record {
namespace fs = std::filesystem;

namespace {

bool IsPathComponent(const std::string& value) {
  return !value.empty() && value != "." && value != ".." &&
         value.find_first_of("/\\") == std::string::npos;
}

}  // namespace

// ---- SegmentTracker ----

SegmentTracker::SegmentTracker(std::string root, std::string source_id, std::string branch)
    : root_(std::move(root)), source_id_(std::move(source_id)), branch_(std::move(branch)) {
  if (root_.empty() || !IsPathComponent(source_id_) || !IsPathComponent(branch_)) {
    throw std::invalid_argument("recording root, source id and branch must be valid paths");
  }
}

std::string SegmentTracker::Directory() const {
  return (fs::path(root_) / source_id_ / branch_).string();
}

Segment SegmentTracker::Open(std::uint64_t start_ms) {
  fs::create_directories(Directory());
  Segment segment;
  segment.start_ms = start_ms;
  do {
    segment.id = source_id_ + "-" + branch_ + "-" + std::to_string(start_ms) + "-" +
                 std::to_string(sequence_++);
    segment.final_path = (fs::path(Directory()) / (segment.id + ".mp4")).string();
    segment.temp_path = segment.final_path + ".tmp";
  } while (open_.count(segment.temp_path) != 0 || fs::exists(segment.final_path) ||
           fs::exists(segment.temp_path));
  open_[segment.temp_path] = segment;
  return segment;
}

bool SegmentTracker::Close(const std::string& temp_path, Segment* closed) {
  const auto found = open_.find(temp_path);
  if (found == open_.end()) return false;
  *closed = found->second;
  open_.erase(found);
  ++closed_;
  return true;
}

bool SegmentTracker::Finalize(const Segment& segment, std::string* error) {
  std::error_code code;
  fs::rename(segment.temp_path, segment.final_path, code);
  if (code) {
    *error = "cannot rename " + segment.temp_path + ": " + code.message();
    return false;
  }
  return true;
}

int SegmentTracker::RemoveStale() const {
  std::error_code code;
  int removed = 0;
  for (const auto& entry : fs::directory_iterator(Directory(), code)) {
    if (entry.path().extension() == ".tmp" && fs::remove(entry.path(), code)) ++removed;
  }
  return removed;
}

void SegmentTracker::MarkDetected() {
  for (auto& [path, segment] : open_) segment.detected = true;
}

// ---- SegmentRecorder ----

SegmentRecorder::SegmentRecorder(RecorderOptions options, Events events)
    : options_(std::move(options)),
      events_(std::move(events)),
      name_("recorder:" + options_.element),
      tracker_(options_.root, options_.source_id, options_.branch) {}

SegmentRecorder::~SegmentRecorder() { OnReleased(); }

void SegmentRecorder::MarkDetected() {
  std::lock_guard<std::mutex> lock(mutex_);
  tracker_.MarkDetected();
}

void SegmentRecorder::Require(gst::Requirements& requirements) const {
  requirements.Signal("splitmuxsink", "format-location", options_.element);
}

void SegmentRecorder::OnCreated(gst::Pipeline& pipeline, GstElement* bin) {
  pipeline_ = &pipeline;
  sink_ = gst_bin_get_by_name(GST_BIN(bin), options_.element.c_str());
  if (sink_ == nullptr) {
    throw std::runtime_error("recording element '" + options_.element + "' is not in the pipeline");
  }
  // 遗留的 .tmp 来自上次没有收尾的运行，不能当作有效录像。
  if (const int stale = tracker_.RemoveStale(); stale > 0) {
    spdlog::warn("removed {} unfinished segment(s) from {}", stale, tracker_.Directory());
  }
  handler_ = g_signal_connect(sink_, "format-location", G_CALLBACK(&SegmentRecorder::OnFormatLocation),
                              this);
}

void SegmentRecorder::OnReleased() {
  fallback_.Cancel();
  pipeline_ = nullptr;
  if (sink_ == nullptr) return;
  if (handler_ != 0) g_signal_handler_disconnect(sink_, handler_);
  handler_ = 0;
  gst_object_unref(sink_);
  sink_ = nullptr;
}

gchar* SegmentRecorder::OnFormatLocation(GstElement* /*sink*/, guint /*fragment*/, gpointer self) {
  auto* recorder = static_cast<SegmentRecorder*>(self);
  try {
    Segment segment;
    {
      std::lock_guard<std::mutex> lock(recorder->mutex_);
      segment = recorder->tracker_.Open(ds::WallMilliseconds());
    }
    if (recorder->events_.started) recorder->events_.started(segment);
    return g_strdup(segment.temp_path.c_str());  // splitmuxsink 接管所有权
  } catch (const std::exception& error) {
    {
      std::lock_guard<std::mutex> lock(recorder->mutex_);
      ++recorder->failures_;
    }
    spdlog::error("cannot allocate a recording segment for {}: {}", recorder->options_.element,
                  error.what());
    return nullptr;
  }
}

void SegmentRecorder::OnMessage(GstMessage* message) {
  if (sink_ == nullptr || GST_MESSAGE_TYPE(message) != GST_MESSAGE_ELEMENT ||
      GST_MESSAGE_SRC(message) != GST_OBJECT_CAST(sink_)) {
    return;
  }
  const GstStructure* structure = gst_message_get_structure(message);
  if (structure == nullptr || !gst_structure_has_name(structure, "splitmuxsink-fragment-closed")) {
    return;
  }
  const gchar* location = gst_structure_get_string(structure, "location");
  if (location == nullptr) return;

  Segment segment;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!tracker_.Close(location, &segment)) {
      spdlog::warn("{}: closed a segment that was not allocated here: {}", options_.element, location);
      return;
    }
  }
  std::string error;
  if (!SegmentTracker::Finalize(segment, &error)) {
    std::lock_guard<std::mutex> lock(mutex_);
    ++failures_;
    spdlog::error("{}: {}", options_.element, error);
    return;
  }
  spdlog::info("recording finished: {} detected={}", segment.final_path, segment.detected);
  if (events_.finished) events_.finished(segment, ds::WallMilliseconds());
}

void SegmentRecorder::OnStopping() {
  // 不在这里立即向 splitmuxsink 发送 EOS：停止时管线在每个源之后注入 EOS，它跟在最后一帧后面
  // 依次流经编码器、parser 与队列。提前在录像入口发 EOS 会让分段立即封装，上游仍在处理的帧
  // （编码器内部、tee 分支的队列）随后到达时被丢弃。排空完成以 fragment-closed 为准（见 Drained）。
  //
  // 兜底：EOS 在 eos_fallback 内仍未到达录像入口（某个上游元件不传递 EOS）时，直接向录像入口
  // 发送 EOS，封装已经收到的帧。录像入口已收到 EOS 时重复发送不会产生影响。
  if (sink_ == nullptr || pipeline_ == nullptr || options_.eos_fallback.count() <= 0) return;
  fallback_ = pipeline_->Every(options_.eos_fallback, [this] {
    fallback_.Cancel();
    if (sink_ == nullptr || Drained()) return;
    spdlog::warn("{}: end of stream did not reach the recorder within {} ms; closing the segment",
                 options_.element, options_.eos_fallback.count());
    GstIterator* pads = gst_element_iterate_sink_pads(sink_);
    GValue item = G_VALUE_INIT;
    while (gst_iterator_next(pads, &item) == GST_ITERATOR_OK) {
      gst_pad_send_event(GST_PAD(g_value_get_object(&item)), gst_event_new_eos());
      g_value_reset(&item);
    }
    g_value_unset(&item);
    gst_iterator_free(pads);
  });
}

bool SegmentRecorder::Healthy() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return failures_ == 0 && tracker_.open_count() == 0;
}

void SegmentRecorder::Report(gst::Stats& stats) const {
  std::lock_guard<std::mutex> lock(mutex_);
  stats.Add(options_.element + "_segments", tracker_.closed_count());
  if (failures_ > 0) stats.Add(options_.element + "_failures", failures_);
}

std::uint64_t SegmentRecorder::failures() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return failures_;
}

std::size_t SegmentRecorder::open_count() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return tracker_.open_count();
}

gst::Element AddSegmentSink(gst::Graph& graph, const gst::Endpoint& encoded,
                            const std::string& codec, int segment_seconds, const std::string& name,
                            const config::Properties& properties) {
  if (segment_seconds <= 0) throw std::invalid_argument("segment length must be positive");
  // MP4 封装需要 avc/hvc1 格式的码流，parser 负责从编码器的 byte-stream 转换。
  gst::Element parser = ds::AddParser(graph, encoded, codec, name + "_parse", false);
  gst::Element sink = graph.Add("splitmuxsink", name);
  sink.Set("muxer-factory", "mp4mux")
      .Set("max-size-time", static_cast<std::uint64_t>(segment_seconds) * std::uint64_t{1000000000})
      .Apply(properties);
  graph.Link(parser, sink.Pad("video"));
  return sink;
}

}  // namespace dsf::record
