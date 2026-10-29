// 分段录像的实现。

#include "dsf/record/segment_recorder.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <limits>
#include <regex>
#include <stdexcept>
#include <utility>
#include <vector>

#include "dsf/ds/meta.h"
#include "dsf/ds/blocks.h"

namespace dsf::record {
namespace fs = std::filesystem;

namespace {

bool IsPathComponent(const std::string& value) {
  return !value.empty() && value != "." && value != ".." &&
         value.find_first_of("/\\") == std::string::npos;
}

// 本地时间，带毫秒，不带时区标记，例如 20261005T163012.345。
// 时区来自容器挂载的 /etc/localtime；文件名只供人阅读，绝对时间以事件中的 start_ms 为准。
// 不含冒号，可用于各类文件系统。
std::string LocalTimestamp(std::uint64_t epoch_ms) {
  ::tzset();  // 进程启动后 TZ 可能变化（测试中会切换），每次重新读取
  const std::time_t seconds = static_cast<std::time_t>(epoch_ms / 1000);
  std::tm local{};
  ::localtime_r(&seconds, &local);
  char date[32];
  std::strftime(date, sizeof(date), "%Y%m%dT%H%M%S", &local);
  char text[48];
  std::snprintf(text, sizeof(text), "%s.%03u", date, static_cast<unsigned>(epoch_ms % 1000));
  return text;
}

constexpr const char* kDetectedSuffix = "-detected";

// 已收尾分段文件名在 "<source_id>-<branch>-" 之后的部分；匹配时写入是否带检测标记。
bool ParseFinishedName(const std::string& rest, bool* detected) {
  static const std::regex kPattern(R"(\d{8}T\d{6}\.\d{3}-\d+(-detected)?\.mp4)");
  std::smatch match;
  if (!std::regex_match(rest, match, kPattern)) return false;
  *detected = match[1].matched;
  return true;
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

std::string SegmentTracker::FinalPath(const std::string& id, bool detected) const {
  return (fs::path(Directory()) / (id + (detected ? kDetectedSuffix : "") + ".mp4")).string();
}

Segment SegmentTracker::Open(std::uint64_t start_ms) {
  fs::create_directories(Directory());
  Segment segment;
  segment.start_ms = start_ms;
  do {
    segment.id = source_id_ + "-" + branch_ + "-" + LocalTimestamp(start_ms) + "-" +
                 std::to_string(sequence_++);
    segment.final_path = FinalPath(segment.id, false);
    segment.temp_path = segment.final_path + ".tmp";
  } while (open_.count(segment.temp_path) != 0 || fs::exists(segment.final_path) ||
           fs::exists(FinalPath(segment.id, true)) || fs::exists(segment.temp_path));
  open_[segment.temp_path] = segment;
  return segment;
}

bool SegmentTracker::Close(const std::string& temp_path, Segment* closed) {
  const auto found = open_.find(temp_path);
  if (found == open_.end()) return false;
  *closed = found->second;
  // 是否出现过目标到收尾时才确定，因此只有最终文件名带标记，分段 id 不变。
  if (closed->detected) closed->final_path = FinalPath(closed->id, true);
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

PruneResult SegmentTracker::Prune(const Retention& retention,
                                  const std::function<std::uint64_t()>& available) const {
  PruneResult result;
  if (retention.max_files == 0 && retention.min_free_bytes == 0) return result;

  struct File {
    fs::path path;
    fs::file_time_type time;
    bool detected;
  };
  // 只统计本录像器命名的已收尾分段；.tmp 与其他文件不参与计数，也不会被删除。
  // 按修改时间（收尾时刻）排序：文件名中的本地时间在时区调整或夏令时回拨时不单调。
  const std::string prefix = source_id_ + "-" + branch_ + "-";
  std::vector<File> files;
  std::error_code code;
  for (const auto& entry : fs::directory_iterator(Directory(), code)) {
    const std::string name = entry.path().filename().string();
    bool detected = false;
    if (name.compare(0, prefix.size(), prefix) != 0 ||
        !ParseFinishedName(name.substr(prefix.size()), &detected)) {
      continue;
    }
    std::error_code time_code;
    if (!entry.is_regular_file(time_code)) continue;
    const auto time = entry.last_write_time(time_code);
    if (time_code) continue;
    files.push_back({entry.path(), time, detected});
  }
  std::sort(files.begin(), files.end(), [](const File& a, const File& b) {
    return a.time != b.time ? a.time < b.time : a.path < b.path;
  });

  // 删除顺序：保护区（最新 keep_files 个）之外先删无目标、再删有目标，各自从旧到新；
  // 保护区内同样排序，只在剩余空间不足时才会进入。
  const std::size_t total = files.size();
  const std::size_t unprotected = total > retention.keep_files ? total - retention.keep_files : 0;
  std::vector<const File*> order;
  for (const auto& [begin, end] : {std::pair<std::size_t, std::size_t>{0, unprotected},
                                   std::pair<std::size_t, std::size_t>{unprotected, total}}) {
    for (const bool detected : {false, true}) {
      for (std::size_t i = begin; i < end; ++i) {
        if (files[i].detected == detected) order.push_back(&files[i]);
      }
    }
  }

  const auto free_bytes = [&]() -> std::uint64_t {
    if (available) return available();
    std::error_code space_code;
    const fs::space_info space = fs::space(Directory(), space_code);
    // 查询失败（例如目录尚未创建）视为空间充足：无法判断时不删除录像。
    return space_code ? std::numeric_limits<std::uint64_t>::max()
                      : static_cast<std::uint64_t>(space.available);
  };
  const auto remove = [&](const File& file) {
    std::error_code remove_code;
    fs::remove(file.path, remove_code);
    if (remove_code) {
      ++result.failed;
      result.errors.push_back("cannot remove " + file.path.string() + ": " + remove_code.message());
      return;
    }
    ++result.removed;
    if (&file - files.data() >= static_cast<std::ptrdiff_t>(unprotected)) ++result.removed_protected;
  };

  std::size_t next = 0;
  // 数量上限：keep_files ≤ max_files，所以超出部分总能在保护区之外删完。
  const std::size_t excess =
      retention.max_files > 0 && total > retention.max_files ? total - retention.max_files : 0;
  for (; next < excess && next < unprotected; ++next) remove(*order[next]);
  // 剩余空间兜底：可以进入保护区，因为写满磁盘会使全部录像失败。
  if (retention.min_free_bytes > 0) {
    while (free_bytes() < retention.min_free_bytes) {
      if (next == order.size()) {
        result.low_space = true;
        break;
      }
      remove(*order[next++]);
    }
  }
  return result;
}

void SegmentTracker::MarkDetected() {
  for (auto& [path, segment] : open_) segment.detected = true;
}

// ---- SegmentRecorder ----

SegmentRecorder::SegmentRecorder(RecorderOptions options, Events events)
    : options_(std::move(options)),
      events_(std::move(events)),
      name_("recorder:" + options_.element),
      tracker_(options_.root, options_.source_id, options_.branch) {
  if (options_.retention.max_files > 0 && options_.retention.keep_files > options_.retention.max_files) {
    throw std::invalid_argument("retention keep_files must not exceed max_files");
  }
}

SegmentRecorder::~SegmentRecorder() { OnReleased(); }

void SegmentRecorder::MarkDetected() {
  std::lock_guard<std::mutex> lock(mutex_);
  tracker_.MarkDetected();
}

void SegmentRecorder::Require(gst::Requirements& requirements) const {
  requirements.Signal("splitmuxsink", "format-location", options_.element);
  if (options_.split_on_gap.count() > 0) {
    requirements.Signal("splitmuxsink", "split-at-running-time", options_.element);
  }
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
  Prune();
  handler_ = g_signal_connect(sink_, "format-location", G_CALLBACK(&SegmentRecorder::OnFormatLocation),
                              this);
  if (options_.split_on_gap.count() > 0) {
    // video 是 AddSegmentSink 请求的 pad。
    input_.Attach(GST_BIN(bin), options_.element, "video", GST_PAD_PROBE_TYPE_BUFFER, &SegmentRecorder::OnInput,
                  this);
  }
}

GstPadProbeReturn SegmentRecorder::OnInput(GstPad* pad, GstPadProbeInfo* info, gpointer self) {
  auto* recorder = static_cast<SegmentRecorder*>(self);
  GstBuffer* buffer = GST_PAD_PROBE_INFO_BUFFER(info);
  const GstClockTime pts = GST_BUFFER_PTS_IS_VALID(buffer) ? GST_BUFFER_PTS(buffer) : GST_BUFFER_DTS(buffer);
  if (!GST_CLOCK_TIME_IS_VALID(pts)) return GST_PAD_PROBE_OK;
  const GstClockTime last = recorder->last_input_;
  const GstClockTime last_duration = recorder->last_duration_;
  recorder->last_input_ = pts;
  recorder->last_duration_ = GST_BUFFER_DURATION(buffer);
  const bool keyframe = !GST_BUFFER_FLAG_IS_SET(buffer, GST_BUFFER_FLAG_DELTA_UNIT);
  const auto gap =
      static_cast<GstClockTime>(std::chrono::nanoseconds(recorder->options_.split_on_gap).count());
  if (!GST_CLOCK_TIME_IS_VALID(last) || pts <= last || pts - last <= gap) {
    if (!recorder->awaiting_keyframe_) return GST_PAD_PROBE_OK;
    // 断流后、第一个关键帧之前的帧：留在旧分段会把断流带进去，新分段又不能从它们开始。
    if (!keyframe) {
      ++recorder->skipped_;
      return GST_PAD_PROBE_DROP;
    }
    spdlog::info("{}: dropped {} frame(s) before the first keyframe after the gap",
                 recorder->options_.element, recorder->skipped_);
    recorder->awaiting_keyframe_ = false;
    return GST_PAD_PROBE_OK;
  }
  GstClockTime running = pts;
  GstClockTime last_end = last + (GST_CLOCK_TIME_IS_VALID(last_duration) ? last_duration : 0);
  if (GstEvent* event = gst_pad_get_sticky_event(pad, GST_EVENT_SEGMENT, 0)) {
    const GstSegment* segment = nullptr;
    gst_event_parse_segment(event, &segment);
    running = gst_segment_to_running_time(segment, GST_FORMAT_TIME, pts);
    last_end = gst_segment_to_running_time(segment, GST_FORMAT_TIME, last_end);
    gst_event_unref(event);
  }
  // 这一帧还没有进入 splitmuxsink：新分段从 running-time 不早于它的第一个关键帧开始，
  // 断流前的帧（包括正在收集的 GOP）全部留在当前分段。
  g_signal_emit_by_name(recorder->sink_, "split-at-running-time", static_cast<guint64>(running));
  {
    std::lock_guard<std::mutex> lock(recorder->mutex_);
    ++recorder->gap_splits_;
    if (GST_CLOCK_TIME_IS_VALID(last_end)) recorder->gap_ends_.push_back(last_end);
  }
  spdlog::info("{}: input resumed after {:.1f} s; starting a new segment", recorder->options_.element,
               static_cast<double>(pts - last) / GST_SECOND);
  if (keyframe) return GST_PAD_PROBE_OK;
  recorder->awaiting_keyframe_ = true;
  recorder->skipped_ = 1;
  return GST_PAD_PROBE_DROP;
}

void SegmentRecorder::OnReleased() {
  fallback_.Cancel();
  pipeline_ = nullptr;
  input_.Remove();
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
  if (structure == nullptr) return;
  const gchar* location = gst_structure_get_string(structure, "location");
  if (location == nullptr) return;
  GstClockTime running = GST_CLOCK_TIME_NONE;
  gst_structure_get_clock_time(structure, "running-time", &running);
  if (gst_structure_has_name(structure, "splitmuxsink-fragment-opened")) {
    std::lock_guard<std::mutex> lock(mutex_);
    opened_at_[location] = running;
    return;
  }
  if (!gst_structure_has_name(structure, "splitmuxsink-fragment-closed")) return;
  GstClockTime opened = GST_CLOCK_TIME_NONE;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (const auto it = opened_at_.find(location); it != opened_at_.end()) {
      opened = it->second;
      opened_at_.erase(it);
    }
    // 在本分段之内断流：分段的画面到断流前最后一帧为止。更早的记录属于已经收尾的分段，一并丢弃。
    if (GST_CLOCK_TIME_IS_VALID(running)) {
      for (const GstClockTime end : gap_ends_) {
        if (GST_CLOCK_TIME_IS_VALID(opened) && end >= opened && end < running) running = end;
      }
      gap_ends_.erase(std::remove_if(gap_ends_.begin(), gap_ends_.end(),
                                     [&](GstClockTime end) { return end <= running; }),
                      gap_ends_.end());
    }
  }

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
  std::uint64_t end_ms = ds::WallMilliseconds();
  if (GST_CLOCK_TIME_IS_VALID(opened) && GST_CLOCK_TIME_IS_VALID(running) && running >= opened) {
    end_ms = segment.start_ms + (running - opened) / GST_MSECOND;
  }
  if (events_.finished) events_.finished(segment, end_ms);
  Prune();
}

void SegmentRecorder::Prune() {
  const PruneResult result = tracker_.Prune(options_.retention);
  if (result.removed_protected > 0) {
    spdlog::warn("{}: free space below {} bytes; removed {} recent segment(s) inside keep_files",
                 options_.element, options_.retention.min_free_bytes, result.removed_protected);
  }
  if (result.low_space) {
    spdlog::warn("{}: free space is still below {} bytes after removing every finished segment",
                 options_.element, options_.retention.min_free_bytes);
  }
  for (const auto& error : result.errors) spdlog::error("{}: {}", options_.element, error);
  std::lock_guard<std::mutex> lock(mutex_);
  pruned_ += static_cast<std::uint64_t>(result.removed);
  failures_ += static_cast<std::uint64_t>(result.failed);
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
  stats.Add(options_.element + "_pruned", pruned_);
  if (gap_splits_ > 0) stats.Add(options_.element + "_gap_splits", gap_splits_);
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

ds::RecordingEvent RecordingStarted(const RecorderOptions& options, const Segment& segment) {
  ds::RecordingEvent event;
  event.type = ds::RecordingEvent::Type::kStarted;
  event.source_id = options.source_id;
  event.branch = options.branch;
  event.segment_id = segment.id;
  event.start_ms = segment.start_ms;
  return event;
}

ds::RecordingEvent RecordingFinished(const RecorderOptions& options, const Segment& segment, std::uint64_t end_ms) {
  ds::RecordingEvent event = RecordingStarted(options, segment);
  event.type = ds::RecordingEvent::Type::kFinished;
  event.end_ms = end_ms;
  event.detected = segment.detected;
  event.path = segment.final_path;
  return event;
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
