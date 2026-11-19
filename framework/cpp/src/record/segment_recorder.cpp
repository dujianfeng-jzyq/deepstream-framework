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
  if (options_.segment_length.count() <= 0) throw std::invalid_argument("segment length must be positive");
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
  requirements.Signal("splitmuxsink", "format-location-full", options_.element);
  requirements.Signal("splitmuxsink", "split-at-running-time", options_.element);
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
  // video 是 AddSegmentSink 请求的 pad；splitmuxsink 的请求 pad 是 ghost pad。
  GstPad* video = gst_element_get_static_pad(sink_, "video");
  if (video == nullptr || !GST_IS_GHOST_PAD(video)) {
    if (video != nullptr) gst_object_unref(video);
    throw std::runtime_error("recording element '" + options_.element + "' has no ghost pad 'video'");
  }
  GstProxyPad* internal = gst_proxy_pad_get_internal(GST_PROXY_PAD(video));
  gst_object_unref(video);
  sink_data_.Attach(GST_PAD(internal),
                    static_cast<GstPadProbeType>(GST_PAD_PROBE_TYPE_BUFFER | GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM),
                    &SegmentRecorder::OnSinkData, this);
  gst_object_unref(internal);
  handler_ = g_signal_connect(sink_, "format-location-full", G_CALLBACK(&SegmentRecorder::OnFormatLocation),
                              this);
  input_.Attach(GST_BIN(bin), options_.element, "video", GST_PAD_PROBE_TYPE_BUFFER, &SegmentRecorder::OnInput,
                this);
  if (options_.split_on_gap.count() > 0) {
    // 检查间隔取阈值的 1/4（不超过 0.5 秒），收尾在断流 1～1.25 倍阈值之间发生。
    const auto interval = std::clamp(options_.split_on_gap / 4, std::chrono::milliseconds(10),
                                     std::chrono::milliseconds(500));
    stall_check_ = pipeline.Every(interval, [this] { CloseOnStall(); });
  }
}

namespace {

// 断流收尾时补入的占位帧。缓冲区标志中 GST_BUFFER_FLAG_LAST 以上的位留给使用者，下游元件不解释。
constexpr auto kPlaceholder = static_cast<GstBufferFlags>(GST_BUFFER_FLAG_LAST << 1);

std::int64_t SteadyMilliseconds() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// 录像入口与 format-location-full 用同一个时间戳计算 running-time，两边才能对上。
GstClockTime Timestamp(GstBuffer* buffer) {
  return GST_BUFFER_PTS_IS_VALID(buffer) ? GST_BUFFER_PTS(buffer) : GST_BUFFER_DTS(buffer);
}

GstClockTime RunningTime(GstPad* pad, GstClockTime timestamp) {
  GstEvent* event = gst_pad_get_sticky_event(pad, GST_EVENT_SEGMENT, 0);
  if (event == nullptr) return timestamp;
  const GstSegment* segment = nullptr;
  gst_event_parse_segment(event, &segment);
  const GstClockTime running = gst_segment_to_running_time(segment, GST_FORMAT_TIME, timestamp);
  gst_event_unref(event);
  return running;
}

}  // namespace

// 帧进入 splitmuxsink 之前请求切分：split-at-running-time 使 splitmuxsink 在起点不早于该时刻的
// 第一个 GOP 之前开始新分段，请求的正是这一帧（关键帧）时恰好在它之前切开。
GstPadProbeReturn SegmentRecorder::OnInput(GstPad* pad, GstPadProbeInfo* info, gpointer self) {
  auto* recorder = static_cast<SegmentRecorder*>(self);
  GstBuffer* buffer = GST_PAD_PROBE_INFO_BUFFER(info);
  if (GST_BUFFER_FLAG_IS_SET(buffer, kPlaceholder)) {
    // CloseOnStall 补入的占位帧；恢复的帧已经先到时丢弃，不能让时间戳倒退。
    if (!recorder->stalled_.load()) return GST_PAD_PROBE_DROP;
    recorder->after_placeholders_ = true;
    return GST_PAD_PROBE_OK;
  }
  recorder->last_arrival_ms_.store(SteadyMilliseconds());
  recorder->stalled_.store(false);
  const GstClockTime pts = Timestamp(buffer);
  if (!GST_CLOCK_TIME_IS_VALID(pts)) return GST_PAD_PROBE_OK;
  const GstClockTime last = recorder->last_input_;
  const GstClockTime last_duration = recorder->last_duration_;
  recorder->last_input_ = pts;
  recorder->last_duration_ = GST_BUFFER_DURATION(buffer);
  const bool keyframe = !GST_BUFFER_FLAG_IS_SET(buffer, GST_BUFFER_FLAG_DELTA_UNIT);
  const auto gap =
      static_cast<GstClockTime>(std::chrono::nanoseconds(recorder->options_.split_on_gap).count());
  const bool after_placeholders = recorder->after_placeholders_;
  recorder->after_placeholders_ = false;
  if (after_placeholders || (gap > 0 && GST_CLOCK_TIME_IS_VALID(last) && pts > last && pts - last > gap)) {
    const GstClockTime running = RunningTime(pad, pts);
    const GstClockTime last_end =
        RunningTime(pad, last + (GST_CLOCK_TIME_IS_VALID(last_duration) ? last_duration : 0));
    // 断流前的帧（包括正在收集的 GOP）全部留在当前分段；新分段从恢复后的第一个关键帧开始计时。
    g_signal_emit_by_name(recorder->sink_, "split-at-running-time", static_cast<guint64>(running));
    recorder->segment_begin_ = GST_CLOCK_TIME_NONE;
    {
      std::lock_guard<std::mutex> lock(recorder->mutex_);
      ++recorder->gap_splits_;
      // 占位帧之后切分时，被收尾的是占位文件：断流前的分段已经在补帧时收尾。
      if (!after_placeholders && GST_CLOCK_TIME_IS_VALID(last_end)) recorder->gap_ends_.push_back(last_end);
    }
    spdlog::info("{}: input resumed after {:.1f} s; starting a new segment", recorder->options_.element,
                 static_cast<double>(pts - last) / GST_SECOND);
    if (!keyframe) {
      recorder->awaiting_keyframe_ = true;
      recorder->skipped_ = 1;
      return GST_PAD_PROBE_DROP;
    }
  } else if (recorder->awaiting_keyframe_) {
    // 断流后、第一个关键帧之前的帧：留在旧分段会把断流带进去，新分段又不能从它们开始。
    if (!keyframe) {
      ++recorder->skipped_;
      return GST_PAD_PROBE_DROP;
    }
    spdlog::info("{}: dropped {} frame(s) before the first keyframe after the gap",
                 recorder->options_.element, recorder->skipped_);
    recorder->awaiting_keyframe_ = false;
  }
  {
    std::lock_guard<std::mutex> lock(recorder->mutex_);
    const GstClockTime duration = GST_BUFFER_DURATION(buffer);
    recorder->last_step_ = duration;
    recorder->last_end_ = pts + (GST_CLOCK_TIME_IS_VALID(duration) ? duration : 0);
  }
  if (!keyframe) return GST_PAD_PROBE_OK;
  const GstClockTime running = RunningTime(pad, pts);
  if (!GST_CLOCK_TIME_IS_VALID(running)) return GST_PAD_PROBE_OK;
  const auto length =
      static_cast<GstClockTime>(std::chrono::nanoseconds(recorder->options_.segment_length).count());
  if (GST_CLOCK_TIME_IS_VALID(recorder->segment_begin_)) {
    if (running < recorder->segment_begin_ + length) return GST_PAD_PROBE_OK;
    g_signal_emit_by_name(recorder->sink_, "split-at-running-time", static_cast<guint64>(running));
  }
  recorder->segment_begin_ = running;
  std::lock_guard<std::mutex> lock(recorder->mutex_);
  recorder->starts_.emplace_back(running, ds::WallMilliseconds());
  return GST_PAD_PROBE_OK;
}

// 断流超过 split_on_gap：在最后一帧之后请求切分并补两帧占位关键帧。第一帧让断流前的 GOP 收齐写入
// 当前分段；第二帧收齐第一帧组成的 GOP，splitmuxsink 据切分请求收尾当前分段，占位帧进入占位文件。
// 补帧经 gst_pad_chain 进入录像入口，与恢复的帧由 pad 的流锁串行化，OnInput 按 stalled_ 决定取舍。
void SegmentRecorder::CloseOnStall() {
  if (sink_ == nullptr || !received_.load() || stalled_.load() || awaiting_keyframe_.load()) return;
  const std::int64_t idle = SteadyMilliseconds() - last_arrival_ms_.load();
  if (idle < options_.split_on_gap.count()) return;
  GstClockTime end = GST_CLOCK_TIME_NONE;
  GstClockTime step = GST_CLOCK_TIME_NONE;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    end = last_end_;
    step = last_step_;
  }
  if (!GST_CLOCK_TIME_IS_VALID(end)) return;
  if (!GST_CLOCK_TIME_IS_VALID(step) || step == 0) step = 40 * GST_MSECOND;
  GstPad* video = gst_element_get_static_pad(sink_, "video");
  if (video == nullptr) return;
  stalled_.store(true);
  g_signal_emit_by_name(sink_, "split-at-running-time", static_cast<guint64>(RunningTime(video, end)));
  for (GstClockTime i = 0; i < 2; ++i) {
    constexpr gsize kSize = 16;
    GstBuffer* placeholder = gst_buffer_new_allocate(nullptr, kSize, nullptr);
    gst_buffer_memset(placeholder, 0, 0, kSize);
    GST_BUFFER_PTS(placeholder) = GST_BUFFER_DTS(placeholder) = end + i * step;
    GST_BUFFER_DURATION(placeholder) = step;
    GST_BUFFER_FLAG_SET(placeholder, kPlaceholder);
    gst_pad_chain(video, placeholder);
  }
  gst_object_unref(video);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    ++stall_closes_;
  }
  spdlog::info("{}: no input for {:.1f} s; closing the current segment", options_.element,
               static_cast<double>(idle) / 1000);
}

bool SegmentRecorder::Drained() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return tracker_.open_count() == 0 && placeholders_.empty();
}

GstPadProbeReturn SegmentRecorder::OnSinkData(GstPad* /*pad*/, GstPadProbeInfo* info, gpointer self) {
  auto* recorder = static_cast<SegmentRecorder*>(self);
  if (GST_PAD_PROBE_INFO_TYPE(info) & GST_PAD_PROBE_TYPE_BUFFER) {
    recorder->received_.store(true);
    return GST_PAD_PROBE_OK;
  }
  if (GST_EVENT_TYPE(GST_PAD_PROBE_INFO_EVENT(info)) != GST_EVENT_EOS || recorder->received_.load()) {
    return GST_PAD_PROBE_OK;
  }
  spdlog::info("{}: end of stream before any frame; nothing to record", recorder->options_.element);
  return GST_PAD_PROBE_DROP;
}

void SegmentRecorder::OnReleased() {
  fallback_.Cancel();
  stall_check_.Cancel();
  pipeline_ = nullptr;
  input_.Remove();
  sink_data_.Remove();
  if (sink_ == nullptr) return;
  if (handler_ != 0) g_signal_handler_disconnect(sink_, handler_);
  handler_ = 0;
  gst_object_unref(sink_);
  sink_ = nullptr;
}

gchar* SegmentRecorder::OnFormatLocation(GstElement* /*sink*/, guint /*fragment*/, GstSample* first,
                                        gpointer self) {
  auto* recorder = static_cast<SegmentRecorder*>(self);
  GstClockTime running = GST_CLOCK_TIME_NONE;
  if (GstBuffer* buffer = first != nullptr ? gst_sample_get_buffer(first) : nullptr) {
    const GstSegment* segment = gst_sample_get_segment(first);
    running = segment != nullptr && segment->format == GST_FORMAT_TIME
                  ? gst_segment_to_running_time(segment, GST_FORMAT_TIME, Timestamp(buffer))
                  : Timestamp(buffer);
  }
  try {
    if (GstBuffer* buffer = first != nullptr ? gst_sample_get_buffer(first) : nullptr;
        buffer != nullptr && GST_BUFFER_FLAG_IS_SET(buffer, kPlaceholder)) {
      std::lock_guard<std::mutex> lock(recorder->mutex_);
      const std::string path = (fs::path(recorder->tracker_.Directory()) /
                                ("placeholder-" + std::to_string(recorder->placeholder_sequence_++) + ".mp4.tmp"))
                                   .string();
      recorder->placeholders_.insert(path);
      return g_strdup(path.c_str());
    }
    Segment segment;
    {
      std::lock_guard<std::mutex> lock(recorder->mutex_);
      // 文件从录像器登记的起点开始时用它进入录像器的墙钟；不是（例如通过 properties 设置了
      // splitmuxsink 自己的切分条件）时退回现在的墙钟。更早的起点属于已经打开的分段。
      std::uint64_t start_ms = ds::WallMilliseconds();
      auto& starts = recorder->starts_;
      while (!starts.empty() && GST_CLOCK_TIME_IS_VALID(running) && starts.front().first <= running) {
        if (starts.front().first == running) start_ms = starts.front().second;
        starts.pop_front();
      }
      segment = recorder->tracker_.Open(start_ms);
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
    // 占位文件：splitmuxsink 发出 fragment-opened 时 filesink 已经打开它，删除目录项后写入照常进行，
    // 内容在关闭时释放；目录中看不到占位文件，进程异常终止也不会留下它。
    if (placeholders_.count(location) > 0) {
      std::error_code code;
      fs::remove(location, code);
      return;
    }
    opened_at_[location] = running;
    return;
  }
  if (!gst_structure_has_name(structure, "splitmuxsink-fragment-closed")) return;
  GstClockTime opened = GST_CLOCK_TIME_NONE;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (placeholders_.erase(location) > 0) {
      std::error_code code;
      fs::remove(location, code);  // 通常已在打开时删除
      return;
    }
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
  stall_check_.Cancel();
  // 不在这里立即向 splitmuxsink 发送 EOS：停止时管线在每个源之后注入 EOS，它跟在最后一帧后面
  // 依次流经编码器、parser 与队列。提前在录像入口发 EOS 会让分段立即封装，上游仍在处理的帧
  // （编码器内部、tee 分支的队列）随后到达时被丢弃。排空完成以 fragment-closed 为准（见 Drained）。
  //
  // 兜底：EOS 在 eos_fallback 内仍未到达录像入口（某个上游元件不传递 EOS）时，直接向录像入口
  // 发送 EOS，封装已经收到的帧。录像入口已收到 EOS 时重复发送不会产生影响。
  if (sink_ == nullptr || pipeline_ == nullptr || options_.eos_fallback.count() <= 0) return;
  fallback_ = pipeline_->After(options_.eos_fallback, [this] {
    if (sink_ == nullptr || Drained()) return;
    spdlog::warn("{}: end of stream did not reach the recorder within {} ms; closing the segment",
                 options_.element, options_.eos_fallback.count());
    gst::SendEosToSinkPads(sink_);
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
  if (stall_closes_ > 0) stats.Add(options_.element + "_stall_closes", stall_closes_);
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

gst::Element AddSegmentSink(gst::Graph& graph, const gst::Endpoint& encoded, const std::string& codec,
                            const std::string& name, const config::Properties& properties) {
  // MP4 封装需要 avc/hvc1 格式的码流，parser 负责从编码器的 byte-stream 转换。
  gst::Element parser = ds::AddParser(graph, encoded, codec, name + "_parse", false);
  gst::Element sink = graph.Add("splitmuxsink", name);
  sink.Set("muxer-factory", "mp4mux").Apply(properties);
  graph.Link(parser, sink.Pad("video"));
  return sink;
}

}  // namespace dsf::record
