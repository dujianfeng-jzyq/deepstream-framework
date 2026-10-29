// dsf::record：分段登记的纯逻辑，以及软件编码 + splitmuxsink 的真实录像流程。

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <mutex>

#include "dsf/app/components.h"
#include "dsf/gst/introspect.h"
#include "dsf/gst/pipeline.h"
#include "dsf/record/segment_recorder.h"
#include <unistd.h>

#include "dsf_test.h"

namespace fs = std::filesystem;
using namespace dsf;
using record::Segment;
using record::SegmentRecorder;
using record::SegmentTracker;

namespace {

// 测试期间切换进程时区，结束后恢复。
struct ScopedTimeZone {
  explicit ScopedTimeZone(const char* zone) {
    const char* previous = std::getenv("TZ");
    had_ = previous != nullptr;
    if (had_) previous_ = previous;
    ::setenv("TZ", zone, 1);
    ::tzset();
  }
  ~ScopedTimeZone() {
    if (had_) {
      ::setenv("TZ", previous_.c_str(), 1);
    } else {
      ::unsetenv("TZ");
    }
    ::tzset();
  }
  bool had_ = false;
  std::string previous_;
};

struct TempDir {
  TempDir() {
    path = fs::temp_directory_path() / ("dsf-record-" + std::to_string(::getpid()) + "-" +
                                        std::to_string(counter++));
    fs::remove_all(path);
    fs::create_directories(path);
  }
  ~TempDir() { fs::remove_all(path); }
  fs::path path;
  static inline int counter = 0;
};

void RequireRecordingPlugins() {
  for (const char* factory : {"x264enc", "h264parse", "splitmuxsink", "mp4mux", "videoconvert"}) {
    if (!gst::HasFactory(factory)) SKIP(std::string("GStreamer element missing: ") + factory);
  }
}

struct Events {
  std::mutex mutex;
  std::vector<std::string> started;
  std::vector<Segment> finished;
  SegmentRecorder::Events Callbacks() {
    return {[this](const Segment& s) {
              std::lock_guard<std::mutex> lock(mutex);
              started.push_back(s.id);
            },
            [this](const Segment& s, std::uint64_t) {
              std::lock_guard<std::mutex> lock(mutex);
              finished.push_back(s);
            }};
  }
};

gst::Graph RecordingGraph(int buffers, bool live) {
  gst::Graph graph;
  gst::Element source = graph.Add("videotestsrc", "src").Set("is-live", live);
  if (buffers > 0) source.Set("num-buffers", buffers);
  gst::Element encoder = graph.Add("x264enc", "enc")
                             .Set("key-int-max", 10)
                             .Set("tune", "zerolatency")
                             .Set("speed-preset", "ultrafast");
  graph.Chain({source, graph.Caps("video/x-raw,width=160,height=120,framerate=30/1"),
               graph.Add("videoconvert"), encoder});
  record::AddSegmentSink(graph, encoder, "h264", 1, "rec");
  return graph;
}

// 录像图：活源 → 编码（有延迟）→ 深队列（模拟 tee 分支与网络抖动缓冲）→ 录像。
// 编码器前的 identity 统计进入编码的帧数，与录像文件里的帧数对账。
gst::Graph BufferedRecordingGraph() {
  gst::Graph graph;
  gst::Element source = graph.Add("videotestsrc", "src").Set("is-live", true);
  gst::Element encoder = graph.Add("x264enc", "enc")
                             .Set("key-int-max", 15)
                             .Set("speed-preset", "ultrafast");  // 不设 zerolatency：编码器持有帧
  gst::Element queue = graph.Add("queue", "rec_queue")
                           .Set("max-size-buffers", 0u)
                           .Set("max-size-bytes", 0u)
                           .Set("max-size-time", std::uint64_t{0})
                           .Set("min-threshold-buffers", 20u);  // 队列里始终压着 20 帧
  graph.Chain({source, graph.Caps("video/x-raw,width=160,height=120,framerate=30/1"),
               graph.Add("videoconvert"), graph.Add("identity", "counted"), encoder, queue});
  record::AddSegmentSink(graph, queue, "h264", 60, "rec");
  return graph;
}

// 断流：从 at 起，经过 identity 的帧时间戳整体后移 shift（直播源断流恢复后 running-time 的跳变）。
class TimestampGap : public gst::Component {
 public:
  TimestampGap(GstClockTime at, GstClockTime shift) : at_(at), shift_(shift) {}
  std::string_view Name() const override { return "timestamp-gap"; }
  void OnCreated(gst::Pipeline& pipeline, GstElement* /*bin*/) override {
    GstPad* pad = gst_element_get_static_pad(pipeline.Element("gap"), "src");
    gst_pad_add_probe(pad, GST_PAD_PROBE_TYPE_BUFFER,
                      +[](GstPad*, GstPadProbeInfo* info, gpointer data) {
                        const auto* self = static_cast<TimestampGap*>(data);
                        GstBuffer* buffer = GST_PAD_PROBE_INFO_BUFFER(info);
                        if (GST_BUFFER_PTS(buffer) < self->at_) return GST_PAD_PROBE_OK;
                        buffer = gst_buffer_make_writable(buffer);
                        GST_BUFFER_PTS(buffer) += self->shift_;
                        if (GST_BUFFER_DTS_IS_VALID(buffer)) GST_BUFFER_DTS(buffer) += self->shift_;
                        GST_PAD_PROBE_INFO_DATA(info) = buffer;
                        return GST_PAD_PROBE_OK;
                      },
                      this, nullptr);
    gst_object_unref(pad);
  }

 private:
  GstClockTime at_;
  GstClockTime shift_;
};

// 统计经过 identity 的帧。
class FrameCounter : public gst::Component {
 public:
  std::string_view Name() const override { return "frame-counter"; }
  void OnCreated(gst::Pipeline& pipeline, GstElement* /*bin*/) override {
    GstPad* pad = gst_element_get_static_pad(pipeline.Element("counted"), "src");
    gst_pad_add_probe(pad, GST_PAD_PROBE_TYPE_BUFFER,
                      +[](GstPad*, GstPadProbeInfo*, gpointer data) {
                        ++*static_cast<std::atomic<int>*>(data);
                        return GST_PAD_PROBE_OK;
                      },
                      &frames, nullptr);
    gst_object_unref(pad);
  }
  std::atomic<int> frames{0};
};

// 用 GStreamer 解出 MP4 中的视频帧数。
int FramesIn(const std::string& path) {
  gst::Graph graph;
  gst::Element source = graph.Add("filesrc", "file").Set("location", path);
  gst::Element demux = graph.Add("qtdemux", "demux");
  gst::Element sink = graph.Add("fakesink", "sink").Set("sync", false).Set("signal-handoffs", true);
  graph.Link(source, demux);
  graph.Link(demux, sink);
  struct Counter : gst::Component {
    std::string_view Name() const override { return "mp4-counter"; }
    void OnCreated(gst::Pipeline&, GstElement* bin) override {
      GstElement* element = gst_bin_get_by_name(GST_BIN(bin), "sink");
      g_signal_connect(element, "handoff",
                       G_CALLBACK(+[](GstElement*, GstBuffer*, GstPad*, gpointer data) {
                         ++*static_cast<int*>(data);
                       }),
                       &frames);
      gst_object_unref(element);
    }
    int frames = 0;
  } counter;
  gst::Pipeline pipeline("count", graph);
  pipeline.Add(counter);
  gst::Options options;
  options.handle_signals = false;
  options.private_context = true;
  pipeline.Run(options);
  return counter.frames;
}

}  // namespace

TEST(drain_stop_records_every_encoded_frame) {
  RequireRecordingPlugins();
  if (!gst::HasFactory("qtdemux")) SKIP("qtdemux missing");
  TempDir dir;
  Events events;
  SegmentRecorder recorder({dir.path.string(), "cam", "osd", "rec"}, events.Callbacks());
  FrameCounter counter;
  gst::Pipeline pipeline("record-no-loss", BufferedRecordingGraph());
  pipeline.Add(counter);
  pipeline.Add(recorder);
  gst::Timer timer;
  pipeline.Post([&] {
    timer = pipeline.Every(std::chrono::milliseconds(2000), [&] { pipeline.RequestStop(); });
  });
  gst::Options options;
  options.handle_signals = false;
  options.stop_timeout = std::chrono::seconds(10);
  const gst::Result result = pipeline.Run(options);
  CHECK(result.ok());
  CHECK(recorder.Healthy());
  int recorded = 0;
  for (const Segment& segment : events.finished) recorded += FramesIn(segment.final_path);
  // 进入编码器的每一帧都必须出现在录像里；排空不能丢掉编码器与队列中仍在处理的帧。
  CHECK_EQ(recorded, counter.frames.load());
}

// 丢弃到达录像入口的 EOS，模拟上游不传递 EOS 的元件。
class DropRecorderEos : public gst::Component {
 public:
  std::string_view Name() const override { return "drop-eos"; }
  void OnCreated(gst::Pipeline& pipeline, GstElement* /*bin*/) override {
    GstPad* pad = gst_element_get_static_pad(pipeline.Element("rec_queue"), "src");
    gst_pad_add_probe(pad, GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM,
                      +[](GstPad*, GstPadProbeInfo* info, gpointer) {
                        return GST_EVENT_TYPE(GST_PAD_PROBE_INFO_EVENT(info)) == GST_EVENT_EOS
                                   ? GST_PAD_PROBE_DROP : GST_PAD_PROBE_OK;
                      },
                      nullptr, nullptr);
    gst_object_unref(pad);
  }
};

TEST(eos_fallback_closes_the_segment_when_upstream_swallows_eos) {
  RequireRecordingPlugins();
  TempDir dir;
  Events events;
  record::RecorderOptions options{dir.path.string(), "cam", "osd", "rec"};
  options.eos_fallback = std::chrono::milliseconds(300);
  SegmentRecorder recorder(options, events.Callbacks());
  DropRecorderEos dropper;
  gst::Pipeline pipeline("record-fallback", BufferedRecordingGraph());
  pipeline.Add(dropper);
  pipeline.Add(recorder);
  gst::Timer timer;
  pipeline.Post([&] {
    timer = pipeline.Every(std::chrono::milliseconds(1500), [&] { pipeline.RequestStop(); });
  });
  gst::Options options_pipeline;
  options_pipeline.handle_signals = false;
  options_pipeline.stop_timeout = std::chrono::seconds(5);
  const gst::Result result = pipeline.Run(options_pipeline);
  CHECK(result.ok());
  CHECK(recorder.Healthy());
  CHECK_EQ(events.finished.size(), 1u);  // 兜底封装了已经收到的帧
  CHECK(FramesIn(events.finished.front().final_path) > 0);
}

TEST(recording_events_carry_the_recorder_identity) {
  record::RecorderOptions options{"/data/rec", "gate-1", "osd", "rec_0"};
  Segment segment{"gate-1-osd-20261005T163012.345-1", "/data/rec/a.mp4.tmp", "/data/rec/a-detected.mp4",
                  1000, true};
  const ds::RecordingEvent started = record::RecordingStarted(options, segment);
  CHECK(started.type == ds::RecordingEvent::Type::kStarted);
  CHECK_EQ(started.source_id, std::string("gate-1"));
  CHECK_EQ(started.branch, std::string("osd"));
  CHECK_EQ(started.segment_id, segment.id);
  CHECK_EQ(started.start_ms, std::uint64_t{1000});
  CHECK(started.path.empty());  // 开始时还没有可读的文件
  const ds::RecordingEvent finished = record::RecordingFinished(options, segment, 61000);
  CHECK(finished.type == ds::RecordingEvent::Type::kFinished);
  CHECK_EQ(finished.end_ms, std::uint64_t{61000});
  CHECK(finished.detected);
  CHECK_EQ(finished.path, std::string("/data/rec/a-detected.mp4"));
  const auto message = ds::RecordingMessage(finished);
  CHECK_EQ(message["event_type"].get<std::string>(), std::string("recording_finished"));
  CHECK_EQ(message["source_id"].get<std::string>(), std::string("gate-1"));
}

TEST(tracker_names_segments_and_never_overwrites) {
  TempDir dir;
  ScopedTimeZone zone("UTC0");
  SegmentTracker tracker(dir.path.string(), "cam", "osd");
  const Segment first = tracker.Open(1000);
  CHECK_EQ(first.id, std::string("cam-osd-19700101T000001.000-0"));
  CHECK_EQ(first.temp_path,
           (dir.path / "cam/osd/cam-osd-19700101T000001.000-0.mp4.tmp").string());
  std::ofstream(fs::path(first.final_path).parent_path() /
                "cam-osd-19700101T000002.000-1.mp4") << "old";
  const Segment second = tracker.Open(2000);
  CHECK_EQ(second.id, std::string("cam-osd-19700101T000002.000-2"));  // 跳过已存在的文件
  tracker.MarkDetected();
  Segment closed;
  CHECK(!tracker.Close("/nope", &closed));
  std::ofstream(first.temp_path) << "data";
  CHECK(tracker.Close(first.temp_path, &closed));
  CHECK(closed.detected);
  CHECK_EQ(closed.id, first.id);  // started 事件中的 id 不变，只有最终文件名带标记
  CHECK_EQ(closed.final_path,
           (dir.path / "cam/osd/cam-osd-19700101T000001.000-0-detected.mp4").string());
  std::string error;
  CHECK(SegmentTracker::Finalize(closed, &error));
  CHECK(fs::exists(closed.final_path));
  CHECK(!fs::exists(first.final_path));
  CHECK(!SegmentTracker::Finalize(closed, &error));
  CHECK_EQ(tracker.open_count(), 1u);
  std::ofstream(fs::path(first.final_path).parent_path() / "stale.mp4.tmp") << "x";
  CHECK_EQ(tracker.RemoveStale(), 1);
  CHECK_THROWS(SegmentTracker(dir.path.string(), "../x", "osd"), "valid paths");
}

TEST(tracker_skips_names_taken_by_detected_segments) {
  TempDir dir;
  ScopedTimeZone zone("UTC0");
  SegmentTracker tracker(dir.path.string(), "cam", "osd");
  fs::create_directories(dir.path / "cam/osd");
  std::ofstream(dir.path / "cam/osd/cam-osd-19700101T000001.000-0-detected.mp4") << "old";
  CHECK_EQ(tracker.Open(1000).id, std::string("cam-osd-19700101T000001.000-1"));
}

namespace {

// 在录像目录中按给定顺序（从旧到新）创建已收尾分段，修改时间依次递增。
// 名称中的 'd' 表示带检测标记，返回各文件名。
std::vector<std::string> MakeSegments(const fs::path& dir, const std::string& kinds) {
  fs::create_directories(dir);
  std::vector<std::string> names;
  const auto base = fs::file_time_type::clock::now() - std::chrono::hours(1);
  for (std::size_t i = 0; i < kinds.size(); ++i) {
    const std::string name = "cam-osd-20261008T100000.000-" + std::to_string(i) +
                             (kinds[i] == 'd' ? "-detected" : "") + ".mp4";
    std::ofstream(dir / name) << "x";
    fs::last_write_time(dir / name, base + std::chrono::seconds(i));
    names.push_back(name);
  }
  return names;
}

std::string Remaining(const fs::path& dir, const std::vector<std::string>& names) {
  std::string kept;
  for (std::size_t i = 0; i < names.size(); ++i) kept += fs::exists(dir / names[i]) ? '1' : '0';
  return kept;
}

}  // namespace

TEST(prune_keeps_recent_files_and_removes_undetected_first) {
  TempDir dir;
  const fs::path segments = dir.path / "cam/osd";
  SegmentTracker tracker(dir.path.string(), "cam", "osd");
  // 从旧到新：d n d n n | n n（最后 2 个为保护区）
  const auto names = MakeSegments(segments, "dndnnnn");
  std::ofstream(segments / "cam-osd-20261008T100000.000-99.mp4.tmp") << "x";  // 正在写
  std::ofstream(segments / "notes.mp4") << "x";                              // 不是本录像器的命名
  std::ofstream(segments / "cam-raw-20261008T100000.000-0.mp4") << "x";      // 其他分支
  record::Retention retention;
  retention.max_files = 4;
  retention.keep_files = 2;
  const record::PruneResult result = tracker.Prune(retention);
  CHECK_EQ(result.removed, 3);
  CHECK_EQ(result.removed_protected, 0);
  // 删除顺序：保护区外最旧的无目标（1、3、4）先于有目标（0、2）。
  CHECK_EQ(Remaining(segments, names), std::string("1010011"));
  CHECK(fs::exists(segments / "cam-osd-20261008T100000.000-99.mp4.tmp"));
  CHECK(fs::exists(segments / "notes.mp4"));
  CHECK(fs::exists(segments / "cam-raw-20261008T100000.000-0.mp4"));

  // 再超出一个：保护区外已没有无目标分段，删最旧的有目标分段。
  retention.max_files = 3;
  CHECK_EQ(tracker.Prune(retention).removed, 1);
  CHECK_EQ(Remaining(segments, names), std::string("0010011"));

  // 未超限时不删除；全部为 0 时不清理。
  CHECK_EQ(tracker.Prune(retention).removed, 0);
  CHECK_EQ(tracker.Prune(record::Retention{}).removed, 0);
}

TEST(prune_never_enters_keep_files_for_the_count_limit) {
  TempDir dir;
  const fs::path segments = dir.path / "cam/osd";
  SegmentTracker tracker(dir.path.string(), "cam", "osd");
  const auto names = MakeSegments(segments, "dddnn");
  record::Retention retention;
  retention.max_files = 3;
  retention.keep_files = 3;
  CHECK_EQ(tracker.Prune(retention).removed, 2);
  CHECK_EQ(Remaining(segments, names), std::string("00111"));
}

TEST(prune_frees_space_even_inside_keep_files) {
  TempDir dir;
  const fs::path segments = dir.path / "cam/osd";
  SegmentTracker tracker(dir.path.string(), "cam", "osd");
  const auto names = MakeSegments(segments, "ndnnd");
  record::Retention retention;
  retention.max_files = 10;
  retention.keep_files = 3;
  retention.min_free_bytes = 100;
  // 模拟磁盘：起始剩余 10 字节，每删一个分段释放 30 字节，需要删 3 个。
  const auto available = [&] {
    std::uint64_t free_bytes = 10;
    for (const auto& name : names) {
      if (!fs::exists(segments / name)) free_bytes += 30;
    }
    return free_bytes;
  };
  const record::PruneResult result = tracker.Prune(retention, available);
  CHECK_EQ(result.removed, 3);
  CHECK_EQ(result.removed_protected, 1);
  CHECK(!result.low_space);
  // 保护区外：先删无目标 0，再删有目标 1；仍不足时进入保护区（2、3、4）并先删无目标 2。
  CHECK_EQ(Remaining(segments, names), std::string("00011"));

  // 删光仍不足时报告 low_space，不删除其他文件。
  std::ofstream(segments / "notes.mp4") << "x";
  const record::PruneResult exhausted = tracker.Prune(retention, [] { return std::uint64_t{0}; });
  CHECK_EQ(exhausted.removed, 2);
  CHECK(exhausted.low_space);
  CHECK(fs::exists(segments / "notes.mp4"));
}

TEST(prune_reports_files_it_cannot_remove) {
  if (::geteuid() == 0) SKIP("root ignores directory permissions");
  TempDir dir;
  const fs::path segments = dir.path / "cam/osd";
  SegmentTracker tracker(dir.path.string(), "cam", "osd");
  const auto names = MakeSegments(segments, "nnn");
  fs::permissions(segments, fs::perms::owner_read | fs::perms::owner_exec);
  record::Retention retention;
  retention.max_files = 1;
  const record::PruneResult result = tracker.Prune(retention);
  fs::permissions(segments, fs::perms::owner_all);
  CHECK_EQ(result.removed, 0);
  CHECK_EQ(result.failed, 2);
  CHECK_EQ(result.errors.size(), 2u);
  CHECK(result.errors[0].find("cannot remove") != std::string::npos);
  CHECK_EQ(Remaining(segments, names), std::string("111"));
}

TEST(recorder_rejects_keep_files_above_max_files) {
  record::RecorderOptions options{"/tmp", "cam", "osd", "rec"};
  options.retention.max_files = 2;
  options.retention.keep_files = 3;
  CHECK_THROWS(SegmentRecorder(options, {}), "keep_files");
}

TEST(tracker_names_segments_in_local_time) {
  TempDir dir;
  ScopedTimeZone zone("CST-8");  // POSIX 写法：UTC+8，不依赖 tzdata
  SegmentTracker tracker(dir.path.string(), "cam", "raw");
  CHECK_EQ(tracker.Open(1791196212345).id, std::string("cam-raw-20261005T183012.345-0"));
}

TEST(recorder_closes_segments_at_eos) {
  RequireRecordingPlugins();
  TempDir dir;
  Events events;
  SegmentRecorder recorder({dir.path.string(), "cam", "raw", "rec"}, events.Callbacks());
  gst::Pipeline pipeline("record-eos", RecordingGraph(75, false));
  pipeline.Add(recorder);
  const gst::Result result = pipeline.Run();
  CHECK(result.ok());
  CHECK(recorder.Healthy());
  CHECK(events.finished.size() >= 2);
  CHECK_EQ(events.started.size(), events.finished.size());
  for (const Segment& segment : events.finished) {
    CHECK(fs::file_size(segment.final_path) > 0);
    CHECK(!fs::exists(segment.temp_path));
  }
}

TEST(recorder_reports_segments_it_cannot_finalize) {
  RequireRecordingPlugins();
  TempDir dir;
  // 分段开始时在最终路径上放一个目录：收尾改名必然失败，分段不能被报告为完成。
  std::vector<std::string> started;
  SegmentRecorder::Events callbacks;
  callbacks.started = [&](const Segment& segment) {
    fs::create_directories(segment.final_path);
    started.push_back(segment.id);
  };
  int finished = 0;
  callbacks.finished = [&](const Segment&, std::uint64_t) { ++finished; };
  SegmentRecorder recorder({dir.path.string(), "cam", "raw", "rec"}, callbacks);
  gst::Pipeline pipeline("record-rename-fails", RecordingGraph(45, false));
  pipeline.Add(recorder);
  CHECK(pipeline.Run().ok());  // 管线本身正常结束，失败由录像器的健康状态报告
  CHECK(!started.empty());
  CHECK_EQ(finished, 0);
  CHECK_EQ(recorder.failures(), static_cast<std::uint64_t>(started.size()));
  CHECK(!recorder.Healthy());
  gst::Stats stats;
  recorder.Report(stats);
  CHECK(stats.Text().find("rec_failures=" + std::to_string(started.size())) != std::string::npos);
}

TEST(recorder_prunes_after_each_segment) {
  RequireRecordingPlugins();
  TempDir dir;
  Events events;
  record::RecorderOptions options{dir.path.string(), "cam", "raw", "rec"};
  options.retention.max_files = 1;
  options.retention.keep_files = 1;
  SegmentRecorder recorder(options, events.Callbacks());
  gst::Pipeline pipeline("record-prune", RecordingGraph(75, false));
  pipeline.Add(recorder);
  const gst::Result result = pipeline.Run();
  CHECK(result.ok());
  CHECK(recorder.Healthy());
  CHECK(events.finished.size() >= 2);
  int files = 0;
  for (const auto& entry : fs::directory_iterator(dir.path / "cam/raw")) {
    ++files;
    CHECK_EQ(entry.path().string(), events.finished.back().final_path);  // 只留最新的分段
  }
  CHECK_EQ(files, 1);
  gst::Stats stats;
  recorder.Report(stats);
  CHECK(stats.Text().find("rec_pruned=" + std::to_string(events.finished.size() - 1)) !=
        std::string::npos);
}

namespace {

// 30 帧（1 秒），第 15 帧起时间戳后移 3 秒（断流）。key_interval 决定周期关键帧的间隔。
gst::Graph GapGraph(int key_interval) {
  gst::Graph graph;
  gst::Element source = graph.Add("videotestsrc", "src").Set("num-buffers", 30);
  gst::Element encoder = graph.Add("x264enc", "enc")
                             .Set("key-int-max", key_interval)
                             .Set("tune", "zerolatency")
                             .Set("speed-preset", "ultrafast");
  graph.Chain({source, graph.Caps("video/x-raw,width=160,height=120,framerate=30/1"),
               graph.Add("videoconvert"), graph.Add("identity", "gap"), encoder});
  record::AddSegmentSink(graph, encoder, "h264", 60, "rec");
  return graph;
}

struct GapRun {
  std::vector<std::pair<Segment, std::uint64_t>> finished;  // 分段与 end_ms
  std::string stats;
};

GapRun RecordAcrossGap(const fs::path& root, int key_interval, bool request_keyframe) {
  GapRun run;
  SegmentRecorder::Events callbacks;
  callbacks.finished = [&](const Segment& segment, std::uint64_t end_ms) {
    run.finished.emplace_back(segment, end_ms);
  };
  record::RecorderOptions options{root.string(), "cam", "osd", "rec"};
  options.split_on_gap = std::chrono::milliseconds(1000);
  SegmentRecorder recorder(options, callbacks);
  TimestampGap gap(GST_SECOND / 2, 3 * GST_SECOND);
  app::KeyframeOnGap keyframes("enc", std::chrono::milliseconds(1000));
  gst::Pipeline pipeline("record-gap", GapGraph(key_interval));
  pipeline.Add(gap);
  if (request_keyframe) pipeline.Add(keyframes);
  pipeline.Add(recorder);
  CHECK(pipeline.Run().ok());
  CHECK(recorder.Healthy());
  gst::Stats stats;
  recorder.Report(stats);
  keyframes.Report(stats);
  run.stats = stats.Text();
  return run;
}

}  // namespace

TEST(recorder_starts_a_new_segment_where_the_input_resumes) {
  RequireRecordingPlugins();
  TempDir dir;
  // GOP 足够长：只有断流处请求的关键帧能切分。
  const GapRun run = RecordAcrossGap(dir.path, 300, true);
  CHECK_EQ(run.finished.size(), std::size_t{2});
  if (run.finished.size() != 2) return;
  CHECK_EQ(FramesIn(run.finished[0].first.final_path), 15);  // 断流前的帧全部留在第一个分段
  CHECK_EQ(FramesIn(run.finished[1].first.final_path), 15);  // 恢复后的帧一帧不丢
  // 结束时间按分段内画面的时长计算，不含 3 秒的断流。
  const std::uint64_t first = run.finished[0].second - run.finished[0].first.start_ms;
  CHECK(first >= 450 && first <= 550);
  CHECK(run.stats.find("rec_gap_splits=1") != std::string::npos);
  CHECK(run.stats.find("enc_gap_keyframes=1") != std::string::npos);
}

TEST(recorder_drops_frames_before_the_first_keyframe_after_a_gap) {
  RequireRecordingPlugins();
  TempDir dir;
  // 不请求关键帧：恢复后的第 15～19 帧是延续旧 GOP 的预测帧，新分段从第 20 帧（周期关键帧）开始。
  const GapRun run = RecordAcrossGap(dir.path, 10, false);
  CHECK_EQ(run.finished.size(), std::size_t{2});
  if (run.finished.size() != 2) return;
  CHECK_EQ(FramesIn(run.finished[0].first.final_path), 15);
  CHECK_EQ(FramesIn(run.finished[1].first.final_path), 10);
}

namespace {

// 模拟 Jetson 的 nvv4l2h26xenc：identity 的子类，带 force-IDR 动作信号。注册为元件 dsftestidrenc。
void RegisterIdrSignalElement() {
  static const bool registered = [] {
    GstElementFactory* factory = gst_element_factory_find("identity");
    GstPluginFeature* loaded = gst_plugin_feature_load(GST_PLUGIN_FEATURE(factory));
    const GType parent = gst_element_factory_get_element_type(GST_ELEMENT_FACTORY(loaded));
    gst_object_unref(loaded);
    gst_object_unref(factory);
    GTypeQuery query;
    g_type_query(parent, &query);
    const GType type = g_type_register_static_simple(
        parent, "DsfTestIdrSignalEncoder", query.class_size,
        +[](gpointer klass, gpointer) {
          g_signal_new("force-IDR", G_TYPE_FROM_CLASS(klass),
                       static_cast<GSignalFlags>(G_SIGNAL_RUN_LAST | G_SIGNAL_ACTION), 0, nullptr, nullptr,
                       nullptr, G_TYPE_NONE, 0);
        },
        query.instance_size, nullptr, static_cast<GTypeFlags>(0));
    return gst_element_register(nullptr, "dsftestidrenc", GST_RANK_NONE, type) == TRUE;
  }();
  CHECK(registered);
}

}  // namespace

TEST(keyframe_on_gap_emits_the_encoders_force_idr_signal) {
  RegisterIdrSignalElement();
  gst::Graph graph;
  graph.Chain({graph.Add("videotestsrc", "src").Set("num-buffers", 30),
               graph.Caps("video/x-raw,width=160,height=120,framerate=30/1"), graph.Add("identity", "gap"),
               graph.Add("dsftestidrenc", "enc"), graph.Add("fakesink")});
  TimestampGap gap(GST_SECOND / 2, 3 * GST_SECOND);
  app::KeyframeOnGap keyframes("enc", std::chrono::milliseconds(1000));
  gst::Pipeline pipeline("idr-signal", graph);
  pipeline.Add(gap);
  pipeline.Add(keyframes);
  // 记录信号发出时编码器已经收到的帧数：应在断流后的第一帧（第 16 帧）进入编码器之前。
  std::atomic<int> received{0};
  std::vector<int> signalled;
  struct Probe : gst::Component {
    explicit Probe(std::atomic<int>& count, std::vector<int>& at) : count_(count), at_(at) {}
    std::string_view Name() const override { return "idr-probe"; }
    void OnCreated(gst::Pipeline& owner, GstElement* /*bin*/) override {
      GstElement* encoder = owner.Element("enc");
      g_signal_connect(encoder, "force-IDR", G_CALLBACK(+[](GstElement*, gpointer self) {
                         auto* probe = static_cast<Probe*>(self);
                         probe->at_.push_back(probe->count_.load());
                       }),
                       this);
      GstPad* pad = gst_element_get_static_pad(encoder, "src");
      gst_pad_add_probe(pad, GST_PAD_PROBE_TYPE_BUFFER,
                        +[](GstPad*, GstPadProbeInfo*, gpointer data) {
                          ++*static_cast<std::atomic<int>*>(data);
                          return GST_PAD_PROBE_OK;
                        },
                        &count_, nullptr);
      gst_object_unref(pad);
    }
    std::atomic<int>& count_;
    std::vector<int>& at_;
  } probe(received, signalled);
  pipeline.Add(probe);
  CHECK(pipeline.Run().ok());
  CHECK_EQ(received.load(), 30);
  CHECK_EQ(signalled.size(), std::size_t{1});
  if (signalled.size() == 1) CHECK_EQ(signalled[0], 15);
}

TEST(recorder_finishes_last_segment_on_drain_stop) {
  RequireRecordingPlugins();
  TempDir dir;
  // 上次运行遗留的未完成分段会在启动时删除。
  fs::create_directories(dir.path / "cam/osd");
  std::ofstream(dir.path / "cam/osd/old.mp4.tmp") << "x";
  Events events;
  SegmentRecorder recorder({dir.path.string(), "cam", "osd", "rec"}, events.Callbacks());
  gst::Pipeline pipeline("record-stop", RecordingGraph(0, true));
  pipeline.Add(recorder);
  gst::Timer timer;
  pipeline.Post([&] {
    timer = pipeline.Every(std::chrono::milliseconds(1500), [&] { pipeline.RequestStop(); });
  });
  gst::Options options;
  options.handle_signals = false;
  const gst::Result result = pipeline.Run(options);
  CHECK(result.ok());
  CHECK_EQ(std::string(gst::ToString(result.reason)), std::string("stopped"));
  CHECK(recorder.Healthy());
  CHECK(!events.finished.empty());
  CHECK(!fs::exists(dir.path / "cam/osd/old.mp4.tmp"));
  for (const auto& entry : fs::directory_iterator(dir.path / "cam/osd")) {
    CHECK(entry.path().extension() == ".mp4");
  }
}

DSF_TEST_MAIN()
