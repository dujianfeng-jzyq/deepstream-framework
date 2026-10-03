// dsf::record：分段登记的纯逻辑，以及软件编码 + splitmuxsink 的真实录像流程。

#include <atomic>
#include <filesystem>
#include <fstream>
#include <mutex>

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
  CHECK(result.success);
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
  CHECK(result.success);
  CHECK(recorder.Healthy());
  CHECK_EQ(events.finished.size(), 1u);  // 兜底封装了已经收到的帧
  CHECK(FramesIn(events.finished.front().final_path) > 0);
}

TEST(tracker_names_segments_and_never_overwrites) {
  TempDir dir;
  SegmentTracker tracker(dir.path.string(), "cam", "osd");
  const Segment first = tracker.Open(1000);
  CHECK_EQ(first.id, std::string("cam-osd-1000-0"));
  CHECK_EQ(first.temp_path, (dir.path / "cam/osd/cam-osd-1000-0.mp4.tmp").string());
  std::ofstream(fs::path(first.final_path).parent_path() / "cam-osd-2000-1.mp4") << "old";
  const Segment second = tracker.Open(2000);
  CHECK_EQ(second.id, std::string("cam-osd-2000-2"));  // 跳过已存在的文件
  tracker.MarkDetected();
  Segment closed;
  CHECK(!tracker.Close("/nope", &closed));
  std::ofstream(first.temp_path) << "data";
  CHECK(tracker.Close(first.temp_path, &closed));
  CHECK(closed.detected);
  std::string error;
  CHECK(SegmentTracker::Finalize(closed, &error));
  CHECK(fs::exists(first.final_path));
  CHECK(!SegmentTracker::Finalize(closed, &error));
  CHECK_EQ(tracker.open_count(), 1u);
  std::ofstream(fs::path(first.final_path).parent_path() / "stale.mp4.tmp") << "x";
  CHECK_EQ(tracker.RemoveStale(), 1);
  CHECK_THROWS(SegmentTracker(dir.path.string(), "../x", "osd"), "valid paths");
}

TEST(recorder_closes_segments_at_eos) {
  RequireRecordingPlugins();
  TempDir dir;
  Events events;
  SegmentRecorder recorder({dir.path.string(), "cam", "raw", "rec"}, events.Callbacks());
  gst::Pipeline pipeline("record-eos", RecordingGraph(75, false));
  pipeline.Add(recorder);
  const gst::Result result = pipeline.Run();
  CHECK(result.success);
  CHECK(recorder.Healthy());
  CHECK(events.finished.size() >= 2);
  CHECK_EQ(events.started.size(), events.finished.size());
  for (const Segment& segment : events.finished) {
    CHECK(fs::file_size(segment.final_path) > 0);
    CHECK(!fs::exists(segment.temp_path));
  }
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
  CHECK(result.success);
  CHECK_EQ(std::string(gst::ToString(result.reason)), std::string("stopped"));
  CHECK(recorder.Healthy());
  CHECK(!events.finished.empty());
  CHECK(!fs::exists(dir.path / "cam/osd/old.mp4.tmp"));
  for (const auto& entry : fs::directory_iterator(dir.path / "cam/osd")) {
    CHECK(entry.path().extension() == ".mp4");
  }
}

DSF_TEST_MAIN()
