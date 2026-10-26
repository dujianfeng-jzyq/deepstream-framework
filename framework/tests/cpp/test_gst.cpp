// dsf::gst：内省、图模型与 Pipeline 生命周期。只使用 GStreamer 核心与 base 插件。

#include <atomic>
#include <chrono>
#include <thread>

#include "dsf/gst/component.h"
#include "dsf/gst/graph.h"
#include "dsf/gst/introspect.h"
#include "dsf/gst/pipeline.h"
#include "dsf_test.h"

using namespace dsf::gst;

namespace {

void RequirePlugin(const char* factory) {
  if (!HasFactory(factory)) SKIP(std::string("GStreamer element missing: ") + factory);
}

// 记录回调顺序的组件。
class Recorder : public Component {
 public:
  explicit Recorder(std::string name, std::vector<std::string>* log) : name_(std::move(name)), log_(log) {}
  std::string_view Name() const override { return name_; }
  void OnCreated(Pipeline&, GstElement*) override {
    log_->push_back(name_ + ":created");
    if (throw_on_created) throw std::runtime_error("boom");
  }
  void OnPlaying(Pipeline&) override { log_->push_back(name_ + ":playing"); }
  void OnStopping() override { log_->push_back(name_ + ":stopping"); }
  void OnReleased() override { log_->push_back(name_ + ":released"); }
  bool throw_on_created = false;

 private:
  std::string name_;
  std::vector<std::string>* log_;
};

Graph TestGraph(int buffers, bool live) {
  Graph graph;
  Element source = graph.Add("videotestsrc", "src");
  if (buffers > 0) source.Set("num-buffers", buffers);
  source.Set("is-live", live);
  graph.Chain({source, graph.Caps("video/x-raw,width=64,height=48,framerate=30/1"),
               graph.Add("fakesink", "sink").Set("sync", live)});
  return graph;
}

// 不推数据的 live appsrc：异步 fakesink 等待 preroll，模拟按需产帧的管线。
Graph IdleLiveGraph() {
  Graph graph;
  graph.Chain({graph.Add("appsrc", "src")
                   .Set("is-live", true)
                   .Set("format", "time")
                   .Set("caps", "video/x-raw,format=GRAY8,width=64,height=48,framerate=0/1"),
               graph.Add("fakesink", "sink").Set("sync", false).Set("async", true)});
  return graph;
}

// 同时有 always src pad 与 sometimes 模板的元件：区分“连接失败”与“等待 pad 出现”。
const char* RegisterDualSource() {
  static const GType type = g_type_register_static_simple(
      GST_TYPE_ELEMENT, "DsfTestDualSource", sizeof(GstElementClass),
      +[](gpointer klass, gpointer) {
        auto* element_class = GST_ELEMENT_CLASS(klass);
        GstCaps* caps = gst_caps_from_string("video/x-raw");
        gst_element_class_add_pad_template(element_class,
                                           gst_pad_template_new("src", GST_PAD_SRC, GST_PAD_ALWAYS, caps));
        gst_element_class_add_pad_template(
            element_class, gst_pad_template_new("dyn_%u", GST_PAD_SRC, GST_PAD_SOMETIMES, caps));
        gst_caps_unref(caps);
        gst_element_class_set_static_metadata(element_class, "dual source", "Source/Video", "test", "dsf");
      },
      sizeof(GstElement),
      +[](GTypeInstance* instance, gpointer klass) {
        GstPadTemplate* templ = gst_element_class_get_pad_template(GST_ELEMENT_CLASS(klass), "src");
        gst_element_add_pad(GST_ELEMENT(instance), gst_pad_new_from_template(templ, "src"));
      },
      GTypeFlags(0));
  gst_element_register(nullptr, "dsftestdual", GST_RANK_NONE, type);
  return "dsftestdual";
}

void PushFrame(GstElement* source) {
  GstBuffer* buffer = gst_buffer_new_allocate(nullptr, 64 * 48, nullptr);
  GstFlowReturn flow = GST_FLOW_OK;
  g_signal_emit_by_name(source, "push-buffer", buffer, &flow);
  gst_buffer_unref(buffer);
}

}  // namespace

TEST(introspection_reports_factories_properties_and_signals) {
  CHECK(HasFactory("fakesink"));
  CHECK(!HasFactory("no-such-element"));
  CHECK(HasProperty("fakesink", "sync"));
  CHECK(!HasProperty("fakesink", "no-such-property"));
  CHECK(HasSignal("fakesink", "handoff"));
  CHECK(!HasSignal("fakesink", "no-such-signal"));
}

TEST(requirements_check_values_against_param_specs) {
  Requirements ok;
  ok.Value("videotestsrc", "pattern", "ball");   // 枚举昵称
  ok.Value("videotestsrc", "pattern", "18");     // 枚举数值
  ok.Value("fakesink", "sync", "false");
  ok.Value("capsfilter", "caps", "video/x-raw,width=10");
  ok.Check();

  Requirements bad;
  bad.Factory("no-such-element", "a.b");
  bad.Property("fakesink", "nope");
  bad.Value("videotestsrc", "pattern", "no-such-pattern", "sources[0].properties.pattern");
  bad.Value("queue", "max-size-buffers", "-5");
  bad.Signal("fakesink", "nope");
  try {
    bad.Check();
    CHECK(false);
  } catch (const std::runtime_error& error) {
    const std::string text = error.what();
    CHECK(text.find("a.b: element 'no-such-element' is not available") != std::string::npos);
    CHECK(text.find("has no property 'nope'") != std::string::npos);
    CHECK(text.find("sources[0].properties.pattern") != std::string::npos);
    CHECK(text.find("max-size-buffers") != std::string::npos);
    CHECK(text.find("has no signal 'nope'") != std::string::npos);
  }
}

TEST(graph_describe_redacts_secrets_and_lists_links) {
  Graph graph;
  Element a = graph.Add("videotestsrc");
  Element b = graph.Add("fakesink", "out");
  a.SetSecret("pattern", "hidden").Set("num-buffers", 3);
  graph.Link(a, b.Pad("sink"));
  const std::string text = graph.Describe();
  CHECK(text.find("videotestsrc0 = videotestsrc pattern=*** num-buffers=3") != std::string::npos);
  CHECK(text.find("videotestsrc0 -> out.sink") != std::string::npos);
  CHECK(graph.Describe(false).find("hidden") != std::string::npos);
  // 敏感值不进入能力清单的取值检查
  CHECK(graph.Requirements().Describe().find("hidden") == std::string::npos);
  CHECK_THROWS(graph.Add("fakesink", "out"), "duplicate element name");
}

TEST(drain_points_follow_each_source) {
  Graph graph;
  Element a = graph.Add("videotestsrc", "a");
  Element b = graph.Add("videotestsrc", "b");
  Element mix = graph.Add("funnel", "mix");
  graph.Link(a, graph.Add("queue", "qa"));
  graph.Link(graph.Find("qa"), mix);
  graph.Link(b, mix);
  graph.Link(mix, graph.Add("fakesink", "sink"));
  CHECK(graph.DrainPoints() == (std::vector<std::string>{"qa", "mix"}));
}

TEST(configured_properties_override_block_defaults) {
  Graph graph;
  Element sink = graph.Add("fakesink", "sink").Set("sync", true);
  sink.Apply({{"sync", "false", "outputs.properties.sync"}});
  CHECK_EQ(graph.nodes()[0].properties.size(), 1u);
  CHECK_EQ(graph.nodes()[0].properties[0].value, std::string("false"));
  CHECK_EQ(graph.nodes()[0].properties[0].origin, std::string("outputs.properties.sync"));
}

TEST(instantiate_reports_bad_properties_with_origin) {
  Graph graph;
  graph.Add("fakesink", "sink").Apply({{"no-such", "1", "x.properties.no-such"}});
  CHECK_THROWS(graph.Instantiate("p"), "x.properties.no-such: element 'fakesink' has no property");
  Graph missing;
  missing.Add("no-such-element", "e");
  CHECK_THROWS(missing.Instantiate("p"), "cannot create element 'e'");
}

TEST(pipeline_runs_to_eos_with_component_order) {
  std::vector<std::string> log;
  Recorder first("a", &log), second("b", &log);
  Pipeline pipeline("eos", TestGraph(10, false));
  pipeline.Add(first);
  pipeline.Add(second);
  const Result result = pipeline.Run();
  CHECK(result.success);
  CHECK_EQ(std::string(ToString(result.reason)), std::string("eos"));
  const std::vector<std::string> expected = {"a:created", "b:created", "a:playing", "b:playing",
                                             "b:released", "a:released"};
  CHECK(log == expected);
  CHECK_EQ(pipeline.Run().error, std::string("a Pipeline can only run once"));
}

TEST(drain_stop_finishes_as_stopped) {
  std::vector<std::string> log;
  Recorder component("c", &log);
  Pipeline pipeline("drain", TestGraph(0, true));
  pipeline.Add(component);
  std::atomic<int> ticks{0};
  Timer timer;
  pipeline.Post([&] {
    timer = pipeline.Every(std::chrono::milliseconds(20), [&] {
      if (++ticks == 5) pipeline.RequestStop();
    });
  });
  Options options;
  options.handle_signals = false;
  const Result result = pipeline.Run(options);
  CHECK(result.success);
  CHECK_EQ(std::string(ToString(result.reason)), std::string("stopped"));
  CHECK(std::find(log.begin(), log.end(), "c:stopping") != log.end());
  CHECK(ticks >= 5);
}

// 吞掉到达 sink 的 EOS，模拟收到 EOS 后不结束的网络 sink（例如 rtspclientsink）。
class SwallowEos : public Component {
 public:
  std::string_view Name() const override { return "swallow-eos"; }
  void OnPlaying(Pipeline& pipeline) override {
    // 在框架的 sink 监视探针之后添加：框架先看到 EOS，随后 EOS 被丢弃。
    GstPad* pad = gst_element_get_static_pad(pipeline.Element("sink"), "sink");
    gst_pad_add_probe(pad, GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM,
                      +[](GstPad*, GstPadProbeInfo* info, gpointer) {
                        return GST_EVENT_TYPE(GST_PAD_PROBE_INFO_EVENT(info)) == GST_EVENT_EOS
                                   ? GST_PAD_PROBE_DROP : GST_PAD_PROBE_OK;
                      },
                      nullptr, nullptr);
    gst_object_unref(pad);
  }
  bool Drained() const override { return drained; }
  bool drained = false;
};

TEST(drain_completes_when_eos_reaches_every_sink_and_components_are_done) {
  SwallowEos swallow;
  Pipeline pipeline("swallow", TestGraph(0, true));
  pipeline.Add(swallow);
  Timer stop;
  Timer finish_component;
  pipeline.Post([&] {
    stop = pipeline.Every(std::chrono::milliseconds(200), [&] { pipeline.RequestStop(); });
    // 组件的收尾稍后才完成：排空必须等它。
    finish_component = pipeline.Every(std::chrono::milliseconds(800), [&] { swallow.drained = true; });
  });
  Options options;
  options.handle_signals = false;
  options.stop_timeout = std::chrono::seconds(3);
  const auto started = std::chrono::steady_clock::now();
  const Result result = pipeline.Run(options);
  CHECK(result.success);
  CHECK_EQ(std::string(ToString(result.reason)), std::string("stopped"));
  CHECK(swallow.drained);  // EOS 早已到达 sink：只有等待组件收尾，结束时它才会是 true
  CHECK(std::chrono::steady_clock::now() - started < std::chrono::seconds(3));
}

// 两路 tee 分支：bad 分支的 sink 运行中报告错误但不阻塞数据流（rtspclientsink 断线时如此），
// good 分支等 EOS 到达才算收尾（模拟录像器封装最后一个分段）。
class FailingBranch : public Component {
 public:
  std::string_view Name() const override { return "failing-branch"; }
  void OnPlaying(Pipeline& pipeline) override {
    GstPad* pad = gst_element_get_static_pad(pipeline.Element("good"), "sink");
    gst_pad_add_probe(pad, GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM,
                      +[](GstPad*, GstPadProbeInfo* info, gpointer self) {
                        if (GST_EVENT_TYPE(GST_PAD_PROBE_INFO_EVENT(info)) == GST_EVENT_EOS) {
                          static_cast<FailingBranch*>(self)->good_eos = true;
                        }
                        return GST_PAD_PROBE_OK;
                      },
                      this, nullptr);
    gst_object_unref(pad);
    fail_ = pipeline.Every(std::chrono::milliseconds(200), [&pipeline, this] {
      if (failed_) return;
      failed_ = true;
      // 出错元件常连续报告多条错误。
      for (int i = 0; i < 2; ++i) {
        GST_ELEMENT_ERROR(pipeline.Element("bad"), RESOURCE, WRITE, ("connection lost"), (nullptr));
      }
    });
  }
  void OnStopping() override { stopping = true; }
  bool Drained() const override { return good_eos; }
  std::atomic<bool> good_eos{false};
  bool stopping = false;

 private:
  Timer fail_;
  bool failed_ = false;
};

Graph TwoBranchGraph() {
  Graph graph;
  Element source = graph.Add("videotestsrc", "src").Set("is-live", true);
  Element caps = graph.Caps("video/x-raw,width=64,height=48,framerate=30/1");
  graph.Link(source, caps);
  Element tee = graph.Tee(caps, "t");
  graph.Link(graph.Branch(tee, "qbad"), graph.Add("fakesink", "bad").Set("sync", true));
  graph.Link(graph.Branch(tee, "qgood"), graph.Add("fakesink", "good").Set("sync", true));
  return graph;
}

TEST(runtime_error_drains_first_when_drain_on_error_is_set) {
  FailingBranch branch;
  Pipeline pipeline("drain-on-error", TwoBranchGraph());
  pipeline.Add(branch);
  Options options;
  options.handle_signals = false;
  options.stop_timeout = std::chrono::seconds(3);
  options.drain_on_error = true;
  const auto started = std::chrono::steady_clock::now();
  const Result result = pipeline.Run(options);
  CHECK(!result.success);
  CHECK_EQ(std::string(ToString(result.reason)), std::string("run-error"));
  CHECK(result.error.find("connection lost") != std::string::npos);
  CHECK(result.error.find("(then") == std::string::npos);  // 排空完成，没有附加原因
  CHECK(branch.stopping);
  CHECK(branch.good_eos);
  CHECK(std::chrono::steady_clock::now() - started < std::chrono::seconds(3));
}

TEST(runtime_error_finishes_immediately_by_default) {
  FailingBranch branch;
  Pipeline pipeline("error", TwoBranchGraph());
  pipeline.Add(branch);
  Options options;
  options.handle_signals = false;
  const Result result = pipeline.Run(options);
  CHECK_EQ(std::string(ToString(result.reason)), std::string("run-error"));
  CHECK(result.error.find("connection lost") != std::string::npos);
  CHECK(!branch.stopping);
  CHECK(!branch.good_eos);
}

TEST(drain_on_error_reports_an_unfinished_drain) {
  // 吞掉 good 分支 sink 的 EOS（管线不会 EOS）且从不报告收尾：排空只能超时。
  struct NeverDrained : Component {
    std::string_view Name() const override { return "never-drained"; }
    void OnPlaying(Pipeline& pipeline) override {
      GstPad* pad = gst_element_get_static_pad(pipeline.Element("good"), "sink");
      gst_pad_add_probe(pad, GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM,
                        +[](GstPad*, GstPadProbeInfo* info, gpointer) {
                          return GST_EVENT_TYPE(GST_PAD_PROBE_INFO_EVENT(info)) == GST_EVENT_EOS
                                     ? GST_PAD_PROBE_DROP : GST_PAD_PROBE_OK;
                        },
                        nullptr, nullptr);
      gst_object_unref(pad);
    }
    bool Drained() const override { return false; }
  };
  FailingBranch branch;
  NeverDrained stuck;
  Pipeline pipeline("drain-on-error-timeout", TwoBranchGraph());
  pipeline.Add(branch);
  pipeline.Add(stuck);
  Options options;
  options.handle_signals = false;
  options.stop_timeout = std::chrono::milliseconds(500);
  options.drain_on_error = true;
  const Result result = pipeline.Run(options);
  CHECK_EQ(std::string(ToString(result.reason)), std::string("run-error"));
  CHECK(result.error.find("connection lost") != std::string::npos);
  CHECK(result.error.find(" (then forced: draining did not finish") != std::string::npos);
}

TEST(plugin_timers_on_the_default_context_fire) {
  // 插件用 g_timeout_add 在默认 context 上注册的定时器必须在 Run 期间触发：
  // 这里由这样的定时器触发 5 次后请求停止，管线只有在定时器真的运行时才会结束。
  struct State {
    Pipeline* pipeline;
    int fired = 0;
  };
  Pipeline pipeline("default-context", TestGraph(0, true));
  State state{&pipeline};
  const guint id = g_timeout_add(20, +[](gpointer data) -> gboolean {
    auto* s = static_cast<State*>(data);
    if (++s->fired == 5) s->pipeline->RequestStop();
    return G_SOURCE_CONTINUE;
  }, &state);
  Options options;
  options.handle_signals = false;
  options.startup_timeout = std::chrono::seconds(5);
  options.stop_timeout = std::chrono::seconds(5);
  const Result result = pipeline.Run(options);
  g_source_remove(id);
  CHECK(result.success);
  CHECK(state.fired >= 5);
}

TEST(positive_startup_timeout_fails_without_frames) {
  RequirePlugin("appsrc");
  Pipeline pipeline("idle-timeout", IdleLiveGraph());
  Options options;
  options.handle_signals = false;
  options.startup_timeout = std::chrono::milliseconds(300);
  const Result result = pipeline.Run(options);
  CHECK(!result.success);
  CHECK_EQ(std::string(ToString(result.reason)), std::string("startup-timeout"));
}

TEST(negative_startup_timeout_is_rejected) {
  std::vector<std::string> log;
  Recorder component("c", &log);
  Pipeline pipeline("negative", TestGraph(5, false));
  pipeline.Add(component);
  Options options;
  options.startup_timeout = std::chrono::milliseconds(-1);
  const Result result = pipeline.Run(options);
  CHECK(!result.success);
  CHECK_EQ(std::string(ToString(result.reason)), std::string("start-error"));
  CHECK(log == std::vector<std::string>{"c:released"});
}

TEST(zero_startup_timeout_waits_for_frames_and_stops_cleanly) {
  RequirePlugin("appsrc");
  std::vector<std::string> log;
  Recorder component("c", &log);
  Pipeline pipeline("idle-wait", IdleLiveGraph());
  pipeline.Add(component);
  std::atomic<int> ticks{0};
  Timer timer;
  pipeline.Post([&] {
    // 超过正数超时测试的等待时长仍未出帧：不应超时，停止请求照常排空。
    timer = pipeline.Every(std::chrono::milliseconds(100), [&] {
      if (++ticks == 6) pipeline.RequestStop();
    });
  });
  Options options;
  options.handle_signals = false;
  options.startup_timeout = std::chrono::milliseconds(0);
  options.stop_timeout = std::chrono::seconds(3);
  const Result result = pipeline.Run(options);
  CHECK(result.success);
  CHECK_EQ(std::string(ToString(result.reason)), std::string("stopped"));
  CHECK(std::find(log.begin(), log.end(), "c:playing") == log.end());
  CHECK(std::find(log.begin(), log.end(), "c:stopping") != log.end());
}

// 首帧到达后才进入 PLAYING：OnPlaying 中请求停止。
class StopWhenPlaying : public Component {
 public:
  std::string_view Name() const override { return "stop-when-playing"; }
  void OnPlaying(Pipeline& pipeline) override {
    playing = true;
    pipeline.RequestStop();
  }
  bool playing = false;
};

TEST(zero_startup_timeout_plays_once_frames_arrive) {
  RequirePlugin("appsrc");
  StopWhenPlaying component;
  Pipeline pipeline("late-frames", IdleLiveGraph());
  pipeline.Add(component);
  std::atomic<int> ticks{0};
  Timer timer;
  pipeline.Post([&] {
    timer = pipeline.Every(std::chrono::milliseconds(100), [&] {
      if (++ticks >= 5) PushFrame(pipeline.Element("src"));
      // 回归失败时也应有界退出，避免零启动超时使测试一直等待。
      if (ticks >= 30) pipeline.RequestStop(StopMode::kImmediate);
    });
  });
  Options options;
  options.handle_signals = false;
  options.startup_timeout = std::chrono::milliseconds(0);
  options.stop_timeout = std::chrono::seconds(3);
  const Result result = pipeline.Run(options);
  CHECK(result.success);
  CHECK(component.playing);
  CHECK(ticks >= 5);
}

TEST(private_context_pipelines_run_in_parallel) {
  Options options;
  options.handle_signals = false;
  options.private_context = true;
  Pipeline a("a", TestGraph(20, false)), b("b", TestGraph(20, false));
  Result ra, rb;
  std::thread other([&] { rb = b.Run(options); });
  ra = a.Run(options);
  other.join();
  CHECK(ra.success && rb.success);
}

TEST(immediate_stop_is_forced_failure) {
  Pipeline pipeline("forced", TestGraph(0, true));
  std::thread stopper([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    pipeline.RequestStop(StopMode::kImmediate);
  });
  Options options;
  options.handle_signals = false;
  const Result result = pipeline.Run(options);
  stopper.join();
  CHECK(!result.success);
  CHECK_EQ(std::string(ToString(result.reason)), std::string("forced"));
}

TEST(component_failure_does_not_skip_cleanup) {
  std::vector<std::string> log;
  Recorder failing("f", &log), other("o", &log);
  failing.throw_on_created = true;
  Pipeline pipeline("fail", TestGraph(5, false));
  pipeline.Add(failing);
  pipeline.Add(other);
  const Result result = pipeline.Run();
  CHECK(!result.success);
  CHECK(result.error.find("f: boom") != std::string::npos);
  CHECK(std::find(log.begin(), log.end(), "o:released") != log.end());
  CHECK(std::find(log.begin(), log.end(), "f:released") != log.end());
}

TEST(runtime_errors_and_build_errors_are_classified) {
  Graph broken;
  broken.Add("no-such-element", "x");
  Pipeline build("build", broken);
  CHECK_EQ(std::string(ToString(build.Run().reason)), std::string("build-error"));

  Pipeline launch("launch", Pipeline::Launch("videotestsrc num-buffers=3 ! fakesink"));
  CHECK(launch.Run().success);
}

TEST(sometimes_pads_are_linked_when_they_appear) {
  RequirePlugin("decodebin");
  Graph graph;
  Element source = graph.Add("videotestsrc", "src").Set("num-buffers", 5);
  Element decode = graph.Add("decodebin", "decode");
  Element convert = graph.Add("videoconvert", "convert");
  graph.Chain({source, decode, convert, graph.Add("fakesink", "sink")});
  Pipeline pipeline("delayed", graph);
  const Result result = pipeline.Run();
  CHECK(result.success);
}

TEST(link_failure_with_unlinked_static_pad_is_a_build_error) {
  EnsureInitialized();
  Graph graph;
  graph.Chain({graph.Add(RegisterDualSource(), "dual"), graph.Caps("audio/x-raw", "audio")});
  CHECK_THROWS(gst_object_unref(graph.Instantiate("dual")), "cannot link dual -> audio");
}

TEST(tasks_posted_before_run_execute_on_the_private_context) {
  Pipeline pipeline("early", TestGraph(0, true));
  std::thread::id ran_on;
  pipeline.Post([&] {
    ran_on = std::this_thread::get_id();
    pipeline.RequestStop();
  });
  Options options;
  options.handle_signals = false;
  options.private_context = true;
  // 任务若留在默认 context 上就永不执行，live 管线停不下来（由 CTest 超时判为失败）。
  const Result result = pipeline.Run(options);
  CHECK(result.success);
  CHECK_EQ(std::string(ToString(result.reason)), std::string("stopped"));
  CHECK(ran_on == std::this_thread::get_id());
}

TEST(request_pads_by_template_and_tee_branches) {
  Graph graph;
  Element source = graph.Add("videotestsrc", "src").Set("num-buffers", 5);
  Element tee = graph.Tee(source, "t");
  graph.Link(graph.Branch(tee), graph.Add("fakesink", "a"));
  graph.Link(graph.Branch(tee), graph.Add("fakesink", "b"));
  Pipeline pipeline("tee", graph);
  CHECK(pipeline.Run().success);
}

DSF_TEST_MAIN()
