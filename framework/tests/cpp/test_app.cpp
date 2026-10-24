// dsf::app：命令行、日志、自检与完整入口流程（不需要 DeepStream 插件的软件应用）。

#include <cstdlib>
#include <filesystem>
#include <fstream>

#include "dsf/app/app.h"
#include "dsf/app/components.h"
#include "dsf/ds/blocks.h"
#include "dsf/gst/introspect.h"
#include "dsf/gst/pipeline.h"
#include <unistd.h>

#include "dsf_test.h"

namespace fs = std::filesystem;
using namespace dsf::app;

namespace {

CommandLine Parse(std::vector<std::string> args) {
  args.insert(args.begin(), "app");
  std::vector<char*> argv;
  for (auto& arg : args) argv.push_back(arg.data());
  return ParseCommandLine(static_cast<int>(argv.size()), argv.data());
}

// 一个只用软件元件的最小应用：验证入口在应用侧需要提供的全部内容。
struct SoftwareApp {
  struct Config {
    int frames = 0;
  };
  static Info Describe() { return {"soft-app", "software test app", [] { return "extra=1"; }}; }
  static Config Load(dsf::config::Reader& root) {
    Config config;
    config.frames = root.Get("frames", 10, dsf::config::AtLeast(1));
    return config;
  }
  static void Build(const Config& config, Assembly& assembly) {
    auto& graph = assembly.graph();
    auto source = graph.Add("videotestsrc", "src").Set("num-buffers", config.frames);
    auto tap = graph.Add("identity", "tap");
    graph.Chain({source, tap, graph.Add("fakesink", "sink")});
    assembly.ReadyWhenFramesReach("tap");
  }
};

struct TempConfig {
  explicit TempConfig(const std::string& text) {
    dir = fs::temp_directory_path() / ("dsf-app-" + std::to_string(::getpid()));
    fs::create_directories(dir);
    path = dir / "soft-app.yaml";
    std::ofstream(path) << text;
  }
  ~TempConfig() { fs::remove_all(dir); }
  fs::path dir;
  fs::path path;
};

int RunApp(std::vector<std::string> args) {
  args.insert(args.begin(), "soft-app");
  std::vector<char*> argv;
  for (auto& arg : args) argv.push_back(arg.data());
  return Main<SoftwareApp>(static_cast<int>(argv.size()), argv.data());
}

// 不出帧时，异步 fakesink 等待 preroll，使管线保持在进入 PLAYING 之前。
dsf::gst::Graph IdleLiveGraph() {
  dsf::gst::Graph graph;
  graph.Chain({graph.Add("appsrc", "src")
                   .Set("is-live", true)
                   .Set("format", "time")
                   .Set("caps", "video/x-raw,format=GRAY8,width=64,height=48,framerate=0/1"),
               graph.Add("fakesink", "sink").Set("sync", false).Set("async", true)});
  return graph;
}

}  // namespace

TEST(command_line_modes_are_exclusive) {
  CHECK(Parse({}).mode == Mode::kRun);
  CHECK(Parse({"--check-config", "--config", "/x.yaml"}).config_path == "/x.yaml");
  CHECK(Parse({"--check-config", "--plugins"}).plugins);
  CHECK_THROWS(Parse({"--check-config", "--dump-pipeline"}), "conflicting option");
  CHECK_THROWS(Parse({"--config"}), "requires a path");
  CHECK_THROWS(Parse({"--plugins"}), "only valid with --check-config");
  CHECK_THROWS(Parse({"--bogus"}), "unknown argument");
}

TEST(paths_come_from_environment) {
  setenv("CONTAINER_CONFIG_PATH", "/etc/x", 1);
  CHECK_EQ(Paths::FromEnvironment().config, std::string("/etc/x"));
  unsetenv("CONTAINER_CONFIG_PATH");
  CHECK_EQ(Paths::FromEnvironment().config, std::string("/config"));
}

TEST(logging_options_are_validated) {
  const auto document = dsf::config::Document::LoadText("logging:\n  file: true\n", "t");
  dsf::config::Reader root = document.Root();
  CHECK_THROWS(LogOptions::Read(root.Section("logging")), "logging.directory: is required");
  LogOptions options;
  options.level = "loud";
  CHECK_THROWS(ValidateLogging(options), "unknown log level");
}

TEST(build_info_reports_payload_and_framework) {
  const fs::path dir = fs::temp_directory_path() / "dsf-app-version";
  fs::create_directories(dir);
  std::ofstream(dir / "VERSION") << "1.2.3\n";
  Paths paths;
  paths.workspace = dir.string();
  const std::string text = BuildInfo(SoftwareApp::Describe(), paths);
  CHECK(text.find("soft-app payload=1.2.3 framework=") != std::string::npos);
  CHECK(text.find("gstreamer=GStreamer") != std::string::npos);
  CHECK(text.find("extra=1") != std::string::npos);
  fs::remove_all(dir);
}

TEST(selftest_passes_on_software_pipeline) { CHECK_EQ(RunSelfTest(), 0); }

TEST(entry_point_modes) {
  TempConfig good("frames: 5\nlogging:\n  level: warn\n");
  CHECK_EQ(RunApp({"--check-config", "--config", good.path.string()}), 0);
  CHECK_EQ(RunApp({"--check-config", "--plugins", "--config", good.path.string()}), 0);
  CHECK_EQ(RunApp({"--dump-pipeline", "--config", good.path.string()}), 0);
  CHECK_EQ(RunApp({"--bogus"}), 2);
  CHECK_EQ(RunApp({"--config", "/no/such/file.yaml"}), 1);

  TempConfig unknown("frames: 5\nframe: 1\n");
  CHECK_EQ(RunApp({"--check-config", "--config", unknown.path.string()}), 1);
}

TEST(readiness_and_stats_run_before_playing) {
  if (!dsf::gst::HasFactory("appsrc")) SKIP("GStreamer element missing: appsrc");
  struct Observer : dsf::gst::Component {
    std::string_view Name() const override { return "observer"; }
    void OnPlaying(dsf::gst::Pipeline&) override { playing = true; }
    void Report(dsf::gst::Stats&) const override { ++reports; }
    bool playing = false;
    mutable int reports = 0;
  } observer, disabled_observer;
  TempConfig config("frames: 1\n");
  const fs::path ready = config.dir / "ready";
  ReadinessHeartbeat readiness("", nullptr, ready.string(), std::chrono::milliseconds(50));
  StatsReporter stats(std::chrono::seconds(1), {&observer}, {});
  StatsReporter disabled_stats(std::chrono::seconds(0), {&disabled_observer}, {});
  dsf::gst::Pipeline pipeline("idle-ready", IdleLiveGraph());
  pipeline.Add(observer);
  pipeline.Add(readiness);
  pipeline.Add(stats);
  pipeline.Add(disabled_stats);
  int updates = 0;
  int ticks = 0;
  dsf::gst::Timer timer;
  pipeline.Post([&] {
    timer = pipeline.Every(std::chrono::milliseconds(200), [&] {
      // 删除后应再次创建，验证持续更新而非只写一次，也不依赖文件系统时间精度。
      if (fs::remove(ready)) ++updates;
      if (updates >= 2 && observer.reports > 0) pipeline.RequestStop();
      else if (++ticks >= 15) pipeline.RequestStop(dsf::gst::StopMode::kImmediate);
    });
  });
  dsf::gst::Options options;
  options.handle_signals = false;
  options.startup_timeout = std::chrono::milliseconds(0);
  const dsf::gst::Result result = pipeline.Run(options);
  CHECK(result.success);
  CHECK(!observer.playing);
  CHECK(updates >= 2);
  CHECK(observer.reports > 0);
  CHECK_EQ(disabled_observer.reports, 0);
  CHECK(!fs::exists(ready));
}

TEST(counted_readiness_only_updates_when_progress_changes) {
  if (!dsf::gst::HasFactory("appsrc")) SKIP("GStreamer element missing: appsrc");
  // 分别覆盖计数回调和元件探针；计数不变时不能退化成无条件心跳。
  for (const bool use_callback : {true, false}) {
    TempConfig config("frames: 1\n");
    const fs::path ready = config.dir / "ready";
    std::uint64_t progress = 0;
    std::function<std::uint64_t()> counter;
    if (use_callback) counter = [&] { return progress; };
    ReadinessHeartbeat readiness(use_callback ? "" : "src", counter, ready.string(),
                                 std::chrono::milliseconds(50));
    dsf::gst::Pipeline pipeline("counted-ready", IdleLiveGraph());
    pipeline.Add(readiness);
    int phase = 0;
    int idle_checks = 0;
    int ticks = 0;
    bool quiet_while_idle = true;
    bool counts_match = true;
    bool pushed = true;
    const auto advance = [&] {
      if (use_callback) {
        ++progress;
      } else {
        GstBuffer* buffer = gst_buffer_new_allocate(nullptr, 64 * 48, nullptr);
        GstFlowReturn flow = GST_FLOW_OK;
        g_signal_emit_by_name(pipeline.Element("src"), "push-buffer", buffer, &flow);
        gst_buffer_unref(buffer);
        pushed = pushed && flow == GST_FLOW_OK;
      }
    };
    dsf::gst::Timer timer;
    pipeline.Post([&] {
      timer = pipeline.Every(std::chrono::milliseconds(100), [&] {
        if (++ticks >= 40) {
          pipeline.RequestStop(dsf::gst::StopMode::kImmediate);
          return;
        }
        if (phase == 0 || phase == 2) {
          quiet_while_idle = quiet_while_idle && !fs::exists(ready);
          if (++idle_checks == 3) {
            advance();
            ++phase;
          }
        } else if ((phase == 1 || phase == 3) && fs::exists(ready)) {
          std::uint64_t count = 0;
          std::ifstream(ready) >> count;
          counts_match = counts_match && count == (phase == 1 ? 1u : 2u);
          fs::remove(ready);
          idle_checks = 0;
          if (++phase == 4) pipeline.RequestStop();
        }
      });
    });
    dsf::gst::Options options;
    options.handle_signals = false;
    options.startup_timeout = std::chrono::milliseconds(0);
    const dsf::gst::Result result = pipeline.Run(options);
    CHECK(result.success);
    CHECK_EQ(phase, 4);
    CHECK(quiet_while_idle);
    CHECK(counts_match);
    CHECK(pushed);
    CHECK(!fs::exists(ready));
  }
}

TEST(stalled_progress_stops_the_pipeline_and_reports_unhealthy) {
  if (!dsf::gst::HasFactory("appsrc")) SKIP("GStreamer element missing: appsrc");
  TempConfig config("frames: 1\n");
  ReadinessHeartbeat readiness("", [] { return std::uint64_t{0}; }, (config.dir / "ready").string(),
                               std::chrono::milliseconds(50), std::chrono::milliseconds(300));
  dsf::gst::Pipeline pipeline("stalled", IdleLiveGraph());
  pipeline.Add(readiness);
  dsf::gst::Timer guard;
  pipeline.Post([&] {
    guard = pipeline.Every(std::chrono::seconds(5),
                           [&] { pipeline.RequestStop(dsf::gst::StopMode::kImmediate); });
  });
  dsf::gst::Options options;
  options.handle_signals = false;
  options.startup_timeout = std::chrono::milliseconds(0);
  const auto begin = std::chrono::steady_clock::now();
  const dsf::gst::Result result = pipeline.Run(options);
  CHECK(result.success);  // 排空停止，不是 guard 的强制停止
  CHECK(std::chrono::steady_clock::now() - begin < std::chrono::seconds(4));
  CHECK(!readiness.Healthy());

  // 计数增长或没有设置时限时不触发。
  ReadinessHeartbeat disabled("", [] { return std::uint64_t{0}; }, (config.dir / "ready2").string(),
                              std::chrono::milliseconds(50));
  dsf::gst::Pipeline idle("idle", IdleLiveGraph());
  idle.Add(disabled);
  dsf::gst::Timer stop;
  idle.Post([&] { stop = idle.Every(std::chrono::milliseconds(500), [&] { idle.RequestStop(); }); });
  CHECK(idle.Run(options).success);
  CHECK(disabled.Healthy());
}

TEST(watchdog_section_is_read_by_the_framework) {
  {
    TempConfig valid("frames: 1\nwatchdog:\n  stall_timeout_s: 120\n");
    CHECK_EQ(RunApp({"--check-config", "--config", valid.path.string()}), 0);
  }
  TempConfig negative("frames: 1\nwatchdog:\n  stall_timeout_s: -1\n");
  CHECK_EQ(RunApp({"--check-config", "--config", negative.path.string()}), 1);
}

TEST(entry_point_runs_and_cleans_readiness_file) {
  TempConfig config("frames: 20\nlogging:\n  level: warn\n  stats_interval_s: 0\n");
  const fs::path ready = config.dir / "ready";
  setenv("DSF_READY_FILE", ready.c_str(), 1);
  CHECK_EQ(RunApp({"--config", config.path.string()}), 0);
  CHECK(!fs::exists(ready));
  unsetenv("DSF_READY_FILE");
}

DSF_TEST_MAIN()
