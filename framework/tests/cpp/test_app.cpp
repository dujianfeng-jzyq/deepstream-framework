// dsf::app：命令行、日志、自检与完整入口流程（不需要 DeepStream 插件的软件应用）。

#include <cstdlib>
#include <filesystem>
#include <fstream>

#include "dsf/app/app.h"
#include "dsf/app/components.h"
#include "dsf/ds/blocks.h"
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

TEST(entry_point_runs_and_cleans_readiness_file) {
  TempConfig config("frames: 20\nlogging:\n  level: warn\n  stats_interval_s: 0\n");
  const fs::path ready = config.dir / "ready";
  setenv("DSF_READY_FILE", ready.c_str(), 1);
  CHECK_EQ(RunApp({"--config", config.path.string()}), 0);
  CHECK(!fs::exists(ready));
  unsetenv("DSF_READY_FILE");
}

DSF_TEST_MAIN()
