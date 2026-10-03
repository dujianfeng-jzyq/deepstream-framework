// 应用入口的实现。

#include "dsf/app/app.h"

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "dsf/app/components.h"
#include "dsf/gst/introspect.h"

namespace dsf::app {
namespace {

std::string EnvOr(const char* name, const std::string& fallback) {
  const char* value = std::getenv(name);
  return value != nullptr && *value != '\0' ? std::string(value) : fallback;
}

const std::vector<std::pair<std::string, spdlog::level::level_enum>>& Levels() {
  static const std::vector<std::pair<std::string, spdlog::level::level_enum>> levels = {
      {"trace", spdlog::level::trace}, {"debug", spdlog::level::debug},
      {"info", spdlog::level::info},   {"warn", spdlog::level::warn},
      {"error", spdlog::level::err},   {"critical", spdlog::level::critical},
      {"off", spdlog::level::off},
  };
  return levels;
}

std::string Trim(std::string text) {
  const std::size_t begin = text.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) return {};
  return text.substr(begin, text.find_last_not_of(" \t\r\n") - begin + 1);
}

// 自检：videotestsrc 固定 30 帧，handoff 计数与自然 EOS 两项都满足才算通过；
// 提前结束或丢帧都必须失败，否则自检会在损坏的运行环境上“通过”。
constexpr int kSelfTestFrames = 30;

class HandoffCounter : public gst::Component {
 public:
  std::string_view Name() const override { return "selftest-counter"; }
  void OnCreated(gst::Pipeline& /*pipeline*/, GstElement* bin) override {
    GstElement* sink = gst_bin_get_by_name(GST_BIN(bin), "sink");
    if (sink == nullptr) throw std::runtime_error("selftest sink is missing");
    g_signal_connect(sink, "handoff",
                     G_CALLBACK(+[](GstElement*, GstBuffer*, GstPad*, gpointer data) {
                       ++*static_cast<std::atomic<int>*>(data);
                     }),
                     &frames);
    gst_object_unref(sink);
  }
  std::atomic<int> frames{0};
};

}  // namespace

Paths Paths::FromEnvironment() {
  Paths paths;
  paths.workspace = EnvOr("CONTAINER_WORKSPACE", paths.workspace);
  paths.config = EnvOr("CONTAINER_CONFIG_PATH", paths.config);
  paths.data = EnvOr("CONTAINER_DATA_PATH", paths.data);
  paths.models = EnvOr("CONTAINER_MODELS_PATH", paths.models);
  return paths;
}

LogOptions LogOptions::Read(config::Reader reader) {
  LogOptions options;
  std::vector<std::string> names;
  for (const auto& [name, level] : Levels()) names.push_back(name);
  options.level = reader.Get<std::string>("level", "info", config::OneOf(names));
  options.console = reader.Get("console", true);
  options.file = reader.Get("file", false);
  options.directory = reader.Get<std::string>("directory", "");
  options.stats_interval_s = reader.Get("stats_interval_s", 60, config::AtLeast(0));
  if (options.file && options.directory.empty()) {
    reader.Fail("directory", "is required when file logging is enabled");
  }
  return options;
}

void ValidateLogging(const LogOptions& options) {
  bool known = false;
  for (const auto& [name, level] : Levels()) known = known || name == options.level;
  if (!known) throw std::invalid_argument("unknown log level: " + options.level);
  if (options.file && options.directory.empty()) {
    throw std::invalid_argument("logging.directory is required when file logging is enabled");
  }
}

void ConfigureLogging(const LogOptions& options, const std::string& name) {
  ValidateLogging(options);
  std::vector<spdlog::sink_ptr> sinks;
  // 两种输出都关闭时仍保留控制台：完全没有日志的进程在现场无法排查。
  if (options.console || !options.file) {
    sinks.push_back(std::make_shared<spdlog::sinks::stdout_color_sink_mt>());
  }
  if (options.file) {
    std::filesystem::create_directories(options.directory);
    const auto path = std::filesystem::path(options.directory) / (name + ".log");
    sinks.push_back(std::make_shared<spdlog::sinks::basic_file_sink_mt>(path.string()));
  }
  auto logger = std::make_shared<spdlog::logger>(name, sinks.begin(), sinks.end());
  for (const auto& [level_name, level] : Levels()) {
    if (level_name == options.level) logger->set_level(level);
  }
  logger->flush_on(spdlog::level::warn);  // 出问题时最后几行必须已经落地
  spdlog::set_default_logger(logger);
  spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] [%n] %v");
}

publish::Publisher& Assembly::AddPublisher(std::unique_ptr<publish::Publisher> publisher) {
  publishers_.push_back(std::move(publisher));
  return *publishers_.back();
}

gst::Requirements Assembly::Requirements() const {
  gst::Requirements requirements = graph_.Requirements();
  for (const auto& component : components_) component->Require(requirements);
  return requirements;
}

CommandLine ParseCommandLine(int argc, char** argv) {
  CommandLine line;
  bool chosen = false;
  const auto choose = [&](Mode mode, const std::string& flag) {
    if (chosen && line.mode != mode) throw std::invalid_argument("conflicting option: " + flag);
    line.mode = mode;
    chosen = true;
  };
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "-h" || arg == "--help") {
      choose(Mode::kHelp, arg);
    } else if (arg == "--version") {
      choose(Mode::kVersion, arg);
    } else if (arg == "--check-config") {
      choose(Mode::kCheckConfig, arg);
    } else if (arg == "--dump-pipeline") {
      choose(Mode::kDumpPipeline, arg);
    } else if (arg == "--selftest") {
      choose(Mode::kSelfTest, arg);
    } else if (arg == "--plugins") {
      line.plugins = true;
    } else if (arg == "--config") {
      if (i + 1 >= argc) throw std::invalid_argument("--config requires a path");
      line.config_path = argv[++i];
    } else {
      throw std::invalid_argument("unknown argument: " + arg);
    }
  }
  if (line.plugins && line.mode != Mode::kCheckConfig) {
    throw std::invalid_argument("--plugins is only valid with --check-config");
  }
  return line;
}

std::string Usage(const Info& info, const Paths& paths) {
  std::string text = "Usage: " + info.name + " [options]\n";
  if (!info.description.empty()) text += "\n" + info.description + "\n";
  text += "\nOptions:\n";
  text += "  --config PATH            配置文件（默认 " + paths.config + "/" + info.name + ".yaml）\n";
  text += "  --check-config           校验配置并组图，不连接设备、不写文件\n";
  text += "  --check-config --plugins 另外按实际插件校验元件与属性\n";
  text += "  --dump-pipeline          输出管线图与插件需求（敏感值打码）\n";
  text += "  --selftest               运行 30 帧软件自检（不需要 GPU、相机与配置）\n";
  text += "  --version                输出载荷、框架与运行时版本\n";
  text += "  -h, --help               显示本帮助\n";
  return text;
}

std::string BuildInfo(const Info& info, const Paths& paths) {
  std::ifstream stream(paths.workspace + "/VERSION");
  std::string payload;
  std::getline(stream, payload);
  payload = Trim(payload);
  std::string text = info.name + " payload=" + (payload.empty() ? "unknown" : payload) +
                     " framework=" + DSF_VERSION + " gstreamer=" + gst::GStreamerVersion();
  if (info.versions) {
    const std::string extra = info.versions();
    if (!extra.empty()) text += " " + extra;
  }
  return text;
}

int RunSelfTest() {
  gst::Graph graph;
  graph.Add("videotestsrc", "source").Set("num-buffers", kSelfTestFrames);
  graph.Caps("video/x-raw,format=RGB,width=320,height=240,framerate=30/1", "caps");
  graph.Add("videoconvert", "convert");
  graph.Add("fakesink", "sink").Set("signal-handoffs", true).Set("sync", false);
  graph.Link(graph.Find("source"), graph.Find("caps"));
  graph.Link(graph.Find("caps"), graph.Find("convert"));
  graph.Link(graph.Find("convert"), graph.Find("sink"));

  HandoffCounter counter;
  gst::Pipeline pipeline("selftest", graph);
  pipeline.Add(counter);
  gst::Options options;
  options.startup_timeout = std::chrono::seconds(10);
  const gst::Result result = pipeline.Run(options);
  if (!result.success || result.reason != gst::FinishReason::kEos) {
    spdlog::error("selftest failed: {} {}", gst::ToString(result.reason), result.error);
    return kExitFailure;
  }
  if (counter.frames != kSelfTestFrames) {
    spdlog::error("selftest failed: processed {} of {} frames", counter.frames.load(), kSelfTestFrames);
    return kExitFailure;
  }
  spdlog::info("selftest ok: {} frames", kSelfTestFrames);
  return kExitOk;
}

int Run(int argc, char** argv, const Definition& definition) {
  const Info& info = definition.info;
  const Paths paths = Paths::FromEnvironment();
  CommandLine line;
  try {
    line = ParseCommandLine(argc, argv);
  } catch (const std::invalid_argument& error) {
    std::cerr << info.name << ": " << error.what() << "\n\n" << Usage(info, paths);
    return kExitUsage;
  }
  switch (line.mode) {
    case Mode::kHelp:
      std::cout << Usage(info, paths);
      return kExitOk;
    case Mode::kVersion:
      std::cout << BuildInfo(info, paths) << "\n";
      return kExitOk;
    case Mode::kSelfTest:
      ConfigureLogging(LogOptions{}, info.name);
      return RunSelfTest();
    default:
      break;
  }

  // 读取配置并组装。到这里为止没有任何副作用：预检与诊断只走到这一步。
  const std::string config_path =
      line.config_path.empty() ? paths.config + "/" + info.name + ".yaml" : line.config_path;
  LogOptions logging;
  std::string summary;
  Assembly assembly(paths);
  try {
    const config::Document document = config::Document::LoadFile(config_path);
    config::Reader root = document.Root();
    logging = LogOptions::Read(root.Section("logging"));
    definition.load(root);
    root.Finish();
    summary = root.Describe();
    definition.build(assembly);
  } catch (const std::exception& error) {
    std::cerr << info.name << ": " << error.what() << "\n";
    return kExitFailure;
  }
  const gst::Requirements requirements = assembly.Requirements();

  if (line.mode == Mode::kCheckConfig) {
    if (line.plugins) {
      try {
        requirements.Check();
      } catch (const std::exception& error) {
        std::cerr << info.name << ": " << error.what() << "\n";
        return kExitFailure;
      }
    }
    std::cout << "configuration is valid\n";
    return kExitOk;
  }
  if (line.mode == Mode::kDumpPipeline) {
    std::cout << "# pipeline " << info.name << " (" << config_path << ")\n"
              << assembly.graph().Describe() << "\n# requirements\n" << requirements.Describe();
    return kExitOk;
  }

  // ---- 运行 ----
  try {
    ConfigureLogging(logging, info.name);
  } catch (const std::exception& error) {
    std::cerr << info.name << ": cannot configure logging: " << error.what() << "\n";
    return kExitFailure;
  }
  spdlog::info("{}", BuildInfo(info, paths));
  spdlog::info("config({}): {}", config_path, summary);
  try {
    requirements.Check();
  } catch (const std::exception& error) {
    spdlog::error("{}", error.what());
    return kExitFailure;
  }

  std::vector<const gst::Component*> reported;
  for (const auto& component : assembly.components()) reported.push_back(component.get());
  std::vector<const publish::Channel*> channels;
  for (const auto& publisher : assembly.publishers()) {
    for (const publish::Channel* channel : publisher->Channels()) channels.push_back(channel);
  }
  ReadinessHeartbeat readiness(assembly.ready_element(), assembly.ready_progress(), ReadyFilePath());
  reported.push_back(&readiness);
  StatsReporter stats(std::chrono::seconds(logging.stats_interval_s), reported, channels);

  gst::Pipeline pipeline(info.name, assembly.graph());
  for (const auto& component : assembly.components()) pipeline.Add(*component);
  pipeline.Add(readiness);
  pipeline.Add(stats);

  // 发布器先于管线启动、晚于管线停止：排空阶段产生的最后一批事件仍能发出。
  for (const auto& publisher : assembly.publishers()) {
    spdlog::info("publisher: {}", publisher->Describe());
    publisher->Start();
  }
  const gst::Result result = pipeline.Run(assembly.options());
  for (const auto& publisher : assembly.publishers()) publisher->Stop();

  spdlog::info("final stats: {}", stats.Line());
  bool healthy = true;
  for (const auto& component : assembly.components()) {
    if (!component->Healthy()) {
      spdlog::error("component {} did not finish cleanly", component->Name());
      healthy = false;
    }
  }
  return result.success && healthy ? kExitOk : kExitFailure;
}

}  // namespace dsf::app
