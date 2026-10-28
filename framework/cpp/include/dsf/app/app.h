// 应用入口。应用只提供“读取配置”和“组装”两步，其余由框架统一处理：
//
//   struct MyApp {
//     using Config = my::Config;
//     static dsf::app::Info Describe();
//     static Config Load(dsf::config::Reader& root);           // 读取并校验；不得有副作用
//     static void Build(const Config&, dsf::app::Assembly&);   // 组图、登记组件与发布器；不得有副作用
//   };
//   int main(int argc, char** argv) { return dsf::app::Main<MyApp>(argc, argv); }
//
// CLI：--config PATH、--check-config [--plugins]、--dump-pipeline、--selftest、--version、--help。
// 退出码：0 成功，1 失败，2 用法错误。配置中的 logging 与 watchdog 段由框架读取，所有应用一致。

#ifndef DSF_APP_APP_H
#define DSF_APP_APP_H

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "dsf/config/config.h"
#include "dsf/gst/component.h"
#include "dsf/gst/graph.h"
#include "dsf/gst/pipeline.h"
#include "dsf/publish/publisher.h"

namespace dsf::app {

enum ExitCode { kExitOk = 0, kExitFailure = 1, kExitUsage = 2 };

// 容器内的固定挂载点，来自环境变量 CONTAINER_*；未设置时使用部署约定的默认值。
struct Paths {
  std::string workspace = "/workspace";  // 只读载荷
  std::string config = "/config";        // 只读的渲染配置
  std::string data = "/data";            // 可写的持久数据
  std::string models = "/models";        // 只读的模型
  static Paths FromEnvironment();
};

struct Info {
  std::string name;         // 决定默认配置文件 <config>/<name>.yaml 与日志名
  std::string description;  // --help 中的一行说明
  // 追加到 --version 的运行时信息，例如 "deepstream=8.0"。
  std::function<std::string()> versions;
};

struct LogOptions {
  std::string level = "info";
  bool console = true;
  bool file = false;
  std::string directory;  // 文件日志目录，应位于数据目录下
  int stats_interval_s = 60;  // 组件统计的输出周期，0 表示不输出

  static LogOptions Read(config::Reader reader);
};

// 停滞看门狗：就绪计数（ReadyWhen / ReadyWhenFramesReach）超过时限不增长时排空并以失败退出，
// 交给容器的重启策略恢复。Docker 只在进程退出时重启，不会重启 unhealthy 的容器。
// 没有登记计数来源时主循环运行即算推进，看门狗不会触发。
struct WatchdogOptions {
  int stall_timeout_s = 0;  // 0 表示关闭

  static WatchdogOptions Read(config::Reader reader);
};

// 就绪（健康检查）的含义。require_input 为 false（默认）时就绪表示应用活着：主循环在运行且看门狗没有
// 触发，没有输入也算就绪，线阵、外触发等不一定产帧的源不会被当成故障；输入是否在供帧由看门狗把守，
// 并记录在就绪文件中（见 ReadinessHeartbeat）。为 true 时就绪还要求就绪计数在健康检查周期内增长过。
struct ReadinessOptions {
  bool require_input = false;

  static ReadinessOptions Read(config::Reader reader);
};

// 装配：应用在 Build 中向这里添加图、组件与发布器。
class Assembly {
 public:
  explicit Assembly(Paths paths) : paths_(std::move(paths)) {}

  const Paths& paths() const { return paths_; }
  gst::Graph& graph() { return graph_; }
  gst::Options& options() { return options_; }

  // 组件由装配持有，登记顺序即 OnCreated 的调用顺序。
  template <typename C, typename... Args>
  C& Add(Args&&... args) {
    auto component = std::make_unique<C>(std::forward<Args>(args)...);
    C& reference = *component;
    components_.push_back(std::move(component));
    return reference;
  }

  // 发布器在管线启动前 Start，管线结束后 Stop（录像的最后一个 finished 事件因此能发出）。
  publish::Publisher& AddPublisher(std::unique_ptr<publish::Publisher> publisher);

  // 就绪心跳：progress 返回单调递增的处理计数，计数增长时更新就绪文件。
  // DeepStream 管线应使用 MetaTap::frames()：mux 之后一个 buffer 是一个批次，帧数随源而变，
  // 多路时某一路断流批次仍会推送；按真实帧计数才反映输入在供帧。都不调用时，主循环运行即视为就绪
  // （适用于车辆停止时不出帧这类按需产帧的源，配合 options().startup_timeout = 0）。
  void ReadyWhen(std::function<std::uint64_t()> progress) { ready_progress_ = std::move(progress); }
  // 不经过 nvstreammux 的普通 GStreamer 管线：按经过该元件的 buffer 计数。
  void ReadyWhenFramesReach(const std::string& element) { ready_element_ = element; }

  gst::Requirements Requirements() const;

  const std::vector<std::unique_ptr<gst::Component>>& components() const { return components_; }
  const std::vector<std::unique_ptr<publish::Publisher>>& publishers() const { return publishers_; }
  const std::string& ready_element() const { return ready_element_; }
  const std::function<std::uint64_t()>& ready_progress() const { return ready_progress_; }

 private:
  Paths paths_;
  gst::Graph graph_;
  gst::Options options_;
  std::vector<std::unique_ptr<gst::Component>> components_;
  std::vector<std::unique_ptr<publish::Publisher>> publishers_;
  std::string ready_element_;
  std::function<std::uint64_t()> ready_progress_;
};

struct Definition {
  Info info;
  std::function<void(config::Reader&)> load;
  std::function<void(Assembly&)> build;
};

int Run(int argc, char** argv, const Definition& definition);

template <typename App>
int Main(int argc, char** argv) {
  auto config = std::make_shared<typename App::Config>();
  Definition definition;
  definition.info = App::Describe();
  definition.load = [config](config::Reader& root) { *config = App::Load(root); };
  definition.build = [config](Assembly& assembly) { App::Build(*config, assembly); };
  return Run(argc, argv, definition);
}

// ---- 以下供框架测试与自定义入口使用 ----

enum class Mode { kRun, kCheckConfig, kDumpPipeline, kSelfTest, kVersion, kHelp };

struct CommandLine {
  Mode mode = Mode::kRun;
  std::string config_path;  // 空表示 <config>/<name>.yaml
  bool plugins = false;     // --check-config --plugins：同时按实际插件校验属性
};

// 参数错误抛 std::invalid_argument。
CommandLine ParseCommandLine(int argc, char** argv);
std::string Usage(const Info& info, const Paths& paths);
std::string BuildInfo(const Info& info, const Paths& paths);

// 校验日志选项，不创建任何文件；失败抛 std::invalid_argument。
void ValidateLogging(const LogOptions& options);
void ConfigureLogging(const LogOptions& options, const std::string& name);

// 公共自检：30 帧软件管线，不需要 GPU、相机或配置。返回退出码。
int RunSelfTest();

}  // namespace dsf::app

#endif  // DSF_APP_APP_H
