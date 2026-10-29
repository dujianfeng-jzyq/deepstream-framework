// 单条管线的生命周期。
//
// Run() 在调用线程上阻塞运行主循环，实例只能运行一次。默认使用全局默认 GMainContext：
// 不少插件用 g_timeout_add 在默认 context 上注册内部定时器（例如 nvurisrcbin 的 RTSP 断流重连
// 看门狗），只跑私有 context 时这些定时器永远不会触发。多条管线在不同线程并行运行时，
// 次要管线设置 Options::private_context，代价是上述插件定时器需要另有线程驱动默认 context。
// 控制线程（Run 所在线程）处理总线消息、组件回调、定时任务和停止；媒体线程只应复制数据并入队。
//
// 停止：首次请求进入排空（先通知组件 OnStopping，再注入 EOS：由图构建时注入到每个源之后的元件，
// 否则发给整条管线），排空超时或 kImmediate 强制结束；
// 排空期间再收到 SIGINT/SIGTERM 升级为强制结束。
//
// 运行期错误：默认立即结束。Options::drain_on_error 开启时，PLAYING 或排空中的总线错误先排空再结束，
// 结果仍是 run-error（带第一条错误），让其余分支收尾（例如 rtspclientsink 断线时录像器封装当前分段）。
// 只在出错元件不向上游返回流错误时排空才能自然完成（rtspclientsink 断线属于这种）；否则共用 tee 的
// 其他分支也会停住，只能靠组件自己的兜底（录像器的 eos_fallback）或在 stop_timeout 后强制结束。
//
// 启动：startup_timeout 为正数时必须在时限内真正进入 PLAYING。异步切换状态的 sink 可能等待首帧，
// 按需产帧的源可设为 0 禁用启动超时。状态切换返回失败或总线报告错误时仍以 start-error / run-error
// 结束，等待期间的停止请求照常排空；OnPlaying 仍只在真正进入 PLAYING 后调用。
//
// 排空完成的判据：管线 EOS 消息；或者 EOS 已到达每个 sink 的输入、且所有组件 Drained()。
// 后者让网络 sink（例如 rtspclientsink 等待服务端拆除会话）不会拖住停止，
// 而需要收尾的组件（录像器等待最后一个分段封装完成）仍然被等待。
// 只观察管线直接子元件中的 sink：sink 本身是 bin（splitmuxsink、rtspclientsink、应用自建的
// mux+filesink bin）时，EOS 到达的是 bin 的输入，不代表内部已经收尾，而内部元件可能运行后才创建，
// 无法事先观察。这类 sink 若需要完整收尾（例如封装 MP4），必须由组件的 Drained() 报告，
// 否则排空会在封装完成前结束（参见 record::SegmentRecorder）。

#ifndef DSF_GST_PIPELINE_H
#define DSF_GST_PIPELINE_H

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <set>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gst/gst.h>

#include "dsf/gst/graph.h"
#include "dsf/gst/probe.h"

namespace dsf::gst {

class Component;

enum class StopMode { kDrain, kImmediate };

enum class FinishReason {
  kEos,         // 管线自然结束
  kStopped,     // 请求停止后正常排空
  kForced,      // 排空超时或立即停止
  kBuildError,  // 建图失败
  kStartError,  // 无法进入 PLAYING
  kRunError,    // 运行期错误或组件回调失败
  kStartupTimeout,  // 启动超时
};

const char* ToString(FinishReason reason);

struct Result {
  FinishReason reason = FinishReason::kBuildError;
  std::string error;
  // 只有自然结束与请求停止后正常排空算成功。
  bool ok() const { return reason == FinishReason::kEos || reason == FinishReason::kStopped; }
};

struct Options {
  // 0 表示不限等待（见文件头说明）；负数视为启动失败。
  std::chrono::milliseconds startup_timeout{30000};
  // 必须小于容器的停止宽限期，否则收尾前就会被 SIGKILL。
  std::chrono::milliseconds stop_timeout{5000};
  // 在 Run 期间处理 SIGINT/SIGTERM。多条管线并行时都可以开启：每条都会收到停止请求。
  bool handle_signals = true;
  // 使用私有 GMainContext（见文件头说明）。同一时刻只能有一条管线使用默认 context。
  bool private_context = false;
  // 运行期错误先排空再以 run-error 结束（见文件头说明）；启动阶段的错误仍立即结束。
  bool drain_on_error = false;
};

// 周期任务句柄：析构或 Cancel() 后不再触发；可以晚于 Pipeline 析构。
class Timer {
 public:
  Timer() = default;
  explicit Timer(GSource* source) : source_(source) {}
  ~Timer() { Cancel(); }
  Timer(const Timer&) = delete;
  Timer& operator=(const Timer&) = delete;
  Timer(Timer&& other) noexcept : source_(other.source_) { other.source_ = nullptr; }
  Timer& operator=(Timer&& other) noexcept;
  void Cancel() noexcept;

 private:
  GSource* source_ = nullptr;
};

class Pipeline {
 public:
  using Builder = std::function<GstElement*()>;  // 返回 GstPipeline 的唯一引用

  Pipeline(std::string name, const Graph& graph);
  Pipeline(std::string name, Builder builder);
  ~Pipeline();

  Pipeline(const Pipeline&) = delete;
  Pipeline& operator=(const Pipeline&) = delete;

  // gst-launch 描述的建图器：图模型之外的逃生口（例如移植现有的命令行管线）。
  static Builder Launch(std::string description);

  // 登记组件；必须在 Run 之前，组件寿命须覆盖 Run。
  void Add(Component& component);

  Result Run(const Options& options = {});

  // 任意线程调用；动作排队到控制线程执行。
  void RequestStop(StopMode mode = StopMode::kDrain);
  // 在控制线程上执行任务（排队，绝不内联）；任务异常只记录。
  // Run 之前提交的任务暂存，Run 选定 context 后按提交顺序执行。
  // 管线结束后提交的任务与尚未执行的任务都会被丢弃（默认 context 跨管线共享，不能留下悬空任务）。
  void Post(std::function<void()> task);
  // 在控制线程上周期执行；任务异常只记录。管线结束时自动停止，Timer 可以提前取消。
  // 只在控制线程调用（例如组件回调或 Post 的任务中）。
  Timer Every(std::chrono::milliseconds interval, std::function<void()> task);

  const std::string& name() const { return name_; }
  // 借用指针，只在控制线程、Run 期间有效。
  GstElement* bin() const { return pipeline_; }
  GstElement* Element(const std::string& name) const;

 private:
  enum class State { kReady, kStarting, kPlaying, kDraining, kFinished };

  void Start();  // 建图到请求 PLAYING；任何一步失败都经 Finish 记录结果并返回
  void Finish(Result result);
  void HandleMessage(GstMessage* message);
  void NotifyPlaying();
  void StopOnControlThread(StopMode mode);
  void WatchSinks();
  void CheckDrained();
  std::string PendingSinks() const;
  static GstPadProbeReturn OnSinkEvent(GstPad* pad, GstPadProbeInfo* info, gpointer self);
  // 依次调用组件，捕获并记录异常；一个组件失败不影响其他组件。返回第一个错误，成功为空。
  std::string ForEach(const char* stage, const std::function<void(Component&)>& call,
                      bool reverse = false);
  void Cleanup();
  void ClosePosted();     // 丢弃并拒绝之后的 Post 任务
  void DestroySources();  // 销毁并释放 sources_
  GSource* Attach(GSource* source, GSourceFunc callback);
  void Destroy(GSource** source);

  void AttachPostedLocked(std::function<void()> task);  // 调用方持有 posted_mutex_
  static gboolean RunPosted(gpointer data);
  static gboolean OnBus(GstBus* bus, GstMessage* message, gpointer self);
  static gboolean OnSignal(gpointer self);
  static gboolean OnStartupTimeout(gpointer self);
  static gboolean OnStopTimeout(gpointer self);

  std::string name_;
  Builder builder_;
  std::vector<std::string> drain_points_;  // 为空时向整条管线发送 EOS
  std::vector<Component*> components_;
  GMainContext* context_ = nullptr;
  GMainLoop* loop_ = nullptr;
  GstElement* pipeline_ = nullptr;
  std::vector<GSource*> sources_;  // 内部源（总线、超时、信号），清理时销毁
  std::mutex posted_mutex_;
  std::set<GSource*> posted_;      // Post 的一次性任务（持有引用），执行或清理时移除
  std::vector<std::function<void()>> pending_;  // Run 选定 context 之前 Post 的任务
  bool attached_ = false;          // Run 已选定 context，Post 直接挂上
  bool closed_ = false;            // Cleanup 之后不再接受 Post
  GSource* startup_timer_ = nullptr;
  std::atomic<State> state_{State::kReady};
  std::optional<Result> result_;
  std::optional<Result> run_error_;  // 触发排空的第一条运行期错误（drain_on_error）
  Options options_;
  struct SinkWatch {
    std::atomic<bool> eos{false};
    PadProbe probe;
  };
  std::vector<std::unique_ptr<SinkWatch>> sink_watches_;
  std::atomic<std::size_t> sinks_at_eos_{0};
};

}  // namespace dsf::gst

#endif  // DSF_GST_PIPELINE_H
