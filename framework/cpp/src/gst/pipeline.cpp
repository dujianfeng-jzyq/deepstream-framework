// Pipeline 的实现。所有内部源都挂在私有 context 上，清理时按源对象销毁，
// 不使用只查默认 context 的 g_source_remove。

#include "dsf/gst/pipeline.h"

#include <glib-unix.h>
#include <spdlog/spdlog.h>

#include <csignal>
#include <stdexcept>
#include <utility>

#include "dsf/gst/component.h"

namespace dsf::gst {
namespace {

// 控制线程上的任务：异常只记录，不能穿过 GLib 的 C 回调。
void InvokeLogged(const std::function<void()>& task) {
  try {
    task();
  } catch (const std::exception& error) {
    spdlog::error("pipeline task failed: {}", error.what());
  } catch (...) {
    spdlog::error("pipeline task failed with an unknown exception");
  }
}

struct Task {
  std::function<void()> run;
  bool repeat;
};

gboolean RunTask(gpointer data) {
  auto* task = static_cast<Task*>(data);
  InvokeLogged(task->run);
  return task->repeat ? G_SOURCE_CONTINUE : G_SOURCE_REMOVE;
}

void DeleteTask(gpointer data) { delete static_cast<Task*>(data); }

GSource* AttachTask(GMainContext* context, GSource* source, std::function<void()> run, bool repeat) {
  g_source_set_callback(source, RunTask, new Task{std::move(run), repeat}, DeleteTask);
  g_source_attach(source, context);
  return source;
}

std::string ErrorText(GstMessage* message, bool warning) {
  GError* error = nullptr;
  gchar* debug = nullptr;
  if (warning) {
    gst_message_parse_warning(message, &error, &debug);
  } else {
    gst_message_parse_error(message, &error, &debug);
  }
  std::string text = GST_OBJECT_NAME(GST_MESSAGE_SRC(message));
  text += ": ";
  text += error != nullptr ? error->message : "unknown error";
  if (debug != nullptr) text += std::string(" (") + debug + ")";
  if (error != nullptr) g_error_free(error);
  g_free(debug);
  return text;
}

}  // namespace

const char* ToString(FinishReason reason) {
  switch (reason) {
    case FinishReason::kEos: return "eos";
    case FinishReason::kStopped: return "stopped";
    case FinishReason::kForced: return "forced";
    case FinishReason::kBuildError: return "build-error";
    case FinishReason::kStartError: return "start-error";
    case FinishReason::kRunError: return "run-error";
    case FinishReason::kStartupTimeout: return "startup-timeout";
  }
  return "unknown";
}

Timer& Timer::operator=(Timer&& other) noexcept {
  if (this != &other) {
    Cancel();
    source_ = other.source_;
    other.source_ = nullptr;
  }
  return *this;
}

void Timer::Cancel() noexcept {
  if (source_ != nullptr) {
    g_source_destroy(source_);  // 源已随 context 销毁时也是安全的
    g_source_unref(source_);
    source_ = nullptr;
  }
}

Pipeline::Pipeline(std::string name, const Graph& graph)
    : Pipeline(name, [graph, name] { return graph.Instantiate(name); }) {
  drain_points_ = graph.DrainPoints();
}

Pipeline::Pipeline(std::string name, Builder builder)
    : name_(std::move(name)), builder_(std::move(builder)) {
  // 默认 context；Run 时选择私有 context 会替换它。Run 之前 Post 的任务暂存，Run 选定 context 后再挂上。
  context_ = g_main_context_ref(g_main_context_default());
  loop_ = g_main_loop_new(context_, FALSE);
}

Pipeline::Builder Pipeline::Launch(std::string description) {
  return [description = std::move(description)]() -> GstElement* {
    EnsureInitialized();
    GError* error = nullptr;
    GstElement* element = gst_parse_launch(description.c_str(), &error);
    if (error != nullptr) {
      const std::string message = error->message;
      g_error_free(error);
      if (element != nullptr) gst_object_unref(element);
      throw std::runtime_error(message);
    }
    return element;
  };
}

Pipeline::~Pipeline() {
  ClosePosted();
  if (pipeline_ != nullptr) {  // 仅在 Run 异常中断时出现
    gst_element_set_state(pipeline_, GST_STATE_NULL);
    gst_object_unref(pipeline_);
  }
  DestroySources();
  g_main_loop_unref(loop_);
  g_main_context_unref(context_);
}

void Pipeline::Add(Component& component) {
  if (state_ != State::kReady) throw std::logic_error("components must be added before Run");
  components_.push_back(&component);
}

GstElement* Pipeline::Element(const std::string& name) const {
  if (pipeline_ == nullptr) return nullptr;
  GstElement* element = gst_bin_get_by_name(GST_BIN(pipeline_), name.c_str());
  if (element != nullptr) gst_object_unref(element);  // bin 仍持有引用，这里返回借用指针
  return element;
}

namespace {

struct PostedTask {
  Pipeline* owner;
  GSource* source;
  std::function<void()> run;
};

void DeletePosted(gpointer data) { delete static_cast<PostedTask*>(data); }

}  // namespace

gboolean Pipeline::RunPosted(gpointer data) {
  auto* task = static_cast<PostedTask*>(data);
  Pipeline* owner = task->owner;
  {
    std::lock_guard<std::mutex> lock(owner->posted_mutex_);
    if (owner->posted_.erase(task->source) == 0) return G_SOURCE_REMOVE;  // 已被清理
  }
  GSource* source = task->source;
  InvokeLogged(task->run);
  g_source_unref(source);  // posted_ 持有的引用
  return G_SOURCE_REMOVE;
}

void Pipeline::Post(std::function<void()> task) {
  std::lock_guard<std::mutex> lock(posted_mutex_);
  if (closed_) return;
  if (!attached_) {
    pending_.push_back(std::move(task));
    return;
  }
  AttachPostedLocked(std::move(task));
}

void Pipeline::AttachPostedLocked(std::function<void()> task) {
  GSource* source = g_idle_source_new();
  g_source_set_callback(source, &Pipeline::RunPosted, new PostedTask{this, source, std::move(task)},
                        DeletePosted);
  posted_.insert(source);  // 引用转给 posted_
  g_source_attach(source, context_);
}

Timer Pipeline::Every(std::chrono::milliseconds interval, std::function<void()> task) {
  return Schedule(interval, std::move(task), true);
}

Timer Pipeline::After(std::chrono::milliseconds delay, std::function<void()> task) {
  return Schedule(delay, std::move(task), false);
}

Timer Pipeline::Schedule(std::chrono::milliseconds delay, std::function<void()> task, bool repeat) {
  GSource* source = g_timeout_source_new(static_cast<guint>(delay.count()));
  AttachTask(context_, source, std::move(task), repeat);
  // 管线结束时一并停止：默认 context 跨管线共享，遗留的定时器会在下一次运行中触发。
  sources_.push_back(g_source_ref(source));
  return Timer(source);
}

void Pipeline::RequestStop(StopMode mode) {
  Post([this, mode] { StopOnControlThread(mode); });
}

GSource* Pipeline::Attach(GSource* source, GSourceFunc callback) {
  g_source_set_callback(source, callback, this, nullptr);
  g_source_attach(source, context_);
  sources_.push_back(source);
  return source;
}

void Pipeline::Destroy(GSource** source) {
  if (*source == nullptr) return;
  g_source_destroy(*source);
  *source = nullptr;  // 引用仍在 sources_ 中，Cleanup 统一释放
}

std::string Pipeline::ForEach(const char* stage, const std::function<void(Component&)>& call,
                              bool reverse) {
  std::string first;
  const auto invoke = [&](Component* component) {
    try {
      call(*component);
    } catch (const std::exception& error) {
      spdlog::error("pipeline {}: component {} failed in {}: {}", name_, component->Name(), stage,
                    error.what());
      if (first.empty()) first = std::string(component->Name()) + ": " + error.what();
    }
  };
  if (reverse) {
    for (auto it = components_.rbegin(); it != components_.rend(); ++it) invoke(*it);
  } else {
    for (Component* component : components_) invoke(component);
  }
  return first;
}

void Pipeline::Finish(Result result) {
  if (!result_) {
    // 由运行期错误触发的排空：无论排空是否完成都以该错误结束，排空未完成的原因附在后面。
    if (run_error_) {
      Result failure = *run_error_;
      if (!result.ok()) {
        failure.error += " (then " + std::string(ToString(result.reason)) + ": " + result.error + ")";
      }
      result = std::move(failure);
    }
    result_ = std::move(result);
  }
  g_main_loop_quit(loop_);
}

Result Pipeline::Run(const Options& options) {
  State expected = State::kReady;
  if (!state_.compare_exchange_strong(expected, State::kStarting)) {
    return {FinishReason::kStartError, "a Pipeline can only run once"};
  }
  options_ = options;
  EnsureInitialized();
  if (options_.private_context) {
    g_main_loop_unref(loop_);
    g_main_context_unref(context_);
    context_ = g_main_context_new();
    loop_ = g_main_loop_new(context_, FALSE);
  } else if (!g_main_context_acquire(context_)) {
    return {FinishReason::kStartError,
            "the default main context is owned by another thread; use Options::private_context"};
  }
  g_main_context_push_thread_default(context_);
  {
    // context_ 已确定：此后 Post 直接挂到它上面。在锁内切换，Post 不会读到被替换的 context。
    std::lock_guard<std::mutex> lock(posted_mutex_);
    attached_ = true;
    for (auto& task : pending_) AttachPostedLocked(std::move(task));
    pending_.clear();
  }

  try {
    Start();
    // Finish 在主循环运行之前调用时 quit 不生效：已有结果就不能再进入主循环。
    if (!result_) g_main_loop_run(loop_);
    if (!result_) Finish({FinishReason::kRunError, "main loop exited without a result"});
  } catch (const std::exception& error) {
    Finish({FinishReason::kRunError, error.what()});
  }
  Cleanup();
  g_main_context_pop_thread_default(context_);
  if (!options_.private_context) g_main_context_release(context_);
  return *result_;
}

void Pipeline::Start() {
  if (options_.startup_timeout.count() < 0) {
    Finish({FinishReason::kStartError, "startup_timeout must not be negative"});
    return;
  }
  try {
    pipeline_ = builder_();
  } catch (const std::exception& error) {
    Finish({FinishReason::kBuildError, error.what()});
    return;
  }
  if (pipeline_ == nullptr) {
    Finish({FinishReason::kBuildError, "the pipeline builder returned nothing"});
    return;
  }
  if (!GST_IS_PIPELINE(pipeline_)) {
    gst_object_unref(pipeline_);
    pipeline_ = nullptr;
    Finish({FinishReason::kBuildError, "the pipeline builder must return a GstPipeline"});
    return;
  }
  if (const std::string error = ForEach("OnCreated", [this](Component& c) { c.OnCreated(*this, pipeline_); });
      !error.empty()) {
    Finish({FinishReason::kRunError, error});
    return;
  }

  WatchSinks();
  GstBus* bus = gst_element_get_bus(pipeline_);
  Attach(gst_bus_create_watch(bus), G_SOURCE_FUNC(&Pipeline::OnBus));
  gst_object_unref(bus);
  if (options_.handle_signals) {
    Attach(g_unix_signal_source_new(SIGINT), &Pipeline::OnSignal);
    Attach(g_unix_signal_source_new(SIGTERM), &Pipeline::OnSignal);
  }
  spdlog::info("pipeline {} starting", name_);
  switch (gst_element_set_state(pipeline_, GST_STATE_PLAYING)) {
    case GST_STATE_CHANGE_FAILURE:
      Finish({FinishReason::kStartError, "the pipeline refused to start"});
      break;
    case GST_STATE_CHANGE_SUCCESS:
      NotifyPlaying();
      break;
    default:  // ASYNC 或直播源的 NO_PREROLL：等待 STATE_CHANGED 消息
      // 时限为 0 时禁用检测：按需产帧的管线可能等待首帧才能完成状态切换。
      if (options_.startup_timeout.count() > 0) {
        startup_timer_ = Attach(g_timeout_source_new(static_cast<guint>(options_.startup_timeout.count())),
                                &Pipeline::OnStartupTimeout);
      }
      break;
  }
}

void Pipeline::WatchSinks() {
  GstIterator* sinks = gst_bin_iterate_sinks(GST_BIN(pipeline_));
  GValue sink = G_VALUE_INIT;
  while (gst_iterator_next(sinks, &sink) == GST_ITERATOR_OK) {
    GstIterator* pads = gst_element_iterate_sink_pads(GST_ELEMENT(g_value_get_object(&sink)));
    GValue pad = G_VALUE_INIT;
    while (gst_iterator_next(pads, &pad) == GST_ITERATOR_OK) {
      auto watch = std::make_unique<SinkWatch>();
      watch->probe.Attach(GST_PAD(g_value_get_object(&pad)), GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM,
                          &Pipeline::OnSinkEvent, this);
      sink_watches_.push_back(std::move(watch));
      g_value_reset(&pad);
    }
    g_value_unset(&pad);
    gst_iterator_free(pads);
    g_value_reset(&sink);
  }
  g_value_unset(&sink);
  gst_iterator_free(sinks);
}

GstPadProbeReturn Pipeline::OnSinkEvent(GstPad* pad, GstPadProbeInfo* info, gpointer self) {
  if (GST_EVENT_TYPE(GST_PAD_PROBE_INFO_EVENT(info)) != GST_EVENT_EOS) return GST_PAD_PROBE_OK;
  auto* pipeline = static_cast<Pipeline*>(self);
  for (auto& watch : pipeline->sink_watches_) {
    if (watch->probe.pad() == pad && !watch->eos.exchange(true)) ++pipeline->sinks_at_eos_;
  }
  pipeline->Post([pipeline] { pipeline->CheckDrained(); });
  return GST_PAD_PROBE_OK;
}

std::string Pipeline::PendingSinks() const {
  std::string text;
  for (const auto& watch : sink_watches_) {
    if (watch->eos.load()) continue;
    GstPad* pad = watch->probe.pad();
    GstElement* parent = gst_pad_get_parent_element(pad);
    text += std::string(text.empty() ? "" : ", ") + (parent ? GST_ELEMENT_NAME(parent) : "?") + "." +
            GST_PAD_NAME(pad);
    if (parent != nullptr) gst_object_unref(parent);
  }
  return text;
}

void Pipeline::CheckDrained() {
  if (state_ != State::kDraining || result_ || sinks_at_eos_.load() < sink_watches_.size()) return;
  for (Component* component : components_) {
    if (!component->Drained()) return;
  }
  Finish({FinishReason::kStopped, {}});
}

void Pipeline::ClosePosted() {
  std::lock_guard<std::mutex> lock(posted_mutex_);
  closed_ = true;
  for (GSource* source : posted_) {
    g_source_destroy(source);
    g_source_unref(source);
  }
  posted_.clear();
  pending_.clear();
}

void Pipeline::DestroySources() {
  for (GSource* source : sources_) {
    g_source_destroy(source);
    g_source_unref(source);
  }
  sources_.clear();
}

void Pipeline::Cleanup() {
  ClosePosted();
  DestroySources();
  startup_timer_ = nullptr;
  if (pipeline_ != nullptr) {
    gst_element_set_state(pipeline_, GST_STATE_NULL);
  }
  // 流线程已停止，之后才能释放 sink 探针的数据：gst_pad_remove_probe 不等待正在执行的回调，
  // 强制停止时 EOS 可能仍在流向其余 sink。组件同理，在这里之后拆除探针与信号。
  sink_watches_.clear();
  const std::string error = ForEach("OnReleased", [](Component& c) { c.OnReleased(); }, true);
  if (!error.empty() && result_->ok()) {
    result_ = Result{FinishReason::kRunError, error};
  }
  if (pipeline_ != nullptr) {
    gst_object_unref(pipeline_);
    pipeline_ = nullptr;
  }
  state_ = State::kFinished;
  if (result_->ok()) {
    spdlog::info("pipeline {} finished: {}", name_, ToString(result_->reason));
  } else {
    spdlog::error("pipeline {} failed: {}: {}", name_, ToString(result_->reason), result_->error);
  }
}

gboolean Pipeline::OnBus(GstBus* /*bus*/, GstMessage* message, gpointer self) {
  static_cast<Pipeline*>(self)->HandleMessage(message);
  return G_SOURCE_CONTINUE;
}

void Pipeline::HandleMessage(GstMessage* message) {
  const std::string error = ForEach("OnMessage", [message](Component& c) { c.OnMessage(message); });
  if (!error.empty()) Finish({FinishReason::kRunError, error});
  CheckDrained();  // 组件可能刚在这条消息中完成收尾（例如最后一个分段关闭）

  switch (GST_MESSAGE_TYPE(message)) {
    case GST_MESSAGE_STATE_CHANGED: {
      if (GST_MESSAGE_SRC(message) != GST_OBJECT_CAST(pipeline_)) return;
      GstState old_state, new_state, pending;
      gst_message_parse_state_changed(message, &old_state, &new_state, &pending);
      if (new_state == GST_STATE_PLAYING) NotifyPlaying();
      return;
    }
    case GST_MESSAGE_ERROR: {
      const std::string text = ErrorText(message, false);
      if (!options_.drain_on_error || (state_ != State::kPlaying && state_ != State::kDraining)) {
        Finish({FinishReason::kRunError, text});
        return;
      }
      if (run_error_) {
        spdlog::error("pipeline {}: {}", name_, text);  // 出错元件常连续报告；已在排空，不重复处理
        return;
      }
      run_error_ = Result{FinishReason::kRunError, text};
      spdlog::error("pipeline {}: {}; draining before exit", name_, text);
      StopOnControlThread(StopMode::kDrain);
      return;
    }
    case GST_MESSAGE_WARNING:
      spdlog::warn("pipeline {}: {}", name_, ErrorText(message, true));
      return;
    case GST_MESSAGE_EOS:
      Finish({state_ == State::kDraining ? FinishReason::kStopped : FinishReason::kEos, {}});
      return;
    default:
      return;
  }
}

void Pipeline::NotifyPlaying() {
  if (state_ != State::kStarting || result_) return;
  Destroy(&startup_timer_);
  state_ = State::kPlaying;
  spdlog::info("pipeline {} is playing", name_);
  const std::string error = ForEach("OnPlaying", [this](Component& c) { c.OnPlaying(*this); });
  if (!error.empty()) Finish({FinishReason::kRunError, error});
}

void Pipeline::StopOnControlThread(StopMode mode) {
  if (pipeline_ == nullptr || result_) return;
  if (state_ != State::kDraining) {
    state_ = State::kDraining;
    Destroy(&startup_timer_);
    const std::string error = ForEach("OnStopping", [](Component& c) { c.OnStopping(); });
    if (!error.empty()) {
      Finish({FinishReason::kRunError, error});
      return;
    }
  } else if (mode == StopMode::kDrain) {
    return;  // 重复的排空请求不重置截止时间
  }
  if (mode == StopMode::kImmediate) {
    Finish({FinishReason::kForced, "stopped without draining"});
    return;
  }
  Attach(g_timeout_source_new(static_cast<guint>(options_.stop_timeout.count())),
         &Pipeline::OnStopTimeout);
  // 组件的收尾可能在总线消息之外完成，排空期间定期复查。
  Attach(g_timeout_source_new(100), +[](gpointer self) -> gboolean {
    static_cast<Pipeline*>(self)->CheckDrained();
    return G_SOURCE_CONTINUE;
  });
  if (drain_points_.empty()) {
    if (!gst_element_send_event(pipeline_, gst_event_new_eos())) {
      spdlog::warn("pipeline {}: EOS was not accepted by every source", name_);
    }
    return;
  }
  for (const std::string& name : drain_points_) {
    GstElement* element = gst_bin_get_by_name(GST_BIN(pipeline_), name.c_str());
    if (element == nullptr) continue;
    SendEosToSinkPads(element);
    gst_object_unref(element);
  }
}

void SendEosToSinkPads(GstElement* element) {
  GstIterator* pads = gst_element_iterate_sink_pads(element);
  GValue pad = G_VALUE_INIT;
  while (gst_iterator_next(pads, &pad) == GST_ITERATOR_OK) {
    gst_pad_send_event(GST_PAD(g_value_get_object(&pad)), gst_event_new_eos());
    g_value_reset(&pad);
  }
  g_value_unset(&pad);
  gst_iterator_free(pads);
}

gboolean Pipeline::OnSignal(gpointer self) {
  auto* pipeline = static_cast<Pipeline*>(self);
  if (pipeline->state_ == State::kDraining) {
    spdlog::warn("pipeline {}: second stop signal, forcing stop", pipeline->name_);
    pipeline->StopOnControlThread(StopMode::kImmediate);
  } else {
    spdlog::info("pipeline {}: stop signal received, draining", pipeline->name_);
    pipeline->StopOnControlThread(StopMode::kDrain);
  }
  return G_SOURCE_CONTINUE;
}

gboolean Pipeline::OnStartupTimeout(gpointer self) {
  auto* pipeline = static_cast<Pipeline*>(self);
  pipeline->startup_timer_ = nullptr;
  pipeline->Finish({FinishReason::kStartupTimeout, "the pipeline did not reach PLAYING in time"});
  return G_SOURCE_REMOVE;
}

gboolean Pipeline::OnStopTimeout(gpointer self) {
  auto* pipeline = static_cast<Pipeline*>(self);
  std::string waiting = pipeline->PendingSinks();
  for (Component* component : pipeline->components_) {
    if (!component->Drained()) waiting += (waiting.empty() ? "" : ", ") + std::string(component->Name());
  }
  pipeline->Finish({FinishReason::kForced,
                    "draining did not finish before the stop timeout; still waiting for: " +
                        (waiting.empty() ? std::string("pipeline EOS") : waiting)});
  return G_SOURCE_REMOVE;
}

}  // namespace dsf::gst
