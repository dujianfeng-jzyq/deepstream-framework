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

struct Task {
  std::function<void()> run;
  bool repeat;
};

gboolean RunTask(gpointer data) {
  auto* task = static_cast<Task*>(data);
  try {
    task->run();
  } catch (const std::exception& error) {
    spdlog::error("pipeline task failed: {}", error.what());
  } catch (...) {
    spdlog::error("pipeline task failed with an unknown exception");
  }
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
    case FinishReason::kTimeout: return "startup-timeout";
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
  // 默认 context；Run 时选择私有 context 会替换它，此前通过 Post/Every 登记的任务随之丢弃。
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
  {
    std::lock_guard<std::mutex> lock(posted_mutex_);
    closed_ = true;
    for (GSource* source : posted_) {
      g_source_destroy(source);
      g_source_unref(source);
    }
    posted_.clear();
  }
  if (pipeline_ != nullptr) {  // 仅在 Run 异常中断时出现
    gst_element_set_state(pipeline_, GST_STATE_NULL);
    gst_object_unref(pipeline_);
  }
  for (GSource* source : sources_) {
    g_source_destroy(source);
    g_source_unref(source);
  }
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
  try {
    task->run();
  } catch (const std::exception& error) {
    spdlog::error("pipeline task failed: {}", error.what());
  } catch (...) {
    spdlog::error("pipeline task failed with an unknown exception");
  }
  g_source_unref(source);  // posted_ 持有的引用
  return G_SOURCE_REMOVE;
}

void Pipeline::Post(std::function<void()> task) {
  std::lock_guard<std::mutex> lock(posted_mutex_);
  if (closed_) return;
  GSource* source = g_idle_source_new();
  g_source_set_callback(source, &Pipeline::RunPosted, new PostedTask{this, source, std::move(task)},
                        DeletePosted);
  posted_.insert(source);  // 引用转给 posted_
  g_source_attach(source, context_);
}

Timer Pipeline::Every(std::chrono::milliseconds interval, std::function<void()> task) {
  GSource* source = g_timeout_source_new(static_cast<guint>(interval.count()));
  AttachTask(context_, source, std::move(task), true);
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
  if (!result_) result_ = std::move(result);
  g_main_loop_quit(loop_);
}

Result Pipeline::Run(const Options& options) {
  State expected = State::kReady;
  if (!state_.compare_exchange_strong(expected, State::kStarting)) {
    return {false, FinishReason::kStartError, "a Pipeline can only run once"};
  }
  options_ = options;
  EnsureInitialized();
  if (options_.private_context) {
    g_main_loop_unref(loop_);
    g_main_context_unref(context_);
    context_ = g_main_context_new();
    loop_ = g_main_loop_new(context_, FALSE);
  } else if (!g_main_context_acquire(context_)) {
    return {false, FinishReason::kStartError,
            "the default main context is owned by another thread; use Options::private_context"};
  }
  g_main_context_push_thread_default(context_);

  try {
    try {
      pipeline_ = builder_();
    } catch (const std::exception& error) {
      Finish({false, FinishReason::kBuildError, error.what()});
    }
    if (!result_ && pipeline_ == nullptr) {
      Finish({false, FinishReason::kBuildError, "the pipeline builder returned nothing"});
    }
    if (!result_ && !GST_IS_PIPELINE(pipeline_)) {
      gst_object_unref(pipeline_);
      pipeline_ = nullptr;
      Finish({false, FinishReason::kBuildError, "the pipeline builder must return a GstPipeline"});
    }
    if (!result_) {
      const std::string error =
          ForEach("OnCreated", [this](Component& c) { c.OnCreated(*this, pipeline_); });
      if (!error.empty()) Finish({false, FinishReason::kRunError, error});
    }
    if (!result_) {
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
          Finish({false, FinishReason::kStartError, "the pipeline refused to start"});
          break;
        case GST_STATE_CHANGE_SUCCESS:
          NotifyPlaying();
          break;
        default:  // ASYNC 或直播源的 NO_PREROLL：等待 STATE_CHANGED 消息
          startup_timer_ = Attach(
              g_timeout_source_new(static_cast<guint>(options_.startup_timeout.count())),
              &Pipeline::OnStartupTimeout);
          break;
      }
    }
    if (!result_) g_main_loop_run(loop_);
    if (!result_) Finish({false, FinishReason::kRunError, "main loop exited without a result"});
  } catch (const std::exception& error) {
    Finish({false, FinishReason::kRunError, error.what()});
  }
  Cleanup();
  g_main_context_pop_thread_default(context_);
  if (!options_.private_context) g_main_context_release(context_);
  return *result_;
}

void Pipeline::WatchSinks() {
  GstIterator* sinks = gst_bin_iterate_sinks(GST_BIN(pipeline_));
  GValue sink = G_VALUE_INIT;
  while (gst_iterator_next(sinks, &sink) == GST_ITERATOR_OK) {
    GstIterator* pads = gst_element_iterate_sink_pads(GST_ELEMENT(g_value_get_object(&sink)));
    GValue pad = G_VALUE_INIT;
    while (gst_iterator_next(pads, &pad) == GST_ITERATOR_OK) {
      auto watch = std::make_unique<SinkWatch>();
      watch->pad = GST_PAD(g_value_dup_object(&pad));
      watch->probe = gst_pad_add_probe(watch->pad, GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM,
                                       &Pipeline::OnSinkEvent, this, nullptr);
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
    if (watch->pad == pad && !watch->eos.exchange(true)) ++pipeline->sinks_at_eos_;
  }
  pipeline->Post([pipeline] { pipeline->CheckDrained(); });
  return GST_PAD_PROBE_OK;
}

std::string Pipeline::PendingSinks() const {
  std::string text;
  for (const auto& watch : sink_watches_) {
    if (watch->eos.load()) continue;
    GstElement* parent = gst_pad_get_parent_element(watch->pad);
    text += std::string(text.empty() ? "" : ", ") + (parent ? GST_ELEMENT_NAME(parent) : "?") + "." +
            GST_PAD_NAME(watch->pad);
    if (parent != nullptr) gst_object_unref(parent);
  }
  return text;
}

void Pipeline::CheckDrained() {
  if (state_ != State::kDraining || result_ || sinks_at_eos_.load() < sink_watches_.size()) return;
  for (Component* component : components_) {
    if (!component->Drained()) return;
  }
  Finish({true, FinishReason::kStopped, {}});
}

void Pipeline::Cleanup() {
  {
    std::lock_guard<std::mutex> lock(posted_mutex_);
    closed_ = true;
    for (GSource* source : posted_) {
      g_source_destroy(source);
      g_source_unref(source);
    }
    posted_.clear();
  }
  for (auto& watch : sink_watches_) {
    gst_pad_remove_probe(watch->pad, watch->probe);
    gst_object_unref(watch->pad);
  }
  sink_watches_.clear();
  for (GSource* source : sources_) {
    g_source_destroy(source);
    g_source_unref(source);
  }
  sources_.clear();
  startup_timer_ = nullptr;
  if (pipeline_ != nullptr) {
    gst_element_set_state(pipeline_, GST_STATE_NULL);
  }
  // 流线程已停止：组件可以安全拆除探针与信号。
  const std::string error = ForEach("OnReleased", [](Component& c) { c.OnReleased(); }, true);
  if (!error.empty() && result_->success) {
    result_ = Result{false, FinishReason::kRunError, error};
  }
  if (pipeline_ != nullptr) {
    gst_object_unref(pipeline_);
    pipeline_ = nullptr;
  }
  state_ = State::kFinished;
  if (result_->success) {
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
  if (!error.empty()) Finish({false, FinishReason::kRunError, error});
  CheckDrained();  // 组件可能刚在这条消息中完成收尾（例如最后一个分段关闭）

  switch (GST_MESSAGE_TYPE(message)) {
    case GST_MESSAGE_STATE_CHANGED: {
      if (GST_MESSAGE_SRC(message) != GST_OBJECT_CAST(pipeline_)) return;
      GstState old_state, new_state, pending;
      gst_message_parse_state_changed(message, &old_state, &new_state, &pending);
      if (new_state == GST_STATE_PLAYING) NotifyPlaying();
      return;
    }
    case GST_MESSAGE_ERROR:
      Finish({false, FinishReason::kRunError, ErrorText(message, false)});
      return;
    case GST_MESSAGE_WARNING:
      spdlog::warn("pipeline {}: {}", name_, ErrorText(message, true));
      return;
    case GST_MESSAGE_EOS:
      Finish({true, state_ == State::kDraining ? FinishReason::kStopped : FinishReason::kEos, {}});
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
  if (!error.empty()) Finish({false, FinishReason::kRunError, error});
}

void Pipeline::StopOnControlThread(StopMode mode) {
  if (pipeline_ == nullptr || result_) return;
  if (state_ != State::kDraining) {
    state_ = State::kDraining;
    Destroy(&startup_timer_);
    const std::string error = ForEach("OnStopping", [](Component& c) { c.OnStopping(); });
    if (!error.empty()) {
      Finish({false, FinishReason::kRunError, error});
      return;
    }
  } else if (mode == StopMode::kDrain) {
    return;  // 重复的排空请求不重置截止时间
  }
  if (mode == StopMode::kImmediate) {
    Finish({false, FinishReason::kForced, "stopped without draining"});
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
    GstIterator* pads = gst_element_iterate_sink_pads(element);
    GValue pad = G_VALUE_INIT;
    while (gst_iterator_next(pads, &pad) == GST_ITERATOR_OK) {
      gst_pad_send_event(GST_PAD(g_value_get_object(&pad)), gst_event_new_eos());
      g_value_reset(&pad);
    }
    g_value_unset(&pad);
    gst_iterator_free(pads);
    gst_object_unref(element);
  }
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
  pipeline->Finish({false, FinishReason::kTimeout, "the pipeline did not reach PLAYING in time"});
  return G_SOURCE_REMOVE;
}

gboolean Pipeline::OnStopTimeout(gpointer self) {
  auto* pipeline = static_cast<Pipeline*>(self);
  std::string waiting = pipeline->PendingSinks();
  for (Component* component : pipeline->components_) {
    if (!component->Drained()) waiting += (waiting.empty() ? "" : ", ") + std::string(component->Name());
  }
  pipeline->Finish({false, FinishReason::kForced,
                    "draining did not finish before the stop timeout; still waiting for: " +
                        (waiting.empty() ? std::string("pipeline EOS") : waiting)});
  return G_SOURCE_REMOVE;
}

}  // namespace dsf::gst
