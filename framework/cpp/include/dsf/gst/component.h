// 组件：参与管线生命周期的运行期对象（元数据探针、录像器、心跳、统计……）。
//
// 时序（全部在控制线程）：
//   OnCreated   建图成功、进入 PLAYING 之前；按登记顺序。取元件、挂探针、连信号。
//   OnPlaying   真正进入 PLAYING 后一次。异步切换状态的 sink 可能等待首帧，startup_timeout 为 0 时
//               可能长时间不调用；随主循环运行的定时任务应在 OnCreated 中注册。
//   OnMessage   每条总线消息；消息指针只在回调内有效。
//   OnStopping  首次停止请求、发送 EOS 之前一次（自然 EOS 与错误不保证经过这里）。
//   Drained     排空期间轮询，全部为 true 且 EOS 已到达每个 sink 的输入时排空完成。
//               EOS 到达 sink bin 的输入不代表 bin 内部已收尾：需要收尾的 sink bin 必须有组件
//               在这里报告，否则管线会在收尾前置 NULL（见 pipeline.h）。
//   OnReleased  管线已置 NULL、流线程已停止；按登记逆序。拆探针、断信号。
//               即使 OnCreated 没有被调用或抛了异常也会调用，实现须容忍。
// 构造函数不得产生副作用（建目录、删文件、连网络）：--check-config 也会构造组件。

#ifndef DSF_GST_COMPONENT_H
#define DSF_GST_COMPONENT_H

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gst/gst.h>

#include "dsf/gst/introspect.h"

namespace dsf::gst {

class Pipeline;

// 统计行：组件按 key=value 追加，由统计组件周期写日志。
class Stats {
 public:
  void Add(std::string key, std::string value) { entries_.emplace_back(std::move(key), std::move(value)); }
  void Add(std::string key, std::uint64_t value) { Add(std::move(key), std::to_string(value)); }
  std::string Text() const {
    std::string text;
    for (const auto& [key, value] : entries_) text += (text.empty() ? "" : " ") + key + "=" + value;
    return text;
  }

 private:
  std::vector<std::pair<std::string, std::string>> entries_;
};

class Component {
 public:
  virtual ~Component() = default;
  virtual std::string_view Name() const = 0;
  // 图之外的能力需求（例如信号）。
  virtual void Require(Requirements& /*requirements*/) const {}
  virtual void OnCreated(Pipeline& /*pipeline*/, GstElement* /*bin*/) {}
  virtual void OnPlaying(Pipeline& /*pipeline*/) {}
  virtual void OnMessage(GstMessage* /*message*/) {}
  virtual void OnStopping() {}
  virtual void OnReleased() {}
  // 排空期间：组件自己的收尾是否完成（例如最后一个录像分段已经封装并改名）。
  virtual bool Drained() const { return true; }
  // 运行结束后为 false 时，应用以失败退出（例如录像没有完整收尾）。
  virtual bool Healthy() const { return true; }
  // 周期统计可在 OnCreated 后、进入 PLAYING 前调用，须能报告尚未开始处理时的计数。
  virtual void Report(Stats& /*stats*/) const {}
};

}  // namespace dsf::gst

#endif  // DSF_GST_COMPONENT_H
