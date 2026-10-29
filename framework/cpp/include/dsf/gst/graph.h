// 管线图模型：元件、属性与连接的纯数据描述。
//
// 同一份图派生出三样东西，因此它们不会互相脱节：
//   Instantiate()   原生创建并连接元件（支持 request pad 与 sometimes pad 的延迟连接）
//   Requirements()  启动前的插件能力清单
//   Describe()      --dump-pipeline 的文本，敏感属性打码
// 块（dsf::ds）与应用都通过它组图；需要完全手写时仍可用 Pipeline 的原生工厂入口。

#ifndef DSF_GST_GRAPH_H
#define DSF_GST_GRAPH_H

#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

#include <gst/gst.h>

#include "dsf/config/config.h"
#include "dsf/gst/introspect.h"

namespace dsf::gst {

class Graph;

// 连接端点：元件加可选 pad 名。pad 名为空时由 GStreamer 自动选择；
// 也可以是静态 pad 名、request pad 名（"sink_0"）或模板名（"sink_%u"）。
struct Endpoint {
  int element = -1;
  std::string pad;
};

class Element {
 public:
  Element() = default;

  Element& Set(const std::string& property, const std::string& value);
  Element& Set(const std::string& property, const char* value);
  Element& Set(const std::string& property, bool value);
  // 所有数值类型共用一个入口：int64_t 在不同平台上是 long 或 long long，固定的重载集合会产生歧义。
  template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T> && !std::is_same_v<T, bool>>>
  Element& Set(const std::string& property, T value) {
    if constexpr (std::is_floating_point_v<T>) {
      return Store(property, config::FormatNumber(static_cast<double>(value)), false, {});
    } else {
      return Store(property, std::to_string(value), false, {});
    }
  }
  // 敏感值：Describe 打码，也不进入能力清单的取值检查。
  Element& SetSecret(const std::string& property, const std::string& value);
  // 配置透传的属性，覆盖框架或块设置的同名属性。
  Element& Apply(const config::Properties& properties);

  Endpoint Pad(const std::string& pad) const { return {index_, pad}; }
  operator Endpoint() const { return {index_, {}}; }  // NOLINT: 允许直接作为端点

  const std::string& name() const;
  const std::string& factory() const;
  bool valid() const { return graph_ != nullptr; }

 private:
  friend class Graph;
  Element(Graph* graph, int index) : graph_(graph), index_(index) {}
  Element& Store(const std::string& property, std::string value, bool secret, std::string origin);
  Graph* graph_ = nullptr;
  int index_ = -1;
};

class Graph {
 public:
  struct PropertyValue {
    std::string name;
    std::string value;
    bool secret = false;
    std::string origin;  // 来自配置时为配置路径
  };
  struct Node {
    std::string factory;
    std::string name;
    std::vector<PropertyValue> properties;
  };
  struct Edge {
    Endpoint from;
    Endpoint to;
  };

  // 名称为空时自动命名为 <工厂>N；重名抛 std::invalid_argument。
  Element Add(const std::string& factory, const std::string& name = {});
  // capsfilter，caps 为 GStreamer caps 文本。
  Element Caps(const std::string& caps, const std::string& name = {});
  void Link(const Endpoint& from, const Endpoint& to);
  // 依次连接，返回最后一个元件。
  Element Chain(std::initializer_list<Element> elements);
  // from → tee；Branch(tee) 返回新分支上的 queue（tee 之后必须有 queue，否则各分支会互相阻塞）。
  Element Tee(const Endpoint& from, const std::string& name = {});
  Element Branch(const Element& tee, const std::string& queue_name = {});

  bool Contains(const std::string& name) const;
  Element Find(const std::string& name);  // 不存在时抛 std::out_of_range

  const std::vector<Node>& nodes() const { return nodes_; }
  const std::vector<Edge>& links() const { return links_; }

  gst::Requirements Requirements() const;

  // 排空注入点：紧接在每个源元件（没有输入连接的元件）之后的元件名。
  // 停止时 EOS 从这里注入，而不是发给源本身：带重连逻辑的源（nvurisrcbin 的 RTSP）
  // 会把 EOS 当作断流处理而不向下游传递。
  std::vector<std::string> DrainPoints() const;
  std::string Describe(bool redact = true) const;

  // 创建 GstPipeline 并返回唯一引用；任何失败都释放已创建的对象并抛 std::runtime_error。
  GstElement* Instantiate(const std::string& pipeline_name) const;

 private:
  friend class Element;
  std::string EndpointText(const Endpoint& endpoint) const;
  std::vector<Node> nodes_;
  std::vector<Edge> links_;
};

}  // namespace dsf::gst

#endif  // DSF_GST_GRAPH_H
