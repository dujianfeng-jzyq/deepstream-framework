// 图模型的实现。

#include "dsf/gst/graph.h"

#include <algorithm>
#include <memory>
#include <stdexcept>

namespace dsf::gst {
namespace {

std::string DescribeValue(const std::string& value) {
  if (!value.empty() && value.find_first_of(" \t\"\\,;") == std::string::npos) return value;
  std::string quoted = "\"";
  for (char c : value) {
    if (c == '"' || c == '\\') quoted += '\\';
    quoted += c;
  }
  return quoted + "\"";
}

// 静态 pad 优先；否则按名称或模板申请 request pad。返回持有的引用，不存在返回 nullptr。
GstPad* AcquirePad(GstElement* element, const std::string& name) {
  if (GstPad* pad = gst_element_get_static_pad(element, name.c_str())) return pad;
  return gst_element_request_pad_simple(element, name.c_str());
}

bool HasSometimesSource(GstElement* element) {
  for (const GList* item = gst_element_class_get_pad_template_list(GST_ELEMENT_GET_CLASS(element));
       item != nullptr; item = item->next) {
    auto* templ = static_cast<GstPadTemplate*>(item->data);
    if (GST_PAD_TEMPLATE_DIRECTION(templ) == GST_PAD_SRC &&
        GST_PAD_TEMPLATE_PRESENCE(templ) == GST_PAD_SOMETIMES) {
      return true;
    }
  }
  return false;
}

// 已存在且未连接的 src pad：有它时连接失败是真实错误（caps 不兼容等），不能推迟成等待 pad-added，
// 否则错误会变成运行时无声的卡住。
bool HasUnlinkedSource(GstElement* element) {
  bool found = false;
  GstIterator* pads = gst_element_iterate_src_pads(element);
  GValue item = G_VALUE_INIT;
  while (!found && gst_iterator_next(pads, &item) == GST_ITERATOR_OK) {
    found = !gst_pad_is_linked(GST_PAD(g_value_get_object(&item)));
    g_value_reset(&item);
  }
  g_value_unset(&item);
  gst_iterator_free(pads);
  return found;
}

// 源元件运行后才出现的 pad（nvurisrcbin、decodebin 等）：pad-added 时连接第一个能连上的。
struct DelayedLink {
  GstElement* sink;  // 由 bin 持有，回调期间有效
  std::string sink_pad;
  std::string src_pad;  // 为空表示任意
  bool done = false;
};

void OnPadAdded(GstElement* /*element*/, GstPad* pad, gpointer data) {
  auto* link = static_cast<DelayedLink*>(data);
  if (link->done || GST_PAD_DIRECTION(pad) != GST_PAD_SRC) return;
  if (!link->src_pad.empty()) {
    gchar* name = gst_pad_get_name(pad);
    const bool matches = link->src_pad == name;
    g_free(name);
    if (!matches) return;
  }
  GstPad* sink = link->sink_pad.empty()
                     ? nullptr
                     : AcquirePad(link->sink, link->sink_pad);
  if (sink == nullptr && link->sink_pad.empty()) {
    sink = gst_element_get_compatible_pad(link->sink, pad, nullptr);
  }
  if (sink == nullptr) return;
  // 音频等不兼容的 pad 跳过，等下一个。
  if (gst_pad_link(pad, sink) == GST_PAD_LINK_OK) {
    link->done = true;
  } else if (GST_PAD_TEMPLATE_PRESENCE(GST_PAD_PAD_TEMPLATE(sink)) == GST_PAD_REQUEST) {
    gst_element_release_request_pad(link->sink, sink);
  }
  gst_object_unref(sink);
}

void FreeDelayedLink(gpointer data, GClosure* /*closure*/) {
  delete static_cast<DelayedLink*>(data);
}

}  // namespace

// ---- Element ----

Element& Element::Store(const std::string& property, std::string value, bool secret,
                        std::string origin) {
  auto& properties = graph_->nodes_[static_cast<std::size_t>(index_)].properties;
  for (auto& existing : properties) {
    if (existing.name == property) {
      existing.value = std::move(value);
      existing.secret = secret;
      existing.origin = std::move(origin);
      return *this;
    }
  }
  properties.push_back({property, std::move(value), secret, std::move(origin)});
  return *this;
}

Element& Element::Set(const std::string& property, const std::string& value) {
  return Store(property, value, false, {});
}
Element& Element::Set(const std::string& property, const char* value) {
  return Store(property, value, false, {});
}
Element& Element::Set(const std::string& property, bool value) {
  return Store(property, value ? "true" : "false", false, {});
}
Element& Element::SetSecret(const std::string& property, const std::string& value) {
  return Store(property, value, true, {});
}
Element& Element::Apply(const config::Properties& properties) {
  for (const auto& property : properties) {
    Store(property.name, property.value, false, property.origin);
  }
  return *this;
}

const std::string& Element::name() const {
  return graph_->nodes_[static_cast<std::size_t>(index_)].name;
}
const std::string& Element::factory() const {
  return graph_->nodes_[static_cast<std::size_t>(index_)].factory;
}

// ---- Graph ----

Element Graph::Add(const std::string& factory, const std::string& name) {
  std::string actual = name;
  if (actual.empty()) {
    for (int n = 0;; ++n) {
      actual = factory + std::to_string(n);
      if (!Contains(actual)) break;
    }
  } else if (Contains(actual)) {
    throw std::invalid_argument("duplicate element name in graph: " + actual);
  }
  nodes_.push_back({factory, actual, {}});
  return Element(this, static_cast<int>(nodes_.size() - 1));
}

Element Graph::Caps(const std::string& caps, const std::string& name) {
  Element element = Add("capsfilter", name);
  element.Set("caps", caps);
  return element;
}

void Graph::Link(const Endpoint& from, const Endpoint& to) {
  const auto size = static_cast<int>(nodes_.size());
  if (from.element < 0 || from.element >= size || to.element < 0 || to.element >= size) {
    throw std::invalid_argument("graph link refers to an unknown element");
  }
  links_.push_back({from, to});
}

Element Graph::Chain(std::initializer_list<Element> elements) {
  const Element* previous = nullptr;
  for (const Element& element : elements) {
    if (previous != nullptr) Link(*previous, element);
    previous = &element;
  }
  if (previous == nullptr) throw std::invalid_argument("Chain needs at least one element");
  return *previous;
}

Element Graph::Tee(const Endpoint& from, const std::string& name) {
  Element tee = Add("tee", name);
  Link(from, tee);
  return tee;
}

Element Graph::Branch(const Element& tee, const std::string& queue_name) {
  Element queue = Add("queue", queue_name);
  Link(tee.Pad("src_%u"), queue);
  return queue;
}

bool Graph::Contains(const std::string& name) const {
  for (const Node& node : nodes_) {
    if (node.name == name) return true;
  }
  return false;
}

Element Graph::Find(const std::string& name) {
  for (std::size_t i = 0; i < nodes_.size(); ++i) {
    if (nodes_[i].name == name) return Element(this, static_cast<int>(i));
  }
  throw std::out_of_range("no element named " + name + " in graph");
}

std::string Graph::EndpointText(const Endpoint& endpoint) const {
  const std::string& name = nodes_[static_cast<std::size_t>(endpoint.element)].name;
  return endpoint.pad.empty() ? name : name + "." + endpoint.pad;
}

gst::Requirements Graph::Requirements() const {
  gst::Requirements requirements;
  for (const Node& node : nodes_) {
    requirements.Factory(node.factory, node.name);
    for (const PropertyValue& property : node.properties) {
      const std::string origin = property.origin.empty() ? node.name : property.origin;
      if (property.secret) {
        requirements.Property(node.factory, property.name, origin);
      } else {
        requirements.Value(node.factory, property.name, property.value, origin);
      }
    }
  }
  return requirements;
}

std::vector<std::string> Graph::DrainPoints() const {
  std::vector<bool> has_input(nodes_.size(), false);
  for (const Edge& link : links_) has_input[static_cast<std::size_t>(link.to.element)] = true;
  std::vector<std::string> points;
  for (const Edge& link : links_) {
    if (has_input[static_cast<std::size_t>(link.from.element)]) continue;
    const std::string& name = nodes_[static_cast<std::size_t>(link.to.element)].name;
    if (std::find(points.begin(), points.end(), name) == points.end()) points.push_back(name);
  }
  return points;
}

std::string Graph::Describe(bool redact) const {
  std::string text;
  for (const Node& node : nodes_) {
    text += node.name + " = " + node.factory;
    for (const PropertyValue& property : node.properties) {
      const std::string value = property.secret && redact ? "***" : property.value;
      text += " " + property.name + "=" + DescribeValue(value);
    }
    text += "\n";
  }
  for (const Edge& link : links_) {
    text += EndpointText(link.from) + " -> " + EndpointText(link.to) + "\n";
  }
  return text;
}

GstElement* Graph::Instantiate(const std::string& pipeline_name) const {
  EnsureInitialized();
  GstElement* pipeline = gst_pipeline_new(pipeline_name.c_str());
  // 任何失败都要释放已创建的管线；元件由 bin 持有，随之释放。
  std::unique_ptr<GstElement, void (*)(GstElement*)> guard(
      pipeline, [](GstElement* p) { gst_object_unref(p); });
  std::vector<GstElement*> elements;
  for (const Node& node : nodes_) {
    GstElement* element = gst_element_factory_make(node.factory.c_str(), node.name.c_str());
    if (element == nullptr) {
      throw std::runtime_error("cannot create element '" + node.name + "' (" + node.factory +
                               "): plugin missing or failed to load");
    }
    gst_bin_add(GST_BIN(pipeline), element);
    elements.push_back(element);
    for (const PropertyValue& property : node.properties) {
      const std::string origin = property.origin.empty() ? node.name : property.origin;
      GParamSpec* spec =
          g_object_class_find_property(G_OBJECT_GET_CLASS(element), property.name.c_str());
      if (spec == nullptr) {
        throw std::runtime_error(origin + ": element '" + node.factory + "' has no property '" +
                                 property.name + "'");
      }
      GValue value = G_VALUE_INIT;
      std::string error;
      if (!ParsePropertyValue(spec, property.value, &value, &error)) {
        throw std::runtime_error(origin + ": " + error);
      }
      g_object_set_property(G_OBJECT(element), property.name.c_str(), &value);
      g_value_unset(&value);
    }
  }

  for (const Edge& link : links_) {
    GstElement* src = elements[static_cast<std::size_t>(link.from.element)];
    GstElement* sink = elements[static_cast<std::size_t>(link.to.element)];
    const std::string text = EndpointText(link.from) + " -> " + EndpointText(link.to);
    // sometimes pad 要等元件运行后才出现：登记 pad-added 回调，届时再连接。
    const auto delay = [&] {
      if (!HasSometimesSource(src)) throw std::runtime_error("cannot link " + text);
      auto* delayed = new DelayedLink{sink, link.to.pad, link.from.pad};
      g_signal_connect_data(src, "pad-added", G_CALLBACK(OnPadAdded), delayed, FreeDelayedLink,
                            GConnectFlags(0));
    };

    GstPad* src_pad = link.from.pad.empty() ? nullptr : AcquirePad(src, link.from.pad);
    if (src_pad == nullptr) {
      if (!link.from.pad.empty()) {
        delay();
        continue;
      }
      bool linked = false;
      if (link.to.pad.empty()) {
        linked = gst_element_link(src, sink) != FALSE;
      } else {
        GstPad* sink_pad = AcquirePad(sink, link.to.pad);
        if (sink_pad == nullptr) throw std::runtime_error("no pad '" + link.to.pad + "' for " + text);
        linked = gst_element_link_pads(src, nullptr, sink, GST_PAD_NAME(sink_pad)) != FALSE;
        if (!linked && GST_PAD_TEMPLATE_PRESENCE(GST_PAD_PAD_TEMPLATE(sink_pad)) == GST_PAD_REQUEST) {
          gst_element_release_request_pad(sink, sink_pad);
        }
        gst_object_unref(sink_pad);
      }
      if (!linked && HasUnlinkedSource(src)) throw std::runtime_error("cannot link " + text);
      if (!linked) delay();
      continue;
    }

    GstPad* sink_pad = link.to.pad.empty() ? gst_element_get_compatible_pad(sink, src_pad, nullptr)
                                           : AcquirePad(sink, link.to.pad);
    const bool linked = sink_pad != nullptr && gst_pad_link(src_pad, sink_pad) == GST_PAD_LINK_OK;
    if (sink_pad != nullptr) gst_object_unref(sink_pad);
    gst_object_unref(src_pad);
    if (!linked) throw std::runtime_error("cannot link " + text);
  }
  return guard.release();
}

}  // namespace dsf::gst
