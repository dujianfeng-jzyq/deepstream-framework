// 插件内省的实现。注册表中的工厂未必已加载：必须先 load，再取 GObject 类查属性与信号。

#include "dsf/gst/introspect.h"

#include <mutex>
#include <stdexcept>

namespace dsf::gst {
namespace {

// 持有元件类型的类引用；构造时加载插件。类引用保证 GParamSpec 在使用期间有效，
// 也使信号完成注册（信号在 class_init 中注册，只拿 GType 去查会得到误导性的"不存在"）。
class FactoryClass {
 public:
  explicit FactoryClass(const std::string& factory) {
    EnsureInitialized();
    GstElementFactory* found = gst_element_factory_find(factory.c_str());
    if (found == nullptr) return;
    status_ = Availability::kLoadFailed;
    GstPluginFeature* loaded = gst_plugin_feature_load(GST_PLUGIN_FEATURE(found));
    gst_object_unref(found);
    if (loaded == nullptr) return;
    type_ = gst_element_factory_get_element_type(GST_ELEMENT_FACTORY(loaded));
    gst_object_unref(loaded);
    if (type_ == 0) return;
    klass_ = G_OBJECT_CLASS(g_type_class_ref(type_));
    if (klass_ != nullptr) status_ = Availability::kReady;
  }
  ~FactoryClass() {
    if (klass_ != nullptr) g_type_class_unref(klass_);
  }
  FactoryClass(const FactoryClass&) = delete;
  FactoryClass& operator=(const FactoryClass&) = delete;

  Availability status() const { return status_; }
  GParamSpec* Find(const std::string& property) const {
    return klass_ == nullptr ? nullptr : g_object_class_find_property(klass_, property.c_str());
  }
  bool HasSignal(const std::string& signal) const {
    return type_ != 0 && g_signal_lookup(signal.c_str(), type_) != 0;
  }

 private:
  Availability status_ = Availability::kMissing;
  GType type_ = 0;
  GObjectClass* klass_ = nullptr;
};

std::string Unavailable(const std::string& factory, Availability status) {
  if (status == Availability::kLoadFailed) {
    return "element '" + factory + "' is registered but cannot be loaded (missing plugin dependency)";
  }
  return "element '" + factory + "' is not available in this runtime";
}

std::string WithOrigin(const std::string& message, const std::string& origin) {
  return origin.empty() ? message : origin + ": " + message;
}

}  // namespace

void EnsureInitialized() {
  static std::once_flag once;
  std::call_once(once, [] {
    if (!gst_is_initialized()) gst_init(nullptr, nullptr);
  });
}

std::string GStreamerVersion() {
  EnsureInitialized();
  gchar* version = gst_version_string();
  std::string text = version;
  g_free(version);
  return text;
}

Availability FactoryAvailability(const std::string& factory) {
  return FactoryClass(factory).status();
}

bool HasFactory(const std::string& factory) {
  return FactoryAvailability(factory) == Availability::kReady;
}

bool HasProperty(const std::string& factory, const std::string& property) {
  return FactoryClass(factory).Find(property) != nullptr;
}

bool HasSignal(const std::string& factory, const std::string& signal) {
  return FactoryClass(factory).HasSignal(signal);
}

bool ParsePropertyValue(GParamSpec* spec, const std::string& text, GValue* out,
                        std::string* error) {
  if ((spec->flags & G_PARAM_WRITABLE) == 0) {
    *error = "property '" + std::string(spec->name) + "' is read-only";
    return false;
  }
  const GType type = spec->value_type;
  // 无符号类型的反序列化会把负数回绕成极大值，这里显式拒绝。
  if ((type == G_TYPE_UINT || type == G_TYPE_UINT64 || type == G_TYPE_ULONG || type == G_TYPE_UCHAR) &&
      text.find('-') != std::string::npos) {
    *error = "value '" + text + "' must not be negative for property '" + spec->name + "'";
    return false;
  }
  g_value_init(out, type);
  bool parsed = false;
  if (spec->value_type == G_TYPE_STRING) {
    // 字符串原样使用：gst_value_deserialize 会把引号和转义当作语法处理。
    g_value_set_string(out, text.c_str());
    parsed = true;
  } else {
    parsed = gst_value_deserialize(out, text.c_str()) != FALSE;
  }
  if (!parsed) {
    *error = "cannot convert '" + text + "' to " + g_type_name(spec->value_type) +
             " for property '" + spec->name + "'";
    g_value_unset(out);
    return false;
  }
  // g_param_value_validate 返回 TRUE 表示值被修正过，即原值超出范围或不在枚举中。
  if (g_param_value_validate(spec, out)) {
    *error = "value '" + text + "' is out of range for property '" + spec->name + "'";
    g_value_unset(out);
    return false;
  }
  return true;
}

void Requirements::Add(Item item) {
  for (const Item& existing : items_) {
    if (existing.kind == item.kind && existing.factory == item.factory &&
        existing.name == item.name && existing.value == item.value) {
      return;
    }
  }
  items_.push_back(std::move(item));
}

void Requirements::Factory(const std::string& factory, const std::string& origin) {
  Add({Item::Kind::kFactory, factory, {}, {}, origin});
}

void Requirements::Property(const std::string& factory, const std::string& property,
                            const std::string& origin) {
  Add({Item::Kind::kProperty, factory, property, {}, origin});
}

void Requirements::Value(const std::string& factory, const std::string& property,
                         const std::string& value, const std::string& origin) {
  Add({Item::Kind::kValue, factory, property, value, origin});
}

void Requirements::Signal(const std::string& factory, const std::string& signal,
                          const std::string& origin) {
  Add({Item::Kind::kSignal, factory, signal, {}, origin});
}

void Requirements::Merge(const Requirements& other) {
  for (const Item& item : other.items_) Add(item);
}

void Requirements::Check() const {
  std::string problems;
  for (const Item& item : items_) {
    const FactoryClass klass(item.factory);
    std::string problem;
    if (klass.status() != Availability::kReady) {
      problem = Unavailable(item.factory, klass.status());
    } else if (item.kind == Item::Kind::kSignal) {
      if (!klass.HasSignal(item.name)) {
        problem = "element '" + item.factory + "' has no signal '" + item.name + "'";
      }
    } else if (item.kind != Item::Kind::kFactory) {
      GParamSpec* spec = klass.Find(item.name);
      if (spec == nullptr) {
        problem = "element '" + item.factory + "' has no property '" + item.name + "'";
      } else if (item.kind == Item::Kind::kValue) {
        GValue value = G_VALUE_INIT;
        std::string error;
        if (ParsePropertyValue(spec, item.value, &value, &error)) {
          g_value_unset(&value);
        } else {
          problem = item.factory + ": " + error;
        }
      }
    }
    if (!problem.empty()) problems += "\n  " + WithOrigin(problem, item.origin);
  }
  if (!problems.empty()) {
    throw std::runtime_error("the runtime does not provide what the pipeline needs:" + problems);
  }
}

std::string Requirements::Describe() const {
  std::string text;
  for (const Item& item : items_) {
    switch (item.kind) {
      case Item::Kind::kFactory: text += "element  " + item.factory; break;
      case Item::Kind::kProperty: text += "property " + item.factory + "." + item.name; break;
      case Item::Kind::kValue:
        text += "value    " + item.factory + "." + item.name + "=" + item.value;
        break;
      case Item::Kind::kSignal: text += "signal   " + item.factory + "::" + item.name; break;
    }
    text += "\n";
  }
  return text;
}

}  // namespace dsf::gst
