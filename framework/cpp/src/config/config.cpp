// 配置读取的实现。YAML 先转换为自有的有序树（拒绝重复键等），读取器只操作这棵树。

#include "dsf/config/config.h"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <fstream>
#include <limits>
#include <locale>
#include <set>
#include <sstream>
#include <utility>

namespace dsf::config {
namespace detail {

struct Value {
  enum class Kind { kNull, kScalar, kMap, kList };
  Kind kind = Kind::kNull;
  std::string scalar;
  bool quoted = false;  // 引号标量只能读成字符串，"123" 不会被当成整数
  std::vector<std::pair<std::string, Value>> map;
  std::vector<Value> list;
};

}  // namespace detail

namespace {

using detail::Value;

std::string Join(const std::string& parent, std::string_view key) {
  return parent.empty() ? std::string(key) : parent + "." + std::string(key);
}

std::string Item(const std::string& parent, std::size_t index) {
  return parent + "[" + std::to_string(index) + "]";
}

std::string ParentOf(const std::string& path) {
  const std::size_t dot = path.rfind('.');
  return dot == std::string::npos ? std::string() : path.substr(0, dot);
}

std::string LeafOf(const std::string& path) {
  const std::size_t dot = path.rfind('.');
  return dot == std::string::npos ? path : path.substr(dot + 1);
}

[[noreturn]] void Throw(const std::string& origin, const std::string& path,
                        const std::string& reason) {
  throw Error(origin + ": " + (path.empty() ? std::string("(top level)") : path) + ": " + reason);
}

Value Convert(const YAML::Node& node, const std::string& origin, const std::string& path) {
  Value value;
  switch (node.Type()) {
    case YAML::NodeType::Undefined:
    case YAML::NodeType::Null:
      return value;
    case YAML::NodeType::Scalar:
      value.kind = Value::Kind::kScalar;
      value.scalar = node.Scalar();
      value.quoted = node.Tag() == "!";
      return value;
    case YAML::NodeType::Sequence:
      value.kind = Value::Kind::kList;
      for (std::size_t index = 0; index < node.size(); ++index) {
        value.list.push_back(Convert(node[index], origin, Item(path, index)));
      }
      return value;
    case YAML::NodeType::Map: {
      value.kind = Value::Kind::kMap;
      std::set<std::string> seen;
      for (const auto& entry : node) {
        if (!entry.first.IsScalar()) Throw(origin, path, "mapping keys must be strings");
        const std::string key = entry.first.Scalar();
        if (key.empty()) Throw(origin, path, "empty mapping key");
        const std::string child = Join(path, key);
        // 点号键会让 "a.b" 与嵌套的 a: {b:} 成为同一路径，必须写成嵌套映射。
        if (key.find('.') != std::string::npos) {
          Throw(origin, child, "use nested mappings instead of dotted keys");
        }
        if (!seen.insert(key).second) Throw(origin, child, "duplicate key");
        value.map.emplace_back(key, Convert(entry.second, origin, child));
      }
      return value;
    }
  }
  return value;
}

void CollectLeaves(const Value& value, const std::string& path, std::vector<std::string>* out) {
  if (value.kind == Value::Kind::kMap && !value.map.empty()) {
    for (const auto& [key, child] : value.map) CollectLeaves(child, Join(path, key), out);
  } else if (value.kind == Value::Kind::kList && !value.list.empty() &&
             value.list.front().kind == Value::Kind::kMap) {
    for (std::size_t i = 0; i < value.list.size(); ++i) {
      CollectLeaves(value.list[i], Item(path, i), out);
    }
  } else if (!path.empty()) {
    out->push_back(path);
  }
}

const char* KindName(const Value& value) {
  switch (value.kind) {
    case Value::Kind::kNull: return "null";
    case Value::Kind::kScalar: return "a scalar";
    case Value::Kind::kMap: return "a mapping";
    case Value::Kind::kList: return "a list";
  }
  return "a value";
}

std::string Quote(const std::string& text) {
  if (text.empty()) return "\"\"";
  return text.find_first_of(" \t") == std::string::npos ? text : "\"" + text + "\"";
}

// ---- 标量转换 ----

template <typename T>
struct Scalar;

template <>
struct Scalar<std::string> {
  static constexpr const char* kName = "a string";
  static std::optional<std::string> Parse(const Value& value) { return value.scalar; }
  static std::string Text(const std::string& value) { return value; }
};

template <typename Integer>
std::optional<Integer> ParseInteger(const Value& value) {
  if (value.quoted) return std::nullopt;
  const std::string& text = value.scalar;
  Integer result{};
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), result);
  if (error != std::errc() || end != text.data() + text.size()) return std::nullopt;
  return result;
}

template <>
struct Scalar<int> {
  static constexpr const char* kName = "an integer";
  static std::optional<int> Parse(const Value& value) { return ParseInteger<int>(value); }
  static std::string Text(int value) { return std::to_string(value); }
};

template <>
struct Scalar<std::int64_t> {
  static constexpr const char* kName = "an integer";
  static std::optional<std::int64_t> Parse(const Value& value) {
    return ParseInteger<std::int64_t>(value);
  }
  static std::string Text(std::int64_t value) { return std::to_string(value); }
};

template <>
struct Scalar<double> {
  static constexpr const char* kName = "a finite number";
  static std::optional<double> Parse(const Value& value) {
    if (value.quoted || value.scalar.empty()) return std::nullopt;
    std::istringstream stream(value.scalar);
    stream.imbue(std::locale::classic());  // 小数点不随进程 locale 变化
    double result = 0.0;
    stream >> result;
    if (stream.fail() || !stream.eof() || !std::isfinite(result)) return std::nullopt;
    return result;
  }
  static std::string Text(double value) {
    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream << value;
    return stream.str();
  }
};

template <>
struct Scalar<bool> {
  static constexpr const char* kName = "a boolean (true/false)";
  static std::optional<bool> Parse(const Value& value) {
    if (value.quoted) return std::nullopt;
    std::string text = value.scalar;
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (text == "true" || text == "yes" || text == "on") return true;
    if (text == "false" || text == "no" || text == "off") return false;
    return std::nullopt;
  }
  static std::string Text(bool value) { return value ? "true" : "false"; }
};

}  // namespace

// ---- 约束 ----

Check<std::string> OneOf(std::vector<std::string> allowed) {
  return [allowed = std::move(allowed)](const std::string& value) -> std::optional<std::string> {
    if (std::find(allowed.begin(), allowed.end(), value) != allowed.end()) return std::nullopt;
    std::string text = "'" + value + "' is not one of:";
    for (const auto& item : allowed) text += " " + item;
    return text;
  };
}

Check<std::string> NonEmpty() {
  return [](const std::string& value) -> std::optional<std::string> {
    if (value.empty()) return std::string("must not be empty");
    return std::nullopt;
  };
}

Check<std::string> PathComponent() {
  return [](const std::string& value) -> std::optional<std::string> {
    if (value.empty() || value == "." || value == ".." ||
        value.find_first_of("/\\") != std::string::npos) {
      return "'" + value + "' must be a single path component (no '/', not '.' or '..')";
    }
    return std::nullopt;
  };
}

Check<int> AtLeast(int minimum) {
  return [minimum](const int& value) -> std::optional<std::string> {
    if (value < minimum) return "must be at least " + std::to_string(minimum);
    return std::nullopt;
  };
}

Check<int> Range(int minimum, int maximum) {
  return [minimum, maximum](const int& value) -> std::optional<std::string> {
    if (value < minimum || value > maximum) {
      return "must be between " + std::to_string(minimum) + " and " + std::to_string(maximum);
    }
    return std::nullopt;
  };
}

Check<double> AtLeast(double minimum) {
  return [minimum](const double& value) -> std::optional<std::string> {
    if (value < minimum) return "must be at least " + Scalar<double>::Text(minimum);
    return std::nullopt;
  };
}

// ---- Document ----

Document Document::LoadFile(const std::string& path) {
  std::ifstream stream(path);
  if (!stream) throw Error(path + ": cannot open config file");
  std::ostringstream buffer;
  buffer << stream.rdbuf();
  return LoadText(buffer.str(), path);
}

Document Document::LoadText(const std::string& text, const std::string& origin) {
  YAML::Node node;
  try {
    node = YAML::Load(text);
  } catch (const YAML::Exception& error) {
    throw Error(origin + ": cannot parse YAML: " + error.what());
  }
  Value root = Convert(node, origin, "");
  if (root.kind != Value::Kind::kNull && root.kind != Value::Kind::kMap) {
    Throw(origin, "", "top level must be a mapping");
  }
  Document document;
  document.root_ = std::make_shared<const Value>(std::move(root));
  document.origin_ = origin;
  return document;
}

const std::string& Document::origin() const { return origin_; }

std::vector<std::string> Document::LeafPaths() const {
  std::vector<std::string> paths;
  CollectLeaves(*root_, "", &paths);
  return paths;
}

// ---- Reader ----

struct Reader::State {
  std::shared_ptr<const Value> root;
  std::string origin;
  std::set<std::string> leaves;      // 作为叶子读取过的路径（含整棵 Props 子树）
  std::set<std::string> containers;  // 作为映射或列表进入过的路径
  std::vector<std::string> declared;  // 叶子声明顺序（去重）
  std::set<std::string> declared_set;
  std::set<std::string> known;  // 叶子与容器，用于提示"这里能写什么"
  std::vector<std::pair<std::string, std::string>> described;
  bool finished = false;
};

Reader Document::Root() const {
  auto state = std::make_shared<Reader::State>();
  state->root = root_;
  state->origin = origin_;
  const Value* node = root_->kind == Value::Kind::kMap ? root_.get() : nullptr;
  return Reader(std::move(state), node, "");
}

Reader::Reader(std::shared_ptr<State> state, const Value* node, std::string path)
    : state_(std::move(state)), node_(node), path_(std::move(path)) {}

std::string Reader::PathOf(std::string_view key) const {
  return key.empty() ? path_ : Join(path_, key);
}

void Reader::Fail(std::string_view key, const std::string& reason) const {
  Throw(state_->origin, PathOf(key), reason);
}

const Value* Reader::Child(std::string_view key) const {
  if (node_ == nullptr) return nullptr;
  for (const auto& [name, value] : node_->map) {
    if (name == key) return value.kind == Value::Kind::kNull ? nullptr : &value;
  }
  return nullptr;
}

bool Reader::Has(std::string_view key) const { return Child(key) != nullptr; }

template <typename T>
std::optional<T> Reader::Read(std::string_view key, const Check<T>& check, bool secret,
                              const std::string& fallback_text) {
  const std::string path = PathOf(key);
  if (state_->declared_set.insert(path).second) state_->declared.push_back(path);
  state_->known.insert(path);
  state_->leaves.insert(path);
  const Value* node = Child(key);
  std::optional<T> result;
  if (node != nullptr) {
    if constexpr (std::is_same_v<T, std::vector<std::string>>) {
      if (node->kind != Value::Kind::kList) Fail(key, std::string("expected a list, got ") + KindName(*node));
      T items;
      for (const Value& item : node->list) {
        if (item.kind != Value::Kind::kScalar) Fail(key, "expected a list of scalars");
        items.push_back(item.scalar);
      }
      result = std::move(items);
    } else {
      if (node->kind != Value::Kind::kScalar) {
        Fail(key, std::string("expected ") + Scalar<T>::kName + ", got " + KindName(*node));
      }
      result = Scalar<T>::Parse(*node);
      if (!result) {
        Fail(key, std::string("expected ") + Scalar<T>::kName + ", got '" + node->scalar + "'");
      }
    }
    if (check) {
      if (auto problem = check(*result)) Fail(key, *problem);
    }
  }
  std::string text = fallback_text;
  if (result) {
    if constexpr (std::is_same_v<T, std::vector<std::string>>) {
      text.clear();
      for (const auto& item : *result) text += (text.empty() ? "" : ",") + item;
    } else {
      text = Scalar<T>::Text(*result);
    }
  }
  state_->described.emplace_back(path, secret ? std::string("***") : Quote(text));
  return result;
}

template <typename T>
T Reader::Get(std::string_view key, T fallback, detail::CheckOf<T> check) {
  std::string text;
  if constexpr (std::is_same_v<T, std::vector<std::string>>) {
    for (const auto& item : fallback) text += (text.empty() ? "" : ",") + item;
  } else {
    text = Scalar<T>::Text(fallback);
  }
  auto value = Read<T>(key, check, false, text);
  return value ? std::move(*value) : std::move(fallback);
}

template <typename T>
T Reader::Required(std::string_view key, detail::CheckOf<T> check) {
  auto value = Read<T>(key, check, false, "");
  if (!value) Fail(key, "is required");
  return std::move(*value);
}

std::string Reader::Secret(std::string_view key, std::string fallback) {
  auto value = Read<std::string>(key, {}, true, fallback);
  return value ? std::move(*value) : std::move(fallback);
}

Reader Reader::Section(std::string_view key) {
  const std::string path = PathOf(key);
  state_->containers.insert(path);
  state_->known.insert(path);
  const Value* node = Child(key);
  if (node != nullptr && node->kind != Value::Kind::kMap) {
    Fail(key, std::string("expected a mapping, got ") + KindName(*node));
  }
  return Reader(state_, node, path);
}

std::vector<Reader> Reader::List(std::string_view key, std::size_t min_items) {
  const std::string path = PathOf(key);
  state_->containers.insert(path);
  state_->known.insert(path);
  const Value* node = Child(key);
  std::vector<Reader> items;
  if (node != nullptr) {
    if (node->kind != Value::Kind::kList) {
      Fail(key, std::string("expected a list, got ") + KindName(*node));
    }
    for (std::size_t i = 0; i < node->list.size(); ++i) {
      const Value& item = node->list[i];
      const std::string item_path = Item(path, i);
      if (item.kind != Value::Kind::kMap) {
        Throw(state_->origin, item_path, std::string("expected a mapping, got ") + KindName(item));
      }
      state_->containers.insert(item_path);
      items.push_back(Reader(state_, &item, item_path));
    }
  }
  if (items.size() < min_items) {
    Fail(key, "needs at least " + std::to_string(min_items) + " item(s)");
  }
  return items;
}

Properties Reader::Props(std::string_view key) {
  const std::string path = PathOf(key);
  if (state_->declared_set.insert(path).second) state_->declared.push_back(path);
  state_->known.insert(path);
  state_->leaves.insert(path);
  Properties properties;
  const Value* node = Child(key);
  if (node == nullptr) return properties;
  if (node->kind != Value::Kind::kMap) {
    Fail(key, std::string("expected a mapping of properties, got ") + KindName(*node));
  }
  for (const auto& [name, value] : node->map) {
    const std::string origin = Join(path, name);
    if (value.kind != Value::Kind::kScalar) {
      Throw(state_->origin, origin, "property values must be scalars");
    }
    properties.push_back({name, value.scalar, origin});
    state_->described.emplace_back(origin, Quote(value.scalar));
  }
  return properties;
}

void Reader::Finish() const {
  if (!path_.empty()) throw Error("Reader::Finish must be called on the root reader");
  const std::function<void(const Value&, const std::string&)> walk =
      [&](const Value& value, const std::string& path) {
        for (const auto& [key, child] : value.map) {
          const std::string child_path = Join(path, key);
          if (state_->leaves.count(child_path)) continue;
          if (state_->containers.count(child_path)) {
            if (child.kind == Value::Kind::kMap) walk(child, child_path);
            for (std::size_t i = 0; i < child.list.size(); ++i) {
              walk(child.list[i], Item(child_path, i));
            }
            continue;
          }
          // 未读取的键：列出同一层级允许的键，拼错时不必翻代码找正确写法。
          std::string allowed;
          const std::string parent = path;
          for (const std::string& known : state_->known) {
            if (ParentOf(known) == parent) allowed += (allowed.empty() ? "" : " ") + LeafOf(known);
          }
          Throw(state_->origin, child_path,
                "unknown field; allowed here: " + (allowed.empty() ? std::string("(none)") : allowed));
        }
      };
  if (state_->root->kind == Value::Kind::kMap) walk(*state_->root, "");
}

std::string Reader::Describe() const {
  std::string text;
  for (const auto& [path, value] : state_->described) {
    text += (text.empty() ? "" : " ") + path + "=" + value;
  }
  return text;
}

std::vector<std::string> Reader::DeclaredPaths() const { return state_->declared; }

template std::string Reader::Get<std::string>(std::string_view, std::string,
                                              detail::CheckOf<std::string>);
template int Reader::Get<int>(std::string_view, int, detail::CheckOf<int>);
template std::int64_t Reader::Get<std::int64_t>(std::string_view, std::int64_t,
                                                detail::CheckOf<std::int64_t>);
template double Reader::Get<double>(std::string_view, double, detail::CheckOf<double>);
template bool Reader::Get<bool>(std::string_view, bool, detail::CheckOf<bool>);
template std::vector<std::string> Reader::Get<std::vector<std::string>>(
    std::string_view, std::vector<std::string>, detail::CheckOf<std::vector<std::string>>);
template std::string Reader::Required<std::string>(std::string_view, detail::CheckOf<std::string>);
template int Reader::Required<int>(std::string_view, detail::CheckOf<int>);
template std::int64_t Reader::Required<std::int64_t>(std::string_view,
                                                     detail::CheckOf<std::int64_t>);
template double Reader::Required<double>(std::string_view, detail::CheckOf<double>);
template bool Reader::Required<bool>(std::string_view, detail::CheckOf<bool>);

}  // namespace dsf::config
