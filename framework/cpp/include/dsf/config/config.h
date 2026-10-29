// 配置读取：读取代码本身就是 schema。
//
// 每次 Get/Required/Secret/Section/List/Props 都登记一个路径；Finish() 用登记集合
// 拒绝 YAML 中没有被读取过的键，并生成打码后的摘要。没有独立的声明表需要同步。
// 公共头文件不暴露 yaml-cpp 类型。

#ifndef DSF_CONFIG_CONFIG_H
#define DSF_CONFIG_CONFIG_H

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace dsf::config {

// 所有配置问题都抛这个异常，消息形如 "<来源>: <完整路径>: <原因>"。
class Error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// 透传给插件的属性：值保留 YAML 标量的原文，由 dsf::gst 按插件元数据转换。
struct Property {
  std::string name;
  std::string value;
  std::string origin;  // 完整配置路径，转换失败时用于报错
};
using Properties = std::vector<Property>;

// 值约束：返回空表示通过，否则返回失败原因（不含路径，Reader 会补上）。
template <typename T>
using Check = std::function<std::optional<std::string>(const T&)>;

namespace detail {
struct Value;  // 解析后的 YAML 节点，定义在实现文件
// 让约束参数不参与模板推导：Get("k", 5, AtLeast(0)) 由默认值决定 T。
template <typename T>
struct Identity {
  using type = T;
};
template <typename T>
using CheckOf = Check<typename Identity<T>::type>;
}  // namespace detail

// 浮点数的文本形式，与进程 locale 无关（小数点总是 .）。配置摘要与图模型的属性文本都用它，
// 读取（Get<double>）与属性转换按同样的规则解析。
std::string FormatNumber(double value);

Check<std::string> OneOf(std::vector<std::string> allowed);
Check<std::string> NonEmpty();
// 可作为单个路径段（目录名、文件名前缀）：非空、不含 / 与 \、不是 . 或 ..。
Check<std::string> PathComponent();
Check<int> AtLeast(int minimum);
Check<int> Range(int minimum, int maximum);
Check<double> AtLeast(double minimum);

class Reader;

// 解析后的 YAML 文档。加载时拒绝重复键、非字符串键、空键和带点号的键。
class Document {
 public:
  static Document LoadFile(const std::string& path);
  static Document LoadText(const std::string& text, const std::string& origin);

  // 根读取器。一个文档只应创建一个根读取器并在读完后 Finish()。
  Reader Root() const;

  const std::string& origin() const;

  // 文档中的全部叶子路径（列表元素记为 key[i]），供"模板与读取代码一致"的测试使用。
  std::vector<std::string> LeafPaths() const;

 private:
  std::shared_ptr<const detail::Value> root_;
  std::string origin_;
  friend class Reader;
};

class Reader {
 public:
  // 可选字段：键缺失或为 null 时返回默认值。
  template <typename T>
  T Get(std::string_view key, T fallback, detail::CheckOf<T> check = {});

  // 必填字段：缺失或为 null 时失败。
  template <typename T>
  T Required(std::string_view key, detail::CheckOf<T> check = {});

  // 敏感字符串：摘要中打码。
  std::string Secret(std::string_view key, std::string fallback = {});

  // 子映射。键缺失或为 null 时返回空读取器，其中的 Get 全部取默认值。
  Reader Section(std::string_view key);

  // 映射序列。键缺失或为 null 时为空列表。
  std::vector<Reader> List(std::string_view key, std::size_t min_items = 0);

  // 自由映射：名称到标量，整棵子树视为已读取。
  Properties Props(std::string_view key);

  bool Has(std::string_view key) const;

  // 完整路径，例如 sources[1].uri。
  std::string PathOf(std::string_view key) const;

  [[noreturn]] void Fail(std::string_view key, const std::string& reason) const;
  // 合法但多半不是想要的配置（例如浏览器无法播放的编码）：登记警告，不中断读取。消息格式与 Fail 相同，
  // 应用入口在 --check-config 与运行日志中输出。
  void Warn(std::string_view key, const std::string& reason) const;
  // 同一文档的全部读取器登记过的警告，按登记顺序。
  std::vector<std::string> Warnings() const;

  // 只能在根读取器上调用：检查未读取的键，失败抛 Error。
  void Finish() const;

  // 读取过的字段，按读取顺序 "path=value"，敏感值为 ***。
  std::string Describe() const;

  // 读取代码声明过的全部叶子路径（包括取了默认值的），与 Document::LeafPaths 对照。
  std::vector<std::string> DeclaredPaths() const;

 private:
  struct State;
  Reader(std::shared_ptr<State> state, const detail::Value* node, std::string path);

  template <typename T>
  std::optional<T> Read(std::string_view key, const Check<T>& check, bool secret,
                        const std::string& fallback_text);
  const detail::Value* Child(std::string_view key) const;

  std::shared_ptr<State> state_;  // 同一文档的全部读取器共享：登记表与文档所有权
  const detail::Value* node_;     // 为空表示整段缺失
  std::string path_;
  friend class Document;
};

// Get/Required 支持的类型：std::string、int、std::int64_t、double、bool、
// std::vector<std::string>（标量列表）。其他类型在链接时失败。

}  // namespace dsf::config

#endif  // DSF_CONFIG_CONFIG_H
