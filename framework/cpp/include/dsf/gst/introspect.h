// GStreamer 初始化与插件内省：按实际加载的插件检查工厂、属性、属性值和信号，
// 不按 SDK 版本推断能力。同一份代码因此能适配多个 DeepStream 版本与平台。

#ifndef DSF_GST_INTROSPECT_H
#define DSF_GST_INTROSPECT_H

#include <string>
#include <vector>

#include <gst/gst.h>

namespace dsf::gst {

// 幂等；每个使用 GStreamer 的入口都先调用它。
void EnsureInitialized();

std::string GStreamerVersion();

enum class Availability { kMissing, kLoadFailed, kReady };

// 查询工厂状态：不创建元件实例，不占用设备。
Availability FactoryAvailability(const std::string& factory);
bool HasFactory(const std::string& factory);
bool HasProperty(const std::string& factory, const std::string& property);
bool HasSignal(const std::string& factory, const std::string& signal);

// 把文本转换为属性的 GValue（数值、布尔、字符串、枚举名/值、flags、caps）并做范围校验。
// out 必须是未初始化的 GValue（G_VALUE_INIT）；成功时由调用方 g_value_unset。
// 失败返回 false 并写入原因。
bool ParsePropertyValue(GParamSpec* spec, const std::string& text, GValue* out,
                        std::string* error);

// 启动前要满足的能力清单。图模型自动生成，组件可以追加（例如录像需要的信号）。
class Requirements {
 public:
  struct Item {
    enum class Kind { kFactory, kProperty, kValue, kSignal };
    Kind kind;
    std::string factory;
    std::string name;    // 属性名或信号名
    std::string value;   // kValue：属性值文本
    std::string origin;  // 来源（配置路径或元件名），出现在错误消息里
  };

  void Factory(const std::string& factory, const std::string& origin = {});
  void Property(const std::string& factory, const std::string& property,
                const std::string& origin = {});
  void Value(const std::string& factory, const std::string& property, const std::string& value,
             const std::string& origin = {});
  void Signal(const std::string& factory, const std::string& signal,
              const std::string& origin = {});
  void Merge(const Requirements& other);

  const std::vector<Item>& items() const { return items_; }

  // 逐项检查，汇总全部不满足项后抛 std::runtime_error。
  void Check() const;

  // 每行一项，供 --dump-pipeline 输出。
  std::string Describe() const;

 private:
  void Add(Item item);
  std::vector<Item> items_;
};

}  // namespace dsf::gst

#endif  // DSF_GST_INTROSPECT_H
