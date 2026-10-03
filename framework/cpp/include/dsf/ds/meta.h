// 元数据模型：不含 SDK 类型的纯数据，媒体线程复制后可以安全地交给其他线程。
// 由 dsf::ds::MetaTap（需要 SDK）从 NvDsBatchMeta 转换而来；也可以在测试中直接构造。

#ifndef DSF_DS_META_H
#define DSF_DS_META_H

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace dsf::ds {

// 像素坐标，坐标空间是 mux 输出尺寸（FrameRecord::width/height），不是相机原始分辨率。
struct Box {
  double left = 0;
  double top = 0;
  double width = 0;
  double height = 0;
};

// 二级推理（分类器）的结果。
struct Classification {
  int component_id = 0;  // 产生结果的推理单元（nvinfer 的 unique-id）
  int class_id = 0;
  std::string label;
  double confidence = 0;
};

struct ObjectRecord {
  int class_id = 0;
  std::string label;
  double confidence = 0;
  Box box;
  std::optional<std::uint64_t> track_id;  // 只在被跟踪时有值；0 是合法编号
  int component_id = 0;                   // 产生该目标的推理单元
  std::vector<Classification> classes;
  nlohmann::json extra = nlohmann::json::object();  // 提取器追加的数据（例如 analytics 状态）
};

struct FrameRecord {
  std::string source_id;
  std::uint32_t pad_index = 0;
  std::uint64_t frame_num = 0;
  std::uint64_t pts_ns = 0;
  std::uint64_t ntp_ms = 0;   // 源提供的 NTP 时间（RTSP 的发送端时钟），未提供时为 0
  std::uint64_t wall_ms = 0;  // 主机处理时的墙钟
  int width = 0;
  int height = 0;
  std::vector<ObjectRecord> objects;
  nlohmann::json extra = nlohmann::json::object();
};

struct BatchRecord {
  std::vector<FrameRecord> frames;
};

// 源 id 与 mux pad 索引的唯一映射。由应用按 sources 顺序建立，元数据与录像都从这里取 id。
class SourceRegistry {
 public:
  void Add(std::string id) { ids_.push_back(std::move(id)); }
  // 未登记的索引返回 "source<N>"（例如运行期新增的源尚未登记）。
  std::string IdOf(std::uint32_t pad_index) const {
    return pad_index < ids_.size() ? ids_[pad_index] : "source" + std::to_string(pad_index);
  }
  std::optional<std::uint32_t> IndexOf(const std::string& id) const {
    for (std::size_t i = 0; i < ids_.size(); ++i) {
      if (ids_[i] == id) return static_cast<std::uint32_t>(i);
    }
    return std::nullopt;
  }
  std::size_t size() const { return ids_.size(); }
  const std::vector<std::string>& ids() const { return ids_; }

 private:
  std::vector<std::string> ids_;
};

std::uint64_t WallMilliseconds();

}  // namespace dsf::ds

#endif  // DSF_DS_META_H
