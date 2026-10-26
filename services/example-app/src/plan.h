// example-app 的拓扑：只组图，不创建组件，因此可以在没有 SDK 的开发机上测试。
//
//   每路源 → 规范化 ─┬→（raw 录像）编码 → 分段录像
//                  └→ nvstreammux → 推理… → 跟踪 → 分析 → meta ─┬→（无画面输出）fakesink
//                                                             └→ OSD → 输出
// 单路：OSD → 编码 → tee → RTSP / 分段录像（共用一个编码器）
// 多路：OSD → tee ─┬→ 拼接 → 编码 → RTSP
//                 └→ 按源拆分 → 每路编码 → 分段录像

#ifndef EXAMPLE_APP_PLAN_H
#define EXAMPLE_APP_PLAN_H

#include <string>
#include <vector>

#include "config.h"
#include "dsf/gst/graph.h"

namespace example_app {

inline constexpr const char* kMetaTap = "meta";

struct RecorderSlot {
  int source = 0;      // sources 中的序号，即 mux pad 索引
  std::string branch;  // raw / osd
  std::string element; // splitmuxsink 名
};

struct Plan {
  std::vector<RecorderSlot> recorders;
  std::vector<std::string> encoders;  // 编码器名，供登记断流后的关键帧请求
};

Plan BuildGraph(const Config& config, dsf::gst::Graph& graph);

}  // namespace example_app

#endif  // EXAMPLE_APP_PLAN_H
