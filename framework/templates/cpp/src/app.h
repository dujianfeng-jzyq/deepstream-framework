// __NAME__ 的配置与拓扑（不依赖 SDK）。

#ifndef __IDENT___APP_H
#define __IDENT___APP_H

#include <vector>

#include "dsf/config/config.h"
#include "dsf/ds/blocks.h"
#include "dsf/gst/graph.h"

namespace __IDENT__ {

inline constexpr const char* kMetaTap = "meta";

struct Config {
  std::vector<dsf::ds::SourceSpec> sources;
  dsf::ds::MuxSpec mux;
  std::vector<dsf::ds::InferSpec> inference;
};

Config Load(dsf::config::Reader& root);

// sources → mux → inference… → meta → fakesink。在这里加入跟踪、分析与输出分支。
void BuildGraph(const Config& config, dsf::gst::Graph& graph);

}  // namespace __IDENT__

#endif  // __IDENT___APP_H
