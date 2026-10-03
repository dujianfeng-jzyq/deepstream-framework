// __NAME__ 的配置与拓扑实现。

#include "app.h"

namespace __IDENT__ {

Config Load(dsf::config::Reader& root) {
  Config config;
  config.sources = dsf::ds::ReadSources(root);
  config.mux = dsf::ds::MuxSpec::Read(root.Section("mux"));
  config.inference = dsf::ds::ReadInferences(root);
  return config;
}

void BuildGraph(const Config& config, dsf::gst::Graph& graph) {
  bool live = false;
  for (const auto& source : config.sources) live = live || source.live;
  auto mux = dsf::ds::AddMux(graph, config.mux, static_cast<int>(config.sources.size()), live);
  for (std::size_t i = 0; i < config.sources.size(); ++i) {
    auto source = dsf::ds::AddSource(graph, config.sources[i], "src" + std::to_string(i));
    dsf::ds::LinkToMux(graph, source, mux, static_cast<int>(i));
  }
  dsf::gst::Element tail = mux;
  for (const auto& infer : config.inference) tail = dsf::ds::AddInfer(graph, tail, infer);
  tail = dsf::ds::AddMetaTap(graph, tail, kMetaTap);
  dsf::ds::AddFakeSink(graph, tail, "sink");
}

}  // namespace __IDENT__
