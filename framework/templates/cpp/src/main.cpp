// __NAME__ 入口：组装拓扑与组件，其余由框架处理。

#include <spdlog/spdlog.h>

#include "app.h"
#include "dsf/app/app.h"
#include "dsf/ds/meta_tap.h"

namespace {

struct App {
  using Config = __IDENT__::Config;

  static dsf::app::Info Describe() {
    return {"__NAME__", "TODO: 一句话说明这个应用",
            [] { return "deepstream=" + dsf::ds::DeepStreamVersion(); }};
  }

  static Config Load(dsf::config::Reader& root) { return __IDENT__::Load(root); }

  static void Build(const Config& config, dsf::app::Assembly& assembly) {
    __IDENT__::BuildGraph(config, assembly.graph());
    dsf::ds::SourceRegistry sources;
    for (const auto& source : config.sources) sources.Add(source.id);
    dsf::ds::MetaTap::Options options;
    options.element = __IDENT__::kMetaTap;
    options.width = config.mux.width;
    options.height = config.mux.height;
    auto& tap = assembly.Add<dsf::ds::MetaTap>(options, sources);
    // 业务逻辑从这里开始：BatchRecord 是纯数据，可以复制到其他线程处理。
    tap.OnBatch([](const dsf::ds::BatchRecord& batch) {
      for (const auto& frame : batch.frames) {
        if (!frame.objects.empty()) {
          spdlog::debug("{} frame {}: {} object(s)", frame.source_id, frame.frame_num, frame.objects.size());
        }
      }
    });
    // 就绪按批次元数据中的真实帧计数，而不是按 buffer（批次）计数。
    assembly.ReadyWhen([&tap] { return tap.frames(); });
  }
};

}  // namespace

int main(int argc, char** argv) { return dsf::app::Main<App>(argc, argv); }
