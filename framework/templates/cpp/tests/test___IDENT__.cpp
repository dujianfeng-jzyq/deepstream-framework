// __NAME__ 的配置与拓扑测试（不需要 SDK）。

#include <fstream>
#include <regex>
#include <sstream>

#include "app.h"
#include "dsf/app/app.h"
#include "dsf_test.h"

namespace {

std::string Rendered() {
  std::ifstream stream(APP_TEMPLATE);
  std::stringstream buffer;
  buffer << stream.rdbuf();
  return std::regex_replace(buffer.str(), std::regex(R"(\$\{[A-Z_]+\})"), "x");
}

}  // namespace

TEST(template_loads_and_builds) {
  const auto document = dsf::config::Document::LoadText(Rendered(), "__NAME__.yaml");
  dsf::config::Reader root = document.Root();
  (void)dsf::app::FrameworkOptions::Read(root);  // 日志、看门狗与就绪段由框架读取
  const auto config = __IDENT__::Load(root);
  root.Finish();
  dsf::gst::Graph graph;
  __IDENT__::BuildGraph(config, graph);
  CHECK(graph.Describe().find("meta -> sink") != std::string::npos);
}

DSF_TEST_MAIN()
