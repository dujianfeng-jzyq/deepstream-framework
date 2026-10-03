// YOLO parser 的解析规则测试（替身头文件，不需要 SDK）。

#include <cmath>
#include <limits>

#include "dsf_test.h"
#include "yolo_parser_abi.h"

namespace {

struct Fixture {
  std::vector<float> rows;
  NvDsInferNetworkInfo network{640, 640, 3};
  NvDsInferParseDetectionParams params;
  std::vector<NvDsInferParseObjectInfo> objects;

  Fixture() { params.perClassPreclusterThreshold = {0.25F, 0.5F}; }
  void Add(std::vector<float> row) { rows.insert(rows.end(), row.begin(), row.end()); }

  bool Parse(int width = 6, NvDsInferDataType type = FLOAT, std::size_t layers = 1) {
    NvDsInferLayerInfo layer{};
    layer.dataType = type;
    layer.inferDims.numDims = 2;
    layer.inferDims.d[0] = static_cast<int>(rows.size()) / 6;
    layer.inferDims.d[1] = width;
    layer.buffer = rows.data();
    std::vector<NvDsInferLayerInfo> outputs(layers, layer);
    return NvDsInferParseYolo(outputs, network, params, objects);
  }
};

}  // namespace

TEST(decodes_and_clamps_valid_candidates) {
  Fixture f;
  f.Add({-10, 20, 100, 700, 0.9F, 1});
  CHECK(f.Parse());
  CHECK_EQ(f.objects.size(), 1u);
  CHECK_EQ(f.objects[0].classId, 1u);
  CHECK_EQ(f.objects[0].left, 0.0F);
  CHECK_EQ(f.objects[0].height, 620.0F);  // 裁剪到网络高度
  CHECK_EQ(f.objects[0].detectionConfidence, 0.9F);
}

TEST(filters_by_per_class_threshold) {
  Fixture f;
  f.Add({0, 0, 10, 10, 0.3F, 0});  // 类别 0 阈值 0.25：保留
  f.Add({0, 0, 10, 10, 0.3F, 1});  // 类别 1 阈值 0.5：丢弃
  CHECK(f.Parse());
  CHECK_EQ(f.objects.size(), 1u);
}

TEST(rejects_invalid_classes_and_values) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  Fixture f;
  f.Add({0, 0, 10, 10, 0.9F, 2});     // 超出阈值表
  f.Add({0, 0, 10, 10, 0.9F, -1});    // 负数
  f.Add({0, 0, 10, 10, 0.9F, 0.5F});  // 非整数
  f.Add({0, 0, 10, 10, 0.9F, 1e20F});
  f.Add({0, 0, 10, 10, 0.9F, nan});
  f.Add({0, 0, 10, 10, nan, 0});
  f.Add({0, inf, 10, 10, 0.9F, 0});
  f.Add({5, 5, 5.5F, 50, 0.9F, 0});   // 宽度不足 1 像素
  f.Add({700, 0, 800, 10, 0.9F, 0});  // 完全在画面外
  CHECK(f.Parse());
  CHECK(f.objects.empty());
}

TEST(rejects_unexpected_tensors) {
  Fixture f;
  f.Add({0, 0, 10, 10, 0.9F, 0});
  CHECK(!f.Parse(7));
  CHECK(!f.Parse(6, HALF));
  CHECK(!f.Parse(6, FLOAT, 2));
  f.params.perClassPreclusterThreshold.clear();
  CHECK(!f.Parse());
  Fixture empty_network;
  empty_network.Add({0, 0, 10, 10, 0.9F, 0});
  empty_network.network.width = 0;
  CHECK(!empty_network.Parse());
}

TEST(replaces_previous_results) {
  Fixture f;
  f.Add({0, 0, 10, 10, 0.1F, 0});  // 低于阈值
  f.objects.resize(3);
  CHECK(f.Parse());
  CHECK(f.objects.empty());
}

DSF_TEST_MAIN()
