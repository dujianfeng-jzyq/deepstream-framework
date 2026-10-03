// YOLO 检测输出解析：nvinfer 按模型配置的 parse-bbox-func-name / custom-lib-path 调用。
//
// 输入约定（模型导出时决定）：唯一输出层为 FP32 张量 [候选数, 6]，
// 每行 x1, y1, x2, y2, confidence, class_id，坐标为网络输入尺寸下的像素。
// 本函数只做解码与过滤；NMS/聚类由 nvinfer 配置（cluster-mode）负责。
//
// SPDX-License-Identifier: MIT
// 输出布局与接口沿用 marcoslucianops/DeepStream-Yolo（revision 2894babc）的约定，
// 版权与许可见 LICENSE.md。

#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>

#include "yolo_parser_abi.h"

namespace {

constexpr int kCandidateWidth = 6;

struct Candidate {
  float x1, y1, x2, y2, confidence, class_value;
};

// 类别值必须是阈值表范围内的非负整数；浮点值先检查范围再转换，避免未定义行为。
bool ClassIndex(float value, std::size_t classes, int* index) {
  if (!std::isfinite(value) || value < 0.0F || static_cast<double>(value) >= static_cast<double>(classes)) {
    return false;
  }
  const int truncated = static_cast<int>(value);
  if (static_cast<float>(truncated) != value) return false;
  *index = truncated;
  return true;
}

bool Decode(const Candidate& c, float width, float height,
            const std::vector<float>& thresholds, NvDsInferParseObjectInfo* out) {
  int class_id = 0;
  if (!ClassIndex(c.class_value, thresholds.size(), &class_id)) return false;
  if (!std::isfinite(c.confidence) || c.confidence < thresholds[static_cast<std::size_t>(class_id)]) {
    return false;
  }
  for (float v : {c.x1, c.y1, c.x2, c.y2}) {
    if (!std::isfinite(v)) return false;
  }
  const float left = std::fmin(std::fmax(c.x1, 0.0F), width);
  const float top = std::fmin(std::fmax(c.y1, 0.0F), height);
  const float right = std::fmin(std::fmax(c.x2, 0.0F), width);
  const float bottom = std::fmin(std::fmax(c.y2, 0.0F), height);
  // 裁剪后不足一个像素的框没有意义，丢弃。
  if (right - left < 1.0F || bottom - top < 1.0F) return false;
  *out = NvDsInferParseObjectInfo{};
  out->classId = static_cast<unsigned int>(class_id);
  out->left = left;
  out->top = top;
  out->width = right - left;
  out->height = bottom - top;
  out->detectionConfidence = c.confidence;
  return true;
}

}  // namespace

extern "C" bool NvDsInferParseYolo(const std::vector<NvDsInferLayerInfo>& output_layers,
                                   const NvDsInferNetworkInfo& network_info,
                                   const NvDsInferParseDetectionParams& detection_params,
                                   std::vector<NvDsInferParseObjectInfo>& object_list) {
  if (output_layers.size() != 1) {
    std::cerr << "yolo parser: expected exactly one output layer, got " << output_layers.size() << "\n";
    return false;
  }
  const NvDsInferLayerInfo& layer = output_layers.front();
  const NvDsInferDims& dims = layer.inferDims;
  if (layer.dataType != FLOAT || layer.buffer == nullptr || dims.numDims != 2 || dims.d[0] <= 0 ||
      dims.d[1] != kCandidateWidth) {
    std::cerr << "yolo parser: expected one FP32 [candidates, 6] output tensor\n";
    return false;
  }
  const std::vector<float>& thresholds = detection_params.perClassPreclusterThreshold;
  if (network_info.width == 0 || network_info.height == 0 || thresholds.empty()) {
    std::cerr << "yolo parser: invalid network size or empty class thresholds\n";
    return false;
  }

  const auto* candidates = static_cast<const Candidate*>(layer.buffer);
  const auto count = static_cast<std::size_t>(dims.d[0]);
  const auto width = static_cast<float>(network_info.width);
  const auto height = static_cast<float>(network_info.height);
  std::vector<NvDsInferParseObjectInfo> objects;
  objects.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    NvDsInferParseObjectInfo object;
    if (Decode(candidates[i], width, height, thresholds, &object)) objects.push_back(object);
  }
  object_list = std::move(objects);
  return true;
}

static_assert(sizeof(Candidate) == kCandidateWidth * sizeof(float), "candidate rows are packed floats");
