// nvinfer 自定义检测解析的接口声明。
//
// SDK 的 nvdsinfer_custom_impl.h 会连带引入 TensorRT、CUDA 与 ONNX 解析器头文件；
// 只解析输出张量的插件用不到它们，这里按 SDK 的布局（DeepStream 7.x/8.x 相同）
// 声明检测参数结构。更换 SDK 大版本时须对照 nvdsinfer_custom_impl.h 核对。

#ifndef YOLO_PARSER_ABI_H
#define YOLO_PARSER_ABI_H

#include <nvdsinfer.h>

#include <vector>

struct NvDsInferParseDetectionParams {
  unsigned int numClassesConfigured = 0;
  std::vector<float> perClassPreclusterThreshold;
  std::vector<float> perClassPostclusterThreshold;
  std::vector<float>& perClassThreshold = perClassPreclusterThreshold;
};

extern "C" bool NvDsInferParseYolo(const std::vector<NvDsInferLayerInfo>& output_layers,
                                   const NvDsInferNetworkInfo& network_info,
                                   const NvDsInferParseDetectionParams& detection_params,
                                   std::vector<NvDsInferParseObjectInfo>& object_list);

#endif  // YOLO_PARSER_ABI_H
