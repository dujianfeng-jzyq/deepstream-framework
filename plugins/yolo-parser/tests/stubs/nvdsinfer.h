// 测试替身：只声明 parser 用到的 nvdsinfer.h 类型，字段名与 SDK 一致。
// 让解析源码在没有 SDK 的开发机上编译与测试；不随载荷发布，也不能证明 SDK ABI 兼容。

#ifndef NVDSINFER_STUB_H
#define NVDSINFER_STUB_H

#define NVDSINFER_MAX_DIMS 8

typedef enum { FLOAT = 0, HALF = 1, INT8 = 2, INT32 = 3 } NvDsInferDataType;

typedef struct {
  unsigned int numDims;
  int d[NVDSINFER_MAX_DIMS];
  unsigned int numElements;
} NvDsInferDims;

typedef struct {
  NvDsInferDataType dataType;
  union {
    NvDsInferDims inferDims;
    NvDsInferDims dims;
  };
  int bindingIndex;
  const char* layerName;
  void* buffer;
  int isInput;
} NvDsInferLayerInfo;

typedef struct {
  unsigned int width;
  unsigned int height;
  unsigned int channels;
} NvDsInferNetworkInfo;

typedef struct {
  unsigned int classId;
  float left;
  float top;
  float width;
  float height;
  float detectionConfidence;
} NvDsInferObjectDetectionInfo;

typedef NvDsInferObjectDetectionInfo NvDsInferParseObjectInfo;

#endif  // NVDSINFER_STUB_H
