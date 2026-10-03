# yolo-parser

`libnvdsinfer_custom_yolo.so`：nvinfer 的自定义检测解析库，导出 `NvDsInferParseYolo`。模型的 nvinfer 配置通过 `parse-bbox-func-name=NvDsInferParseYolo` 与 `custom-lib-path=/workspace/lib/libnvdsinfer_custom_yolo.so` 引用它。

## 输入约定

唯一输出层为 FP32 张量 `[候选数, 6]`，每行 `x1, y1, x2, y2, confidence, class_id`，坐标是网络输入尺寸下的像素（Ultralytics 导出时 `nms=False`、输出布局 `xyxy-confidence-class_id`）。不支持其他 YOLO 输出布局。

## 行为

- 输出层数量、数据类型、维度不符，网络尺寸为 0 或类别阈值表为空时返回失败。
- 类别必须是落在阈值表范围内的非负整数；置信度必须达到该类别的 `pre-cluster-threshold`。
- 坐标必须有限，裁剪到网络尺寸后宽或高不足 1 像素的候选丢弃。
- 只做解码与过滤；NMS/聚类由 nvinfer 配置（`cluster-mode`）完成，engine 由 `model-engine-builder` 预构建。

## 构建与测试

| 目标 | 说明 |
|---|---|
| `yolo_parser` | SDK 可用时构建，安装到载荷 `lib/`，本说明与许可证安装到 `share/yolo_parser/` |
| `yolo_parser_test` | 用 `tests/stubs/nvdsinfer.h` 替身头文件编译同一份源码，开发机上验证解析规则 |

`yolo_parser_abi.h` 按 SDK 布局声明检测参数结构，避免引入 TensorRT/CUDA 头文件；更换 SDK 大版本时须对照 `nvdsinfer_custom_impl.h` 核对，并在目标机验证加载与推理结果。

## 来源与许可

输出布局与解析接口沿用 [marcoslucianops/DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo)（revision `2894babce8e75c49115dbe0c7b516289ed853565`）的约定，按 MIT 许可证发布，见 [LICENSE.md](LICENSE.md)。
