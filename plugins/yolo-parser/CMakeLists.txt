# YOLO 检测解析插件。库名由模型配置的 custom-lib-path 决定，改名须同步修改 nvinfer 配置。

dsf_add_plugin(yolo_parser
  SOURCES nvdsinfer_custom_yolo.cpp
  OUTPUT_NAME nvdsinfer_custom_yolo
  FILES LICENSE.md README.md)

# 用替身头文件编译同一份源码，开发机上也能验证解析规则。
dsf_add_test(yolo_parser_test
  SOURCES tests/test_yolo_parser.cpp nvdsinfer_custom_yolo.cpp
  INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/tests/stubs" "${CMAKE_CURRENT_SOURCE_DIR}")
