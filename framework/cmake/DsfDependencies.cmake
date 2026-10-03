# 框架的第三方依赖。头文件与链接参数取自同一份包元数据，避免混用 SDK 自带库与系统开发包。
# spdlog/fmt 使用头文件模式：运行环境不需要它们的动态库。

find_package(PkgConfig REQUIRED)
pkg_check_modules(DSF_GSTREAMER REQUIRED IMPORTED_TARGET gstreamer-1.0>=1.20)
# DeepStream 的 gstnvdsmeta.h 包含 gst/video/video.h；SDK 不可用时不强制要求。
pkg_check_modules(DSF_GSTREAMER_VIDEO IMPORTED_TARGET gstreamer-video-1.0)
pkg_check_modules(DSF_YAML_CPP REQUIRED IMPORTED_TARGET yaml-cpp)
pkg_check_modules(DSF_HIREDIS REQUIRED IMPORTED_TARGET hiredis)

# Ubuntu 22.04 的 nlohmann-json 是 3.10，这里只要求 3.9 起提供的接口。
find_package(nlohmann_json 3.9 CONFIG REQUIRED)

find_path(DSF_SPDLOG_INCLUDE_DIR NAMES spdlog/spdlog.h)
find_path(DSF_FMT_INCLUDE_DIR NAMES fmt/format.h)
if(NOT DSF_SPDLOG_INCLUDE_DIR OR NOT DSF_FMT_INCLUDE_DIR)
  message(FATAL_ERROR "需要 spdlog 与 fmt 的头文件（libspdlog-dev、libfmt-dev）")
endif()

add_library(dsf_logging INTERFACE)
target_include_directories(dsf_logging SYSTEM INTERFACE
  "${DSF_SPDLOG_INCLUDE_DIR}" "${DSF_FMT_INCLUDE_DIR}")
# SPDLOG_FMT_EXTERNAL 与 FMT_HEADER_ONLY 必须成对出现，否则会链接到动态 fmt。
target_compile_definitions(dsf_logging INTERFACE
  SPDLOG_HEADER_ONLY SPDLOG_FMT_EXTERNAL FMT_HEADER_ONLY)

# 警告开关只定义一次，框架、单元与测试共用。
add_library(dsf_warnings INTERFACE)
target_compile_options(dsf_warnings INTERFACE -Wall -Wextra -Wpedantic)
