# DeepStream SDK 探测。SDK 可用时创建 dsf::deepstream 接口目标；否则在开发构建中跳过，
# 依赖 SDK 的目标（dsf_add_app 的应用、dsf_add_plugin 的插件、dsf_ds_meta）被跳过，
# 框架其余部分与测试照常构建。发布构建设置 DSF_REQUIRE_DEEPSTREAM=ON，缺 SDK 即失败。

set(DEEPSTREAM_ROOT "/opt/nvidia/deepstream/deepstream" CACHE PATH "DeepStream SDK 安装根目录")
option(DSF_REQUIRE_DEEPSTREAM "SDK 不可用时配置失败（发布构建使用）" OFF)

# 每次配置都重新判定：补装或切换 SDK 后无需清缓存。
foreach(_var DSF_NVDS_META_LIBRARY DSF_NVDSGST_META_LIBRARY DSF_DEEPSTREAM_LINKABLE)
  unset(${_var} CACHE)
  unset(${_var})
endforeach()

set(_dsf_ds_include "${DEEPSTREAM_ROOT}/sources/includes")
set(_dsf_ds_missing)
foreach(_header IN ITEMS nvdsmeta.h gstnvdsmeta.h nvdsinfer.h nvds_analytics_meta.h)
  if(NOT EXISTS "${_dsf_ds_include}/${_header}")
    list(APPEND _dsf_ds_missing "${_dsf_ds_include}/${_header}")
  endif()
endforeach()
find_library(DSF_NVDS_META_LIBRARY NAMES nvds_meta PATHS "${DEEPSTREAM_ROOT}/lib" NO_DEFAULT_PATH)
find_library(DSF_NVDSGST_META_LIBRARY NAMES nvdsgst_meta PATHS "${DEEPSTREAM_ROOT}/lib" NO_DEFAULT_PATH)
foreach(_lib DSF_NVDS_META_LIBRARY DSF_NVDSGST_META_LIBRARY)
  if(NOT ${_lib})
    list(APPEND _dsf_ds_missing "${DEEPSTREAM_ROOT}/lib/${_lib}")
  endif()
endforeach()

if(NOT TARGET PkgConfig::DSF_GSTREAMER_VIDEO)
  list(APPEND _dsf_ds_missing "gstreamer-video-1.0 开发包（libgstreamer-plugins-base1.0-dev，SDK 头文件需要）")
endif()

# 头文件存在不代表它的间接依赖可编译、库可被当前工具链链接（例如另一架构的 SDK）。
# 探针实际编译并链接一个程序，不运行它，也不访问 GPU。
if(NOT _dsf_ds_missing)
  include(CheckCXXSourceCompiles)
  include(CMakePushCheckState)
  cmake_push_check_state(RESET)
  set(CMAKE_TRY_COMPILE_TARGET_TYPE EXECUTABLE)
  set(CMAKE_REQUIRED_INCLUDES "${_dsf_ds_include}")
  set(CMAKE_REQUIRED_LIBRARIES
    "${DSF_NVDSGST_META_LIBRARY}" "${DSF_NVDS_META_LIBRARY}"
    PkgConfig::DSF_GSTREAMER PkgConfig::DSF_GSTREAMER_VIDEO)
  set(CMAKE_REQUIRED_LINK_OPTIONS "-Wl,-rpath,${DEEPSTREAM_ROOT}/lib")
  check_cxx_source_compiles([=[
    #include <string>  // DeepStream 7.1 的 nvds_analytics_meta.h 使用 std::string 却没有包含它
    #include <gstnvdsmeta.h>
    #include <nvds_analytics_meta.h>
    #include <nvdsinfer.h>
    int main() { return gst_buffer_get_nvds_batch_meta(nullptr) != nullptr; }
  ]=] DSF_DEEPSTREAM_LINKABLE)
  cmake_pop_check_state()
  if(NOT DSF_DEEPSTREAM_LINKABLE)
    list(APPEND _dsf_ds_missing "当前工具链无法编译 SDK 头文件或链接 SDK 库（平台、架构或依赖不匹配；"
                                "编译器输出见 CMakeFiles/CMakeConfigureLog.yaml 或 CMakeError.log）")
  endif()
endif()

if(_dsf_ds_missing)
  list(JOIN _dsf_ds_missing "\n  " _dsf_ds_text)
  if(DSF_REQUIRE_DEEPSTREAM)
    message(FATAL_ERROR "DeepStream SDK 不可用（DSF_REQUIRE_DEEPSTREAM=ON）：\n  ${_dsf_ds_text}")
  endif()
  set(DSF_HAVE_DEEPSTREAM OFF)
  return()
endif()

set(DSF_HAVE_DEEPSTREAM ON)
file(STRINGS "${DEEPSTREAM_ROOT}/version" _dsf_ds_version_line LIMIT_COUNT 1)
message(STATUS "DeepStream SDK: ${DEEPSTREAM_ROOT} (${_dsf_ds_version_line})")

add_library(dsf_deepstream INTERFACE)
add_library(dsf::deepstream ALIAS dsf_deepstream)
target_include_directories(dsf_deepstream SYSTEM INTERFACE "${_dsf_ds_include}")
target_link_libraries(dsf_deepstream INTERFACE
  "${DSF_NVDSGST_META_LIBRARY}" "${DSF_NVDS_META_LIBRARY}" PkgConfig::DSF_GSTREAMER_VIDEO)
# SDK 库目录不在系统搜索路径中；写入 rpath，运行环境无需额外配置。
target_link_options(dsf_deepstream INTERFACE "-Wl,-rpath,${DEEPSTREAM_ROOT}/lib")
