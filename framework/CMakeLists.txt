# 框架目标。依赖只能向下：config ← gst ← ds ← record；publish 独立；app 聚合。
# dsf_ds_meta 需要 DeepStream SDK，仅在 SDK 可用时创建。

set(_inc "${CMAKE_CURRENT_SOURCE_DIR}/cpp/include")
set(_src "${CMAKE_CURRENT_SOURCE_DIR}/cpp/src")

add_library(dsf_config STATIC "${_src}/config/config.cpp")
target_include_directories(dsf_config PUBLIC "${_inc}")
target_link_libraries(dsf_config PRIVATE PkgConfig::DSF_YAML_CPP dsf_warnings)

add_library(dsf_gst STATIC
  "${_src}/gst/introspect.cpp"
  "${_src}/gst/graph.cpp"
  "${_src}/gst/pipeline.cpp")
target_include_directories(dsf_gst PUBLIC "${_inc}")
target_link_libraries(dsf_gst
  PUBLIC dsf_config PkgConfig::DSF_GSTREAMER
  PRIVATE dsf_logging dsf_warnings)

add_library(dsf_ds STATIC
  "${_src}/ds/blocks.cpp"
  "${_src}/ds/message.cpp")
target_link_libraries(dsf_ds
  PUBLIC dsf_gst nlohmann_json::nlohmann_json
  PRIVATE dsf_warnings)

add_library(dsf_record STATIC "${_src}/record/segment_recorder.cpp")
target_link_libraries(dsf_record PUBLIC dsf_ds PRIVATE dsf_logging dsf_warnings)

add_library(dsf_publish STATIC "${_src}/publish/redis_publisher.cpp")
target_include_directories(dsf_publish PUBLIC "${_inc}")
target_link_libraries(dsf_publish
  PUBLIC dsf_config
  PRIVATE PkgConfig::DSF_HIREDIS dsf_logging dsf_warnings)

add_library(dsf_app STATIC
  "${_src}/app/app.cpp"
  "${_src}/app/components.cpp")
target_link_libraries(dsf_app
  PUBLIC dsf_gst dsf_publish dsf_logging
  PRIVATE dsf_warnings)
target_compile_definitions(dsf_app PRIVATE DSF_VERSION="${DSF_VERSION}")

add_library(dsf_framework INTERFACE)
add_library(dsf::framework ALIAS dsf_framework)
target_link_libraries(dsf_framework INTERFACE dsf_app dsf_ds dsf_record dsf_publish dsf_gst dsf_config)

if(DSF_HAVE_DEEPSTREAM)
  add_library(dsf_ds_meta STATIC "${_src}/ds/meta_tap.cpp")
  add_library(dsf::ds_meta ALIAS dsf_ds_meta)
  target_link_libraries(dsf_ds_meta
    PUBLIC dsf_ds dsf::deepstream
    PRIVATE "${DSF_NVDS_JPEGENC_LIBRARY}" dsf_logging dsf_warnings)
  target_compile_definitions(dsf_ds_meta PRIVATE
    DSF_DEEPSTREAM_VERSION_FILE="${DEEPSTREAM_ROOT}/version")
endif()

# ---- 载荷：Python 运行时与工具 ----
install(DIRECTORY python/dsf DESTINATION python
        FILE_PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
        PATTERN __pycache__ EXCLUDE)
install(FILES tools/prepare_model_engines.py DESTINATION tools
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE
                    WORLD_READ WORLD_EXECUTE)

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
