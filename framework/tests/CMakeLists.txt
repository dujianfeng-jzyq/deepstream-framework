# 框架测试。C++ 测试不需要 DeepStream；录像与 Redis 测试缺少软件插件或 redis-server 时 SKIP。

set(_cpp "${CMAKE_CURRENT_SOURCE_DIR}/cpp")

dsf_add_test(dsf_test_config SOURCES "${_cpp}/test_config.cpp" LIBRARIES dsf_config)
dsf_add_test(dsf_test_gst SOURCES "${_cpp}/test_gst.cpp" LIBRARIES dsf_gst TIMEOUT 60)
dsf_add_test(dsf_test_ds SOURCES "${_cpp}/test_ds.cpp" LIBRARIES dsf_ds)
dsf_add_test(dsf_test_record SOURCES "${_cpp}/test_record.cpp" LIBRARIES dsf_record dsf_app TIMEOUT 120)
dsf_add_test(dsf_test_publish SOURCES "${_cpp}/test_publish.cpp"
             LIBRARIES dsf_publish PkgConfig::DSF_HIREDIS TIMEOUT 60)
dsf_add_test(dsf_test_app SOURCES "${_cpp}/test_app.cpp" LIBRARIES dsf::framework TIMEOUT 60)
if(TARGET dsf_test_app)
  target_compile_definitions(dsf_test_app PRIVATE
    DSF_FRAMEWORK_SECTIONS="${CMAKE_CURRENT_SOURCE_DIR}/framework_sections.txt")
endif()

# Python：部署工具、服务运行时、engine 工具、仓库约定。
file(GLOB _python_tests "${CMAKE_CURRENT_SOURCE_DIR}/python/test_*.py")
foreach(_test IN LISTS _python_tests)
  get_filename_component(_name "${_test}" NAME_WE)
  dsf_add_python_test(dsf_${_name} "${_test}" TIMEOUT 120)
endforeach()
# 依赖 profiles/ 与项目输入；builder 在编译之后才复制它们，按此标签单独运行。
if(TEST dsf_test_profiles)
  set_tests_properties(dsf_test_profiles PROPERTIES LABELS profiles)
endif()
