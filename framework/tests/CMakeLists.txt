# 框架测试。C++ 测试不需要 DeepStream；录像与 Redis 测试缺少软件插件或 redis-server 时 SKIP。

set(_cpp "${CMAKE_CURRENT_SOURCE_DIR}/cpp")

dsf_add_test(dsf_test_config SOURCES "${_cpp}/test_config.cpp" LIBRARIES dsf_config)
dsf_add_test(dsf_test_gst SOURCES "${_cpp}/test_gst.cpp" LIBRARIES dsf_gst TIMEOUT 60)
dsf_add_test(dsf_test_ds SOURCES "${_cpp}/test_ds.cpp" LIBRARIES dsf_ds)
dsf_add_test(dsf_test_record SOURCES "${_cpp}/test_record.cpp" LIBRARIES dsf_record TIMEOUT 120)
dsf_add_test(dsf_test_publish SOURCES "${_cpp}/test_publish.cpp"
             LIBRARIES dsf_publish PkgConfig::DSF_HIREDIS TIMEOUT 60)
dsf_add_test(dsf_test_app SOURCES "${_cpp}/test_app.cpp" LIBRARIES dsf::framework TIMEOUT 60)

# Python：部署工具、服务运行时、engine 工具、仓库约定。
file(GLOB _python_tests "${CMAKE_CURRENT_SOURCE_DIR}/python/test_*.py")
foreach(_test IN LISTS _python_tests)
  get_filename_component(_name "${_test}" NAME_WE)
  dsf_add_python_test(dsf_${_name} "${_test}" TIMEOUT 120)
endforeach()
