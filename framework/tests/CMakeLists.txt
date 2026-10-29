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
# 模板单元：用 new-app 的同一实现把 framework/templates/<kind>/ 生成到构建目录，照常编译并运行其测试
# （C++ 可执行文件需要 SDK，在 builder 中编译），新应用的起点因此总是可用的。dsf 前缀是框架保留的，
# 不会与项目单元重名；DSF_INSTALL 为 OFF，它们不进入载荷。模板改动后重新配置时重新生成。
if(DSF_PYTHON)
  set(DSF_INSTALL OFF)
  file(GLOB_RECURSE _templates CONFIGURE_DEPENDS "${DSF_FRAMEWORK_DIR}/templates/*")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
               ${_templates} "${DSF_FRAMEWORK_DIR}/deploy/dsf_deploy/project.py")
  foreach(_kind IN ITEMS cpp python)
    set(_unit "${CMAKE_CURRENT_BINARY_DIR}/template-units/dsf-template-${_kind}")
    execute_process(
      COMMAND "${CMAKE_COMMAND}" -E env "PYTHONPATH=${DSF_FRAMEWORK_DIR}/deploy"
              "${DSF_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/python/generate_template_unit.py"
              ${_kind} "dsf-template-${_kind}" "${_unit}"
      RESULT_VARIABLE _result)
    if(NOT _result EQUAL 0)
      message(FATAL_ERROR "cannot generate the ${_kind} template unit")
    endif()
    add_subdirectory("${_unit}" "${CMAKE_CURRENT_BINARY_DIR}/template-units-build/${_kind}")
  endforeach()
endif()

# 依赖 profiles/ 与项目输入；builder 在编译之后才复制它们，按此标签单独运行。
if(TEST dsf_test_profiles)
  set_tests_properties(dsf_test_profiles PROPERTIES LABELS profiles)
endif()
