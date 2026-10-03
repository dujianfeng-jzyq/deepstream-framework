# example-app：配置与拓扑放在不依赖 SDK 的核心库中，开发机上也能测试；
# 可执行文件（组件装配）需要 SDK。

add_library(example_app_core STATIC src/config.cpp src/plan.cpp)
target_include_directories(example_app_core PUBLIC src)
target_link_libraries(example_app_core PUBLIC dsf::framework PRIVATE dsf_warnings)

dsf_add_app(example-app SOURCES src/main.cpp LIBRARIES example_app_core)

dsf_add_test(example_app_test
  SOURCES tests/test_example_app.cpp
  LIBRARIES example_app_core)
if(TARGET example_app_test)
  target_compile_definitions(example_app_test PRIVATE
    EXAMPLE_APP_TEMPLATE="${CMAKE_CURRENT_SOURCE_DIR}/templates/config/example-app.yaml")
endif()
