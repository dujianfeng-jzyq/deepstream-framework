# __NAME__：配置与拓扑在不依赖 SDK 的核心库中（开发机可测试），可执行文件需要 SDK。
add_library(__IDENT___core STATIC src/app.cpp)
target_include_directories(__IDENT___core PUBLIC src)
target_link_libraries(__IDENT___core PUBLIC dsf::framework PRIVATE dsf_warnings)

dsf_add_app(__NAME__ SOURCES src/main.cpp LIBRARIES __IDENT___core)

dsf_add_test(__IDENT___test SOURCES tests/test___IDENT__.cpp LIBRARIES __IDENT___core)
if(TARGET __IDENT___test)
  target_compile_definitions(__IDENT___test PRIVATE
    APP_TEMPLATE="${CMAKE_CURRENT_SOURCE_DIR}/templates/config/__NAME__.yaml")
endif()
