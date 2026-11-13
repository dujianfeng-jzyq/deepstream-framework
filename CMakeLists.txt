# 项目构建入口：先加载框架，再自动发现插件与服务单元。
# 新增单元只需在 plugins/<name>/ 或 services/<name>/ 放一个 CMakeLists.txt。

cmake_minimum_required(VERSION 3.22)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "Generate compile_commands.json for editors")

# 固定的构建工程名，不是项目身份：项目名只写在 project.env 中（apps 镜像在编译之后才复制它）。
project(dsf LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

include(CTest)

include(framework/cmake/Dsf.cmake)
add_subdirectory(framework)

# 插件先于服务：应用可能在测试里引用插件目标。
dsf_add_unit_directories("${CMAKE_SOURCE_DIR}/plugins")
dsf_add_unit_directories("${CMAKE_SOURCE_DIR}/services")

dsf_finalize()
