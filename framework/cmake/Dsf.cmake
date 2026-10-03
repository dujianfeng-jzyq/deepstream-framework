# 框架入口：在项目根 CMakeLists.txt 中 include。依赖目标是目录作用域的，
# 因此依赖探测必须在根目录执行，插件与服务单元才能看到它们。

include_guard(GLOBAL)

get_filename_component(DSF_FRAMEWORK_DIR "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
file(STRINGS "${DSF_FRAMEWORK_DIR}/VERSION" DSF_VERSION LIMIT_COUNT 1)

include("${CMAKE_CURRENT_LIST_DIR}/DsfDependencies.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/DsfDeepStream.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/DsfUnits.cmake")

# Python 测试与 CLI 冒烟测试使用；找不到时只跳过这些测试。
# 优先使用 PATH 上的 python3（装有 PyYAML 的那个），而不是 CMake 找到的最新版本。
set(Python3_FIND_UNVERSIONED_NAMES FIRST)
set(Python3_FIND_FRAMEWORK LAST)
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
  set(DSF_PYTHON "${Python3_EXECUTABLE}")
else()
  set(DSF_PYTHON "")
  message(WARNING "python3 not found: Python tests are skipped")
endif()
