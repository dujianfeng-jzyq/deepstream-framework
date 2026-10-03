# __NAME__

由 `scripts/new-app.sh __NAME__ --kind python` 生成。第三方依赖写进 `requirements.txt`，
并在 `CMakeLists.txt` 的 `dsf_add_python_service` 中加 `REQUIREMENTS requirements.txt`。
