# __NAME__

由 `scripts/new-app.sh __NAME__ --kind cpp` 生成。

- `src/app.cpp`：配置读取与拓扑（不依赖 SDK，`tests/` 中可测试）
- `src/main.cpp`：组件装配与业务逻辑入口
- `templates/config/__NAME__.yaml`：配置模板，渲染到 `generated/config/__NAME__.yaml`
- `compose.yaml`：服务定义；GPU 等平台差异写在 `profiles/*.yaml` 的 `services.__NAME__`

框架接口见 `docs/framework.md`。
