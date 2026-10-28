# example-app

框架的完整示例：多路视频 → 推理 → 跟踪 →（可选 analytics）→ Redis 消息 / 分段录像 / RTSP 输出。拓扑、配置与业务逻辑都是示例策略，派生项目可以整体替换。

- `src/config.cpp`：配置读取与跨字段规则（不依赖 SDK）
- `src/plan.cpp`：拓扑，形状见 `src/plan.h`（不依赖 SDK）
- `src/main.cpp`：元数据采集、录像器、发布通道与就绪指标的装配
- `templates/config/example-app.yaml`：配置模板，渲染到 `generated/config/example-app.yaml`；测试保证模板的每个字段都被读取
- `compose.yaml`：服务定义；GPU 等平台差异写在 `profiles/*.yaml` 的 `services.example-app`

现场参数：`CAMERA_ID`（也是 RTSP 输出路径 `<编号>-out`，只用字母、数字与 `_ . ~ - :`）、`RTSP_URL`、`RTSP_PASSWORD`、`REDIS_PASSWORD`、`MEDIAMTX_PUBLISH_PASSWORD`。

运行时行为：

- 就绪按批次元数据中的真实帧数计；帧计数 5 分钟不增长时退出，由容器重启恢复（`watchdog.stall_timeout_s`）；启动时相机未上线同样按此计时，不会每 30 秒重启。
- RTSP 输出断线时先排空再退出（`drain_on_error`），录像器封装当前分段，重启后恢复推流。
- 有目标的帧发检测消息，并给该路当前的录像分段打 `-detected` 标记；可选整帧 JPEG 随消息发送（`redis.image`）。
- 消息格式见 `docs/framework.md` 的“消息”一节，消费示例见 `services/example-service`。
