# example-service

Python 服务单元的示例：从 Redis Stream 读取 example-app 发布的录像事件与检测消息并记录日志。真实服务在这里上传录像与图片、调用业务接口。

- `main.py`：配置读取（`load`，无副作用）与主循环（`main`）；生命周期、信号、日志与就绪文件由 `dsf.service.Service` 处理
- `templates/config/example-service.yaml`：配置模板，Stream 名与 example-app 一致
- `requirements.txt`：第三方依赖（`redis`），构建时安装到载荷
- `tests/`：配置预检与（有 redis-server 时）端到端消费

行为与限制：

- 只处理启动之后写入的记录；Redis 不可用时每秒重试，主循环与就绪状态照常推进。
- 没有消费组与确认：重启期间写入的记录不会补处理。需要可靠消费时改用 `XREADGROUP` 并确认进度。
- 检测记录的 `image` 字段是 JPEG 原始字节；慢操作（上传、存盘）应放到读取循环之外。
