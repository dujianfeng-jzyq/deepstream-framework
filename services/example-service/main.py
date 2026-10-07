#!/usr/bin/env python3
"""example-service：消费 example-app 发布的录像事件与检测消息，演示 Python 服务单元的写法。

只读取 Redis Stream 并记录日志；真实服务在这里上传录像与图片、调用业务接口或做保留期清理。
检测记录的 image 字段是 JPEG 原始字节（payload 中的 image 描述其尺寸与大小），没有图片时不存在。
Stream 只保留很短的积压：处理慢的工作（上传、存盘）应放到单独线程，读取循环只负责取数据。
Redis 不可用时每秒重试，主循环始终推进，就绪状态反映的是服务本身而不是下游。
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass

from dsf.config import Reader, at_least, non_empty
from dsf.service import Service


@dataclass(frozen=True)
class Config:
    host: str
    port: int
    password: str
    stream: str
    detection_stream: str


def load(root: Reader) -> Config:
    redis = root.section("redis")
    return Config(
        host=redis.get("host", "127.0.0.1", non_empty),
        port=redis.get("port", 6379, at_least(1)),
        password=redis.secret("password"),
        stream=redis.required("stream", str, non_empty),
        detection_stream=redis.get("detection_stream", ""),
    )


def describe(event: dict, image) -> str:
    kind = event.get("event_type", "?")
    if kind == "detection":
        size = f"{len(image)} bytes" if image is not None else "none"
        return (f"detection {event.get('source_id')} frame={event.get('frame_num')} "
                f"objects={len(event.get('objects', []))} image={size}")
    if kind == "recording_finished":
        return (f"{event.get('source_id')} {event.get('branch')} {event.get('path')} "
                f"duration_ms={event.get('duration_ms')} detected={event.get('detected')}")
    return f"{kind} {event.get('source_id')} {event.get('segment_id')}"


def main(config: Config, service: Service) -> None:
    import redis  # 运行时依赖：随载荷安装（requirements.txt）

    client = redis.Redis(host=config.host, port=config.port, password=config.password or None,
                         socket_timeout=5, socket_connect_timeout=2)
    # 只处理启动之后的新事件。不能一直用 "$"：多个 Stream 时某次 XREAD 只返回其中一个，
    # 另一个仍为 "$"，在两次调用之间写入的记录会被跳过。连接后把它解析为当前最后一条的 ID。
    last_ids = {name: None for name in (config.stream, config.detection_stream) if name}
    failures = 0
    while service.running:
        service.alive()
        try:
            for name, last in last_ids.items():
                if last is None:
                    newest = client.xrevrange(name, count=1)
                    last_ids[name] = newest[0][0] if newest else b"0-0"
            batches = client.xread(last_ids, count=100, block=1000)
            failures = 0
        except redis.RedisError as error:
            if failures % 30 == 0:
                service.log.warning("redis %s:%d unavailable: %s", config.host, config.port, error)
            failures += 1
            service.sleep(1)
            continue
        for stream, entries in batches or []:
            name = stream.decode()
            for entry_id, fields in entries:
                last_ids[name] = entry_id
                try:
                    event = json.loads(fields[b"payload"])
                except (KeyError, ValueError):
                    service.log.warning("malformed entry %s in %s", entry_id, name)
                    continue
                service.log.info("event: %s", describe(event, fields.get(b"image")))


if __name__ == "__main__":
    sys.exit(Service("example-service", __doc__.splitlines()[0]).run(load, main))
