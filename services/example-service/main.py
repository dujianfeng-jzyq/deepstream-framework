#!/usr/bin/env python3
"""example-service：消费 example-app 发布的录像事件，演示 Python 服务单元的写法。

只读取 Redis Stream 并记录日志；真实服务在这里上传录像、调用业务接口或做保留期清理。
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


def load(root: Reader) -> Config:
    redis = root.section("redis")
    return Config(
        host=redis.get("host", "127.0.0.1", non_empty),
        port=redis.get("port", 6379, at_least(1)),
        password=redis.secret("password"),
        stream=redis.required("stream", str, non_empty),
    )


def describe(event: dict) -> str:
    kind = event.get("event_type", "?")
    if kind == "recording_finished":
        return (f"{event.get('source_id')} {event.get('branch')} {event.get('path')} "
                f"duration_ms={event.get('duration_ms')} detected={event.get('detected')}")
    return f"{kind} {event.get('source_id')} {event.get('segment_id')}"


def main(config: Config, service: Service) -> None:
    import redis  # 运行时依赖：随载荷安装（requirements.txt）

    client = redis.Redis(host=config.host, port=config.port, password=config.password or None,
                         socket_timeout=5, socket_connect_timeout=2)
    last_id = "$"  # 只处理启动之后的新事件
    failures = 0
    while service.running:
        service.alive()
        try:
            batches = client.xread({config.stream: last_id}, count=100, block=1000)
            failures = 0
        except redis.RedisError as error:
            if failures % 30 == 0:
                service.log.warning("redis %s:%d unavailable: %s", config.host, config.port, error)
            failures += 1
            service.sleep(1)
            continue
        for _stream, entries in batches or []:
            for entry_id, fields in entries:
                last_id = entry_id
                try:
                    event = json.loads(fields[b"payload"])
                except (KeyError, ValueError):
                    service.log.warning("malformed entry %s in %s", entry_id, config.stream)
                    continue
                service.log.info("event: %s", describe(event))


if __name__ == "__main__":
    sys.exit(Service("example-service", __doc__.splitlines()[0]).run(load, main))
