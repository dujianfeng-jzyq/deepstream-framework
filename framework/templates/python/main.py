#!/usr/bin/env python3
"""__NAME__：TODO 一句话说明这个服务。"""

from __future__ import annotations

import sys

from dsf.config import Reader, at_least
from dsf.service import Service


def load(root: Reader) -> float:
    return root.get("interval_s", 10.0, at_least(0.1))


def main(interval: float, service: Service) -> None:
    while service.sleep(interval):
        service.log.info("working")


if __name__ == "__main__":
    sys.exit(Service("__NAME__", __doc__.splitlines()[0]).run(load, main))
