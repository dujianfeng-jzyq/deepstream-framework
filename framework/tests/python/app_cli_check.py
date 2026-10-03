#!/usr/bin/env python3
"""每个 dsf_add_app 应用都登记的 CLI 冒烟测试：入口约定（退出码与诊断模式）对所有应用一致。"""

import subprocess
import sys

app = sys.argv[1]


def run(*args):
    return subprocess.run([app, *args], capture_output=True, text=True, timeout=60)


checks = [
    (("--help",), 0, "--check-config"),
    (("--version",), 0, "framework="),
    (("--bogus",), 2, "unknown argument"),
    (("--check-config", "--dump-pipeline"), 2, "conflicting option"),
    (("--check-config", "--config", "/nonexistent/app.yaml"), 1, "cannot open config file"),
    (("--selftest",), 0, ""),
]
failed = 0
for args, code, fragment in checks:
    result = run(*args)
    output = result.stdout + result.stderr
    if result.returncode != code or fragment not in output:
        print(f"FAIL {' '.join(args)}: exit {result.returncode} (want {code})\n{output}")
        failed += 1
    else:
        print(f"ok   {' '.join(args)}")
sys.exit(1 if failed else 0)
