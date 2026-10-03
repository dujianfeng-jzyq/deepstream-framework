#!/usr/bin/env python3
"""测试用 trtexec 替身：记录调用，按环境变量模拟版本、失败与空输出。"""
import os
import sys

if "--version" in sys.argv:
    print(f"&&&& RUNNING TensorRT.trtexec [TensorRT v{os.environ.get('FAKE_TRT_VERSION', '100900')}]")
    sys.exit(1)  # 真实 trtexec 在没有 GPU 时也会先打印版本再失败
with open(os.environ["FAKE_TRTEXEC_LOG"], "a", encoding="utf-8") as log:
    log.write(" ".join(sys.argv[1:]) + "\n")
if os.environ.get("FAKE_TRTEXEC_FAIL"):
    sys.exit(3)
output = next(arg.split("=", 1)[1] for arg in sys.argv if arg.startswith("--saveEngine="))
with open(output, "w", encoding="utf-8") as engine:
    if not os.environ.get("FAKE_TRTEXEC_EMPTY"):
        engine.write("engine " + " ".join(sys.argv[1:]))
