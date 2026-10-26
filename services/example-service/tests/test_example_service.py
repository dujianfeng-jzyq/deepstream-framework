"""example-service：配置读取、预检，以及（有 redis-server 与 redis-py 时）端到端消费。"""

import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERVICE = HERE.parent / "main.py"
TEMPLATE = HERE.parent / "templates" / "config" / "example-service.yaml"


def render(variables: dict) -> str:
    return re.sub(r"\$\{([A-Z_]+)\}", lambda m: variables.get(m.group(1), "x"), TEMPLATE.read_text())


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class ExampleServiceTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="example-service-"))
        self.env = dict(os.environ, DSF_READY_FILE=str(self.dir / "ready"),
                        CONTAINER_CONFIG_PATH=str(self.dir))

    def tearDown(self):
        shutil.rmtree(self.dir)

    def write_config(self, text: str) -> None:
        (self.dir / "example-service.yaml").write_text(text)

    def run_service(self, *args):
        return subprocess.run([sys.executable, str(SERVICE), *args], env=self.env,
                              capture_output=True, text=True, timeout=30)

    def test_template_passes_check_config_without_side_effects(self):
        self.write_config(render({"PROJECT_NAME": "p"}))
        result = self.run_service("--check-config")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("configuration is valid", result.stdout)
        self.assertFalse((self.dir / "ready").exists())

    def test_unknown_or_invalid_fields_fail(self):
        self.write_config(render({}) + "extra: 1\n")
        result = self.run_service("--check-config")
        self.assertEqual(result.returncode, 1)
        self.assertIn("extra: unknown field", result.stderr)
        self.write_config(render({}).replace("port: 6379", "port: \"6379\""))
        self.assertIn("redis.port: expected int", self.run_service("--check-config").stderr)

    def test_usage_errors_exit_2(self):
        self.assertEqual(self.run_service("--bogus").returncode, 2)

    def test_consumes_recording_events(self):
        server = shutil.which("redis-server")
        try:
            import redis
        except ImportError:
            redis = None
        if server is None or redis is None:
            self.skipTest("redis-server or redis-py not available")
        port = free_port()
        redis_process = subprocess.Popen([server, "--port", str(port), "--save", "", "--appendonly", "no"],
                                         stdout=subprocess.DEVNULL)
        service = None
        try:
            client = redis.Redis(port=port)
            for _ in range(100):
                try:
                    client.ping()
                    break
                except redis.ConnectionError:
                    time.sleep(0.05)
            self.write_config(render({"PROJECT_NAME": "p"}).replace("host: redis", "host: 127.0.0.1")
                              .replace("port: 6379", f"port: {port}").replace("password: x", "password: \"\""))
            service = subprocess.Popen([sys.executable, str(SERVICE)], env=self.env,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            lines = []
            reader = threading.Thread(target=lambda: [lines.append(line) for line in service.stdout], daemon=True)
            reader.start()
            expected = ("event: cam osd /data/a.mp4 duration_ms=5",
                        "event: detection cam frame=3 objects=1 image=5 bytes")
            seen = lambda: all(any(text in line for line in list(lines)) for text in expected)
            # 服务只处理启动后写入的记录，而“启动”的时刻从外面看不到：重复写入直到两条都被处理，
            # 不靠固定等待。
            deadline = time.monotonic() + 20
            while not seen() and time.monotonic() < deadline and service.poll() is None:
                client.xadd("p:recordings", {"payload": '{"event_type":"recording_finished","source_id":"cam",'
                                                        '"branch":"osd","path":"/data/a.mp4","duration_ms":5}'})
                client.xadd("p:detections", {"payload": '{"event_type":"detection","source_id":"cam",'
                                                        '"frame_num":3,"objects":[{}]}',
                                             "image": b"\xff\xd8\x00\xff\xd9"})
                time.sleep(0.2)
            self.assertTrue((self.dir / "ready").exists(), "service never became ready")
            service.send_signal(signal.SIGTERM)
            service.wait(timeout=10)
            reader.join(timeout=10)
            output = "".join(lines)
            self.assertEqual(service.returncode, 0, output)
            for text in expected:
                self.assertIn(text, output)
            self.assertIn("redis.password=***", output)
            self.assertFalse((self.dir / "ready").exists())
        finally:
            if service is not None and service.poll() is None:  # 断言失败时不留下运行中的服务
                service.kill()
                service.wait()
            redis_process.terminate()
            redis_process.wait()

if __name__ == "__main__":
    unittest.main()
