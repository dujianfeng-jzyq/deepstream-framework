"""tools/push_sample_video.sh：参数检查、信号收尾与断线重连（假 ffmpeg，不访问网络）。"""

import os
import signal
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "tools/push_sample_video.sh"

FAKE_FFMPEG = """#!/usr/bin/env bash
echo "$$" >> "$FAKE_LOG"
printf '%s\n' "$@" > "$FAKE_LOG.args"
if [[ -n "${FAKE_EXIT_QUICKLY:-}" ]]; then exit 1; fi
trap '' INT  # 模拟忽略 SIGINT 的编码器，必须由脚本 TERM/KILL 回收
while true; do sleep 0.1; done
"""


class PushSampleVideoTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        (self.dir / "bin").mkdir()
        ffmpeg = self.dir / "bin/ffmpeg"
        ffmpeg.write_text(FAKE_FFMPEG)
        ffmpeg.chmod(0o755)
        self.video = self.dir / "v.mp4"
        self.video.write_bytes(b"x")
        self.log = self.dir / "ffmpeg.log"
        self.env = dict(os.environ, PATH=f"{self.dir / 'bin'}:{os.environ['PATH']}", FAKE_LOG=str(self.log))

    def start(self, **env):
        return subprocess.Popen(["bash", str(SCRIPT), str(self.video)], env={**self.env, **env},
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def pids(self):
        return [int(x) for x in self.log.read_text().split()] if self.log.exists() else []

    def wait_for(self, condition, timeout=5.0):
        deadline = time.time() + timeout
        while not condition() and time.time() < deadline:
            time.sleep(0.05)
        self.assertTrue(condition())

    def alive(self, pid):
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False

    def test_arguments(self):
        run = lambda *a: subprocess.run(["bash", str(SCRIPT), *a], env=self.env, capture_output=True, text=True)
        self.assertEqual(run().returncode, 2)
        self.assertEqual(run("--help").returncode, 0)
        self.assertEqual(run("/nope.mp4").returncode, 2)
        self.assertEqual(run(str(self.video), "http://x").returncode, 2)

    def test_signals_stop_and_reap_the_encoder(self):
        for signum, code in ((signal.SIGTERM, 143), (signal.SIGINT, 130)):
            self.log.unlink(missing_ok=True)
            process = self.start()
            self.wait_for(lambda: self.pids())
            process.send_signal(signum)
            self.assertEqual(process.wait(timeout=10), code)
            self.wait_for(lambda: not self.alive(self.pids()[0]))

    def test_default_url_carries_the_encoded_publish_password(self):
        process = self.start(MEDIAMTX_PUBLISH_PASSWORD="p$w#x@&")
        args = Path(f"{self.log}.args")
        self.wait_for(args.exists)
        process.send_signal(signal.SIGTERM)
        output, _ = process.communicate(timeout=10)
        self.assertIn("rtsp://publisher:p%24w%23x%40%26@127.0.0.1:8554/camera0", args.read_text().split())
        self.assertIn("rtsp://publisher:***@127.0.0.1:8554/camera0", output)
        self.assertNotIn("p%24w", output)

    def test_reconnects_after_the_stream_ends(self):
        process = self.start(FAKE_EXIT_QUICKLY="1")
        self.wait_for(lambda: len(self.pids()) >= 2, timeout=12)  # 5 秒后重连
        process.send_signal(signal.SIGTERM)
        output, _ = process.communicate(timeout=10)
        self.assertEqual(process.returncode, 143)
        self.assertIn("reconnecting in 5 seconds", output)


if __name__ == "__main__":
    unittest.main()
