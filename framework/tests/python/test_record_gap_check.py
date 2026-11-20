"""tools/record_gap_check.py：计划解析、NAL 与 RESP 解析、分段与时间线检查（假探测结果，不需要 ffmpeg）。"""

import importlib.util
import io
import os
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("record_gap_check", ROOT / "tools/record_gap_check.py")
rgc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rgc)

T0 = 1_800_000_000_000  # 测试用的墙钟起点（ms）


def frames(start, seconds, fps=15.0, holes=()):
    """从 start 秒起 seconds 秒的帧时间戳；holes 为 (在第几秒, 时长) 的断流。"""
    pts, t, step = [], start, 1.0 / fps
    pending = sorted(holes)
    while t < start + seconds - 1e-9:
        pts.append(round(t, 6))
        t += step
        if pending and t - start >= pending[0][0]:
            t += pending.pop(0)[1]
    return pts


def make_probe(pts, gop=2.0, idr=True, errors=()):
    keys = [p for p in pts if abs((p - pts[0]) / gop - round((p - pts[0]) / gop)) < 1e-6]
    return rgc.Probe(codec="h264", pts=pts, keyframes=keys or pts[:1], first_is_key=True,
                     first_is_idr=idr, decode_errors=list(errors))


class Fixture:
    """在临时目录中放分段文件，probe 返回预先给定的结果。"""

    def __init__(self, testcase):
        self.dir = Path(tempfile.mkdtemp()) / "camera0" / "osd"
        self.dir.mkdir(parents=True)
        self.probes = {}
        self.events = []
        self.seq = 0
        original = rgc.probe
        rgc.probe = lambda path, decode: self.probes[path.name]
        testcase.addCleanup(setattr, rgc, "probe", original)

    def segment(self, start_ms, probe, closed_ms, event=True):
        name = time.strftime("%Y%m%dT%H%M%S", time.localtime(start_ms // 1000))
        path = self.dir / f"camera0-osd-{name}.{start_ms % 1000:03d}-{self.seq}.mp4"
        self.seq += 1
        path.write_bytes(b"x")
        os.utime(path, (closed_ms / 1000, closed_ms / 1000))
        self.probes[path.name] = probe
        if event:
            sid = f"camera0-osd-{self.seq}"
            duration = int(probe.duration * 1000)
            self.events.append({"event_type": "recording_started", "segment_id": sid, "start_ms": start_ms,
                                "_id_ms": start_ms})
            self.events.append({"event_type": "recording_finished", "segment_id": sid, "start_ms": start_ms,
                                "end_ms": start_ms + duration, "duration_ms": duration,
                                "path": f"/data/recordings/camera0/osd/{path.name}", "_id_ms": closed_ms})
        return path

    def check(self, timeline=None, events=True, **opts):
        out = io.StringIO()
        report = rgc.check(self.dir.parent.parent, rgc.Options(**opts), timeline,
                           self.events if events else None, since_ms=0, report=rgc.Report(out))
        return report, out.getvalue()


def good_recording(fx):
    """推流 0–30 s，30 s 处卡顿 4 s（录像器 2 s 后收尾），34 s 恢复推流到 50 s，结束后 2 s 收尾。"""
    fx.segment(T0, make_probe(frames(0, 30)), T0 + 32_100)
    fx.segment(T0 + 34_050, make_probe(frames(0, 16)), T0 + 52_100)
    return {"start_ms": T0 - 500, "end_ms": T0 + 50_050,
            "gaps": [{"kind": "stall", "start_ms": T0 + 30_000, "end_ms": T0 + 34_000}]}


class PlanTest(unittest.TestCase):
    def test_default_plan(self):
        steps = rgc.parse_plan(rgc.DEFAULT_PLAN)
        self.assertEqual(steps[0].kind, "push")
        self.assertIn("stall", [s.kind for s in steps])
        self.assertIn("drop", [s.kind for s in steps])

    def test_rejects_bad_plans(self):
        for plan in ("", "stall:1 push:2", "push:2 stall:1", "push:x", "push:0", "wait:1 push:1", "push"):
            with self.subTest(plan=plan), self.assertRaises(ValueError):
                rgc.parse_plan(plan)

    def test_redacts_password(self):
        self.assertEqual(rgc.redact("rtsp://publisher:s%40cret@host:8554/a"), "rtsp://publisher:***@host:8554/a")


class ParseTest(unittest.TestCase):
    def test_h264_idr(self):
        sps_pps_idr = b"\x00\x00\x00\x01\x67aa\x00\x00\x00\x01\x68bb\x00\x00\x01\x65cc"
        self.assertEqual(rgc.nal_types(sps_pps_idr, "h264"), [7, 8, 5])
        self.assertTrue(rgc.first_picture_is_idr(rgc.nal_types(sps_pps_idr, "h264"), "h264"))
        non_idr_i = b"\x00\x00\x00\x01\x67aa\x00\x00\x01\x06ss\x00\x00\x01\x41cc"
        self.assertFalse(rgc.first_picture_is_idr(rgc.nal_types(non_idr_i, "h264"), "h264"))
        self.assertFalse(rgc.first_picture_is_idr([], "h264"))

    def test_h265_idr(self):
        vps_idr = b"\x00\x00\x01\x40\x01\x00\x00\x01\x26\x01"   # VPS(32), IDR_W_RADL(19)
        self.assertEqual(rgc.nal_types(vps_idr, "h265"), [32, 19])
        self.assertTrue(rgc.first_picture_is_idr([32, 33, 34, 19], "h265"))
        self.assertFalse(rgc.first_picture_is_idr([32, 33, 34, 21], "h265"))  # CRA

    def test_resp(self):
        reply = (b"*1\r\n*2\r\n$15\r\n1800000000000-0\r\n*2\r\n$7\r\npayload\r\n$2\r\n{}\r\n")
        self.assertEqual(rgc.resp_parse(io.BytesIO(reply)), [[b"1800000000000-0", [b"payload", b"{}"]]])
        self.assertEqual(rgc.resp_parse(io.BytesIO(b"$-1\r\n")), None)
        with self.assertRaises(rgc.RespError):
            rgc.resp_parse(io.BytesIO(b"-WRONGPASS invalid\r\n"))
        self.assertEqual(rgc.resp_command("AUTH", "pw"), b"*2\r\n$4\r\nAUTH\r\n$2\r\npw\r\n")


class CheckTest(unittest.TestCase):
    def test_good_recording_passes(self):
        fx = Fixture(self)
        report, out = fx.check(good_recording(fx))
        self.assertEqual(report.failures, 0, out)
        self.assertIn("finalized 2100 ms after the gap began", out)

    def test_works_without_events_and_timeline(self):
        fx = Fixture(self)
        good_recording(fx)
        report, out = fx.check(events=False)
        self.assertEqual(report.failures, 0, out)

    def test_gap_left_in_file(self):
        fx = Fixture(self)
        fx.segment(T0, make_probe(frames(0, 30, holes=[(10, 3.0)])), T0 + 35_000)
        report, out = fx.check()
        self.assertIn("the gap stayed in the file", out)
        self.assertEqual(report.failures, 1, out)

    def test_short_gap_may_stay_in_file(self):
        fx = Fixture(self)
        fx.segment(T0, make_probe(frames(0, 30, holes=[(10, 1.5)])), T0 + 32_000)
        report, out = fx.check()
        self.assertEqual(report.failures, 0, out)

    def test_first_frame_not_idr(self):
        fx = Fixture(self)
        fx.segment(T0, make_probe(frames(0, 10), idr=False), T0 + 10_000)
        report, out = fx.check()
        self.assertIn("not an IDR picture", out)

    def test_decode_errors(self):
        fx = Fixture(self)
        fx.segment(T0, make_probe(frames(0, 10), errors=["[h264] no frame!"]), T0 + 10_000)
        report, out = fx.check()
        self.assertIn("decode error", out)

    def test_segment_too_long(self):
        fx = Fixture(self)
        fx.segment(T0, make_probe(frames(0, 65)), T0 + 65_000)
        report, out = fx.check(segment_seconds=60)
        self.assertIn("segment_seconds + one GOP", out)

    def test_segment_not_finalized_during_gap(self):
        fx = Fixture(self)
        timeline = good_recording(fx)
        # 卡顿中没有收尾：第一个分段等到恢复后才收尾（录像器的旧行为）。
        first = sorted(fx.dir.iterdir())[0]
        closed = T0 + 36_000
        os.utime(first, (closed / 1000, closed / 1000))
        fx.events[1]["_id_ms"] = closed
        report, out = fx.check(timeline)
        self.assertIn("no segment was finalized during the gap", out)

    def test_no_segment_after_resume(self):
        fx = Fixture(self)
        fx.segment(T0, make_probe(frames(0, 30)), T0 + 32_100)
        timeline = {"start_ms": T0 - 500, "end_ms": T0 + 50_050,
                    "gaps": [{"kind": "drop", "start_ms": T0 + 30_000, "end_ms": T0 + 34_000}]}
        report, out = fx.check(timeline)
        self.assertIn("no new segment started", out)

    def test_last_segment_not_finalized_after_stream_stops(self):
        fx = Fixture(self)
        fx.segment(T0, make_probe(frames(0, 30)), T0 + 32_100)
        (fx.dir / "camera0-osd-x-1.mp4.tmp").write_bytes(b"x")
        timeline = {"start_ms": T0 - 500, "end_ms": T0 + 50_000, "gaps": []}
        report, out = fx.check(timeline)
        self.assertIn("WARN camera0-osd-x-1.mp4.tmp", out)
        self.assertIn("end at +50.5s", out)
        self.assertIn("no segment was finalized during the gap", out)

    def test_event_mismatches(self):
        fx = Fixture(self)
        good_recording(fx)
        fx.events[1]["duration_ms"] += 500
        fx.events[2]["start_ms"] += 1
        report, out = fx.check()
        self.assertIn("duration_ms", out)
        self.assertIn("start_ms differs", out)

    def test_missing_finished_event(self):
        fx = Fixture(self)
        fx.segment(T0, make_probe(frames(0, 10)), T0 + 10_000, event=False)
        report, out = fx.check()
        self.assertIn("no recording_finished event", out)

    def test_overlap(self):
        fx = Fixture(self)
        fx.segment(T0, make_probe(frames(0, 30)), T0 + 30_000)
        fx.segment(T0 + 29_600, make_probe(frames(0, 10)), T0 + 40_000)
        fx.segment(T0 + 38_000, make_probe(frames(0, 10)), T0 + 48_000)
        report, out = fx.check()
        self.assertIn("WARN", out)
        self.assertEqual(report.failures, 1, out)
        self.assertEqual(report.warnings, 1, out)

    def test_visible_placeholder(self):
        fx = Fixture(self)
        good_recording(fx)
        (fx.dir / "placeholder-0.mp4.tmp").write_bytes(b"x")
        report, out = fx.check()
        self.assertIn("placeholder file is visible", out)

    def test_empty_directory(self):
        fx = Fixture(self)
        report, out = fx.check()
        self.assertIn("no finished segments", out)


if __name__ == "__main__":
    unittest.main()
