#!/usr/bin/env python3
"""断流场景下的录像检查：按计划推流并制造断流，再逐个检查录像分段。

  push  <视频> [--url URL] [--plan PLAN] [--timeline 文件]   推流并按计划断流，写出断流时间线
  check <录像目录> [--timeline 文件] [--redis 主机[:端口]] ... 检查分段文件（及录像事件）
  run   <视频> <录像目录> [push 与 check 的参数]           推流、等待最后一个分段收尾、检查

计划是空白分隔的步骤（秒）：push:S 推流；stall:S 暂停推流进程（连接保持，模拟网络卡顿，
恢复后 ffmpeg 会补发积压的帧）；drop:S 断开推流（模拟相机掉线，应用按重连间隔重连）。
默认计划依次覆盖：短于 keyframe_on_gap、介于 keyframe_on_gap 与 split_on_gap 之间、
长于 split_on_gap 的卡顿，以及一次掉线；最后停止推流，录像器应在断流中收尾最后一个分段。

检查项（FAIL 使退出码为 1）：
  - 每个分段第一帧是关键帧且为 IDR（H.264/H.265），整段解码无错误（--no-decode 跳过）；
  - 分段内相邻两帧的时间戳间隔小于 split_on_gap（断流没有以静止画面留在文件中）；
  - 分段时长不超过 segment_seconds 加一个关键帧间隔；目录中没有可见的占位文件；
  - 有 --redis 时：每个分段都有 recording_started/finished，start_ms 一致，duration_ms 与文件时长一致，
    相邻分段重叠不超过 1 秒（超过 100 ms 记 WARN）；
  - 有 --timeline 时：每次长于 split_on_gap 的断流期间有分段收尾（不是等到恢复才收尾），恢复后开始新分段。
    时间线用推流机器的时钟，录像用应用宿主的时钟：两者不在同一台机器时须已对时（NTP）。

依赖 ffmpeg/ffprobe（push 需要 libx264）与 Python 3.9+ 标准库。Redis 在 bridge 网络中不映射端口，
在 Linux 宿主上可直接访问容器地址：
  --redis $(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' <redis 容器>)
口令取自环境变量 REDIS_PASSWORD，Stream 名为配置中的 redis.recording_stream。
"""

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

DEFAULT_PLAN = "push:30 stall:0.5 push:20 stall:1.5 push:20 stall:4 push:20 drop:8 push:30"
DEFAULT_URL_PATH = "camera0"
# 文件名中的起始本地时间与序号，见 docs/framework.md §8。
NAME_RE = re.compile(r"-(\d{8}T\d{6}\.\d{3})-(\d+)(-detected)?\.mp4$")


def now_ms() -> int:
    return int(time.time() * 1000)


# ---------------------------------------------------------------------------
# 推流


@dataclass
class Step:
    kind: str
    seconds: float


def parse_plan(text: str) -> List[Step]:
    steps = []
    for token in text.split():
        kind, sep, value = token.partition(":")
        if not sep or kind not in ("push", "stall", "drop"):
            raise ValueError(f"bad plan step {token!r}: expected push:S, stall:S or drop:S")
        try:
            seconds = float(value)
        except ValueError:
            raise ValueError(f"bad plan step {token!r}: seconds must be a number") from None
        if seconds <= 0:
            raise ValueError(f"bad plan step {token!r}: seconds must be positive")
        steps.append(Step(kind, seconds))
    if not steps or steps[0].kind != "push":
        raise ValueError("plan must start with push:S")
    if steps[-1].kind != "push":
        raise ValueError("plan must end with push:S (the stream stops after the last step)")
    return steps


def default_url() -> str:
    password = quote(os.environ.get("MEDIAMTX_PUBLISH_PASSWORD", ""), safe="")
    return f"rtsp://publisher:{password}@127.0.0.1:8554/{DEFAULT_URL_PATH}"


def redact(url: str) -> str:
    return re.sub(r"^(rtsps?://[^:/@]*):[^/@]*@", r"\1:***@", url)


class Pusher:
    """一个 ffmpeg 推流进程。参数与 tools/push_sample_video.sh 一致。"""

    def __init__(self, video: str, url: str):
        self.video, self.url = video, url
        self.proc: Optional[subprocess.Popen] = None

    def start(self) -> None:
        self.proc = subprocess.Popen(
            ["ffmpeg", "-nostdin", "-loglevel", "warning", "-re", "-stream_loop", "-1", "-i", self.video,
             "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency", "-g", "30", "-pix_fmt", "yuv420p",
             "-an", "-f", "rtsp", "-rtsp_transport", "tcp", self.url])

    def check_alive(self) -> None:
        if self.proc is not None and self.proc.poll() is not None:
            raise RuntimeError(f"ffmpeg exited with {self.proc.returncode} while pushing")

    def signal(self, sig: int) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.send_signal(sig)

    def stop(self) -> None:
        if self.proc is None:
            return
        # 暂停中的进程收不到 TERM：先恢复。
        self.signal(signal.SIGCONT)
        self.signal(signal.SIGTERM)
        try:
            self.proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        self.proc = None


def wait_seconds(seconds: float, pusher: Optional[Pusher] = None) -> None:
    deadline = time.monotonic() + seconds
    while True:
        left = deadline - time.monotonic()
        if left <= 0:
            return
        time.sleep(min(left, 0.2))
        if pusher is not None:
            pusher.check_alive()


def push(video: str, url: str, steps: List[Step], log=print) -> dict:
    """按计划推流，返回时间线：{"start_ms", "end_ms", "gaps": [{"kind", "start_ms", "end_ms"}]}。"""
    if not Path(video).is_file():
        raise RuntimeError(f"video file not found: {video}")
    pusher = Pusher(video, url)
    timeline = {"url": redact(url), "start_ms": now_ms(), "gaps": []}
    log(f"pushing {video} -> {redact(url)}")
    pusher.start()
    try:
        for step in steps:
            if step.kind == "push":
                log(f"  push {step.seconds:g}s")
                wait_seconds(step.seconds, pusher)
                continue
            log(f"  {step.kind} {step.seconds:g}s")
            gap = {"kind": step.kind, "start_ms": now_ms()}
            if step.kind == "stall":
                pusher.signal(signal.SIGSTOP)
                wait_seconds(step.seconds)
                gap["end_ms"] = now_ms()
                pusher.signal(signal.SIGCONT)
            else:
                pusher.stop()
                wait_seconds(step.seconds)
                gap["end_ms"] = now_ms()
                pusher.start()
            timeline["gaps"].append(gap)
    finally:
        pusher.stop()
    timeline["end_ms"] = now_ms()
    log("stream stopped")
    return timeline


# ---------------------------------------------------------------------------
# 读取录像事件（最小的 RESP2 客户端，只用于 AUTH 与 XRANGE）


class RespError(Exception):
    pass


def resp_command(*args: str) -> bytes:
    out = [f"*{len(args)}\r\n".encode()]
    for arg in args:
        data = arg.encode()
        out.append(b"$%d\r\n%s\r\n" % (len(data), data))
    return b"".join(out)


def resp_parse(stream):
    """从类文件对象读取一个 RESP2 回复。"""
    line = stream.readline()
    if not line.endswith(b"\r\n"):
        raise RespError("connection closed")
    kind, body = line[:1], line[1:-2]
    if kind == b"+":
        return body.decode()
    if kind == b"-":
        raise RespError(body.decode(errors="replace"))
    if kind == b":":
        return int(body)
    if kind == b"$":
        size = int(body)
        if size < 0:
            return None
        data = stream.read(size + 2)
        return data[:-2]
    if kind == b"*":
        count = int(body)
        return None if count < 0 else [resp_parse(stream) for _ in range(count)]
    raise RespError(f"unexpected reply {line!r}")


def read_recording_events(address: str, stream_name: str, password: str) -> List[dict]:
    """返回 Stream 中的全部录像事件，每个事件附加 _id_ms（Redis 收到它时的服务器时钟）。"""
    host, _, port = address.partition(":")
    with socket.create_connection((host, int(port or 6379)), timeout=5) as sock:
        reader = sock.makefile("rb")
        if password:
            sock.sendall(resp_command("AUTH", password))
            resp_parse(reader)
        sock.sendall(resp_command("XRANGE", stream_name, "-", "+"))
        entries = resp_parse(reader) or []
    events = []
    for entry_id, fields in entries:
        values = dict(zip(fields[::2], fields[1::2]))
        try:
            event = json.loads(values[b"payload"])
        except (KeyError, ValueError):
            continue
        event["_id_ms"] = int(entry_id.split(b"-")[0])
        events.append(event)
    return events


# ---------------------------------------------------------------------------
# 分段文件


def nal_types(data: bytes, codec: str) -> List[int]:
    """Annex B 字节流中各 NAL 单元的类型。"""
    types = []
    i = 0
    while True:
        i = data.find(b"\x00\x00\x01", i)
        if i < 0 or i + 3 >= len(data):
            return types
        header = data[i + 3]
        types.append(header & 0x1F if codec == "h264" else (header >> 1) & 0x3F)
        i += 3


def first_picture_is_idr(types: List[int], codec: str) -> bool:
    """第一个图像 NAL 是否为 IDR（H.264 类型 5；H.265 类型 19/20）。"""
    for t in types:
        if codec == "h264" and 1 <= t <= 5:
            return t == 5
        if codec == "h265" and t <= 31:
            return t in (19, 20)
    return False


@dataclass
class Probe:
    codec: str
    pts: List[float]                 # 按时间戳排序，秒
    keyframes: List[float]           # 关键帧的时间戳，秒
    first_is_key: bool               # 解码顺序的第一帧
    first_is_idr: Optional[bool] = None
    decode_errors: List[str] = field(default_factory=list)

    @property
    def frame_interval(self) -> float:
        deltas = sorted(b - a for a, b in zip(self.pts, self.pts[1:]))
        return deltas[len(deltas) // 2] if deltas else 0.0

    @property
    def duration(self) -> float:
        """画面时长：第一帧到最后一帧，加最后一帧的时长。"""
        return self.pts[-1] - self.pts[0] + self.frame_interval if self.pts else 0.0

    @property
    def max_frame_gap(self) -> float:
        return max((b - a for a, b in zip(self.pts, self.pts[1:])), default=0.0)

    @property
    def max_gop(self) -> float:
        marks = self.keyframes + ([self.pts[-1] + self.frame_interval] if self.pts else [])
        return max((b - a for a, b in zip(marks, marks[1:])), default=self.duration)


def run_tool(args: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def probe(path: Path, decode: bool) -> Probe:
    out = run_tool(["ffprobe", "-v", "error", "-select_streams", "v:0",
                    "-show_entries", "stream=codec_name:packet=pts_time,flags", "-of", "json", str(path)])
    if out.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {out.stderr.decode(errors='replace').strip()}")
    info = json.loads(out.stdout)
    streams = info.get("streams") or []
    if not streams:
        raise RuntimeError("no video stream")
    codec = {"hevc": "h265"}.get(streams[0]["codec_name"], streams[0]["codec_name"])
    packets = [p for p in info.get("packets", []) if p.get("pts_time") not in (None, "N/A")]
    if not packets:
        raise RuntimeError("no video frames")
    result = Probe(
        codec=codec,
        pts=sorted(float(p["pts_time"]) for p in packets),
        keyframes=sorted(float(p["pts_time"]) for p in packets if "K" in p.get("flags", "")),
        first_is_key="K" in packets[0].get("flags", ""))
    if codec in ("h264", "h265"):
        fmt = "h264" if codec == "h264" else "hevc"
        out = run_tool(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-c", "copy",
                        "-bsf:v", f"{fmt}_mp4toannexb", "-frames:v", "1", "-f", fmt, "-"])
        result.first_is_idr = first_picture_is_idr(nal_types(out.stdout, codec), codec)
    if decode:
        # 时间戳按到达时间生成（可变帧率）：passthrough 避免 null 封装按恒定帧率重排时间戳，
        # 只留下解码器本身的错误。
        out = run_tool(["ffmpeg", "-v", "repeat+error", "-i", str(path), "-map", "0:v:0", "-vsync", "passthrough",
                        "-f", "null", "-"])
        result.decode_errors = [l for l in out.stderr.decode(errors="replace").splitlines()
                                if l.strip() and "non monotonically increasing dts" not in l]
    return result


def start_ms_from_name(path: Path) -> Optional[int]:
    """文件名中的起始本地时间（检查机器的时区）。只在没有事件时使用。"""
    m = NAME_RE.search(path.name)
    if not m:
        return None
    stamp, millis = m.group(1).split(".")
    return int(time.mktime(time.strptime(stamp, "%Y%m%dT%H%M%S"))) * 1000 + int(millis)


# ---------------------------------------------------------------------------
# 检查


@dataclass
class Segment:
    path: Path
    probe: Optional[Probe] = None
    probe_error: str = ""
    started: Optional[dict] = None
    finished: Optional[dict] = None

    @property
    def start_ms(self) -> Optional[int]:
        if self.finished is not None:
            return self.finished.get("start_ms")
        return start_ms_from_name(self.path)

    @property
    def end_ms(self) -> Optional[int]:
        if self.finished is not None:
            return self.finished.get("end_ms")
        start = self.start_ms
        return None if start is None or self.probe is None else start + int(self.probe.duration * 1000)

    @property
    def closed_ms(self) -> int:
        """分段收尾的时间：finished 事件进入 Redis 的时间，没有事件时取文件修改时间。"""
        if self.finished is not None:
            return self.finished["_id_ms"]
        return int(self.path.stat().st_mtime * 1000)


@dataclass
class Options:
    segment_seconds: float = 60.0
    split_on_gap_ms: int = 2000
    reconnect_slack_s: float = 15.0


class Report:
    def __init__(self, out=sys.stdout):
        self.out = out
        self.failures = 0
        self.warnings = 0

    def fail(self, subject: str, message: str) -> None:
        self.failures += 1
        print(f"FAIL {subject}: {message}", file=self.out)

    def warn(self, subject: str, message: str) -> None:
        self.warnings += 1
        print(f"WARN {subject}: {message}", file=self.out)

    def info(self, message: str) -> None:
        print(message, file=self.out)


def check_file(seg: Segment, opts: Options, report: Report) -> None:
    name = seg.path.name
    if seg.probe is None:
        report.fail(name, seg.probe_error or "not probed")
        return
    p = seg.probe
    if not p.first_is_key:
        report.fail(name, "first frame is not a keyframe")
    elif p.first_is_idr is False:
        report.fail(name, f"first frame is a keyframe but not an IDR picture ({p.codec})")
    if p.decode_errors:
        report.fail(name, f"{len(p.decode_errors)} decode error(s), first: {p.decode_errors[0]}")
    if opts.split_on_gap_ms > 0 and p.max_frame_gap * 1000 >= opts.split_on_gap_ms:
        report.fail(name, f"{p.max_frame_gap:.2f}s between two frames >= split_on_gap "
                          f"({opts.split_on_gap_ms} ms): the gap stayed in the file")
    limit = opts.segment_seconds + p.max_gop + 0.5
    if p.duration > limit:
        report.fail(name, f"duration {p.duration:.2f}s > segment_seconds + one GOP ({limit:.2f}s)")


def check_events(segments: List[Segment], events: List[dict], report: Report) -> None:
    started = {e.get("segment_id"): e for e in events if e.get("event_type") == "recording_started"}
    for seg in segments:
        name = seg.path.name
        if seg.finished is None:
            report.fail(name, "no recording_finished event")
            continue
        sid = seg.finished.get("segment_id")
        seg.started = started.get(sid)
        if seg.started is None:
            report.warn(name, f"no recording_started event for {sid} (stream trimmed by MAXLEN?)")
        elif seg.started.get("start_ms") != seg.finished.get("start_ms"):
            report.fail(name, f"start_ms differs: started {seg.started.get('start_ms')}, "
                              f"finished {seg.finished.get('start_ms')}")
        if seg.probe is not None and seg.finished.get("duration_ms") is not None:
            tolerance = 2 * seg.probe.frame_interval * 1000 + 50
            diff = seg.finished["duration_ms"] - seg.probe.duration * 1000
            if abs(diff) > tolerance:
                report.fail(name, f"duration_ms {seg.finished['duration_ms']} vs file "
                                  f"{seg.probe.duration * 1000:.0f} ms (off by {diff:+.0f} ms)")


def check_sequence(segments: List[Segment], report: Report) -> None:
    for prev, cur in zip(segments, segments[1:]):
        if prev.end_ms is None or cur.start_ms is None:
            continue
        # start_ms 是第一帧进入录像器的墙钟，end_ms 是它加画面时长：管线延时的变化（例如源刚上线时的
        # 启动延时）使相邻分段略有重叠。超过 1 秒视为错误。
        overlap = prev.end_ms - cur.start_ms
        if overlap > 1000:
            report.fail(cur.path.name, f"starts {overlap} ms before the previous segment ends")
        elif overlap > 100:
            report.warn(cur.path.name, f"starts {overlap} ms before the previous segment ends")


def check_timeline(segments: List[Segment], timeline: dict, opts: Options, report: Report) -> None:
    gaps = list(timeline.get("gaps", []))
    # 推流停止后录像器应同样在断流中收尾最后一个分段。
    gaps.append({"kind": "end", "start_ms": timeline["end_ms"], "end_ms": None})
    for gap in gaps:
        start, end = gap["start_ms"], gap["end_ms"]
        length = None if end is None else end - start
        label = (f"{gap['kind']} at +{(start - timeline['start_ms']) / 1000:.1f}s"
                 + ("" if length is None else f" ({length / 1000:g}s)"))
        if length is not None and length < opts.split_on_gap_ms + 1000:
            continue  # 短于 split_on_gap（含余量）：分段可以跨越它，文件检查保证它不以长静止留在文件中
        # 收尾在断流开始之后、恢复（加管线延时）之前；最后一次断流给 split_on_gap 加 5 秒。
        upper = (end + 1000) if end is not None else (start + opts.split_on_gap_ms + 5000)
        # 只算最后一帧在断流之前的分段（加 1.5 s 管线延时），排除恰好在此时按时长切分的分段。
        closes = [s for s in segments if start <= s.closed_ms <= upper
                  and (s.end_ms is None or s.end_ms <= start + 1500)]
        if not closes:
            report.fail(label, "no segment was finalized during the gap (expected about "
                               f"{opts.split_on_gap_ms} ms after it began)")
        else:
            delay = closes[0].closed_ms - start
            report.info(f"ok   {label}: {closes[0].path.name} finalized {delay} ms after the gap began")
        if end is None:
            continue
        resumed = [s for s in segments if s.start_ms is not None
                   and end - 500 <= s.start_ms <= end + opts.reconnect_slack_s * 1000]
        if not resumed:
            report.fail(label, f"no new segment started within {opts.reconnect_slack_s:g}s after the stream resumed")
        else:
            report.info(f"ok   {label}: {resumed[0].path.name} started {resumed[0].start_ms - end} ms after resume")


def collect(directory: Path, since_ms: Optional[int]) -> Tuple[List[Path], List[Path], List[Path]]:
    """返回（分段、临时文件、占位文件），只收入修改时间不早于 since_ms 的文件。"""
    segments, temps, placeholders = [], [], []
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        if path.name.startswith("placeholder-"):
            placeholders.append(path)
            continue
        if since_ms is not None and path.stat().st_mtime * 1000 < since_ms:
            continue
        if path.name.endswith(".mp4.tmp"):
            temps.append(path)
        elif path.name.endswith(".mp4"):
            segments.append(path)
    return segments, temps, placeholders


def check(directory: Path, opts: Options, timeline: Optional[dict] = None, events: Optional[List[dict]] = None,
          since_ms: Optional[int] = None, decode: bool = True, report: Optional[Report] = None) -> Report:
    report = report or Report()
    if not directory.is_dir():
        report.fail(str(directory), "not a directory")
        return report
    if since_ms is None and timeline is not None:
        since_ms = timeline["start_ms"] - 2000
    paths, temps, placeholders = collect(directory, since_ms)
    for path in placeholders:
        report.fail(path.name, "placeholder file is visible in the recording directory")
    for path in temps:
        report.warn(path.name, "segment still being written (or left behind)")
    if not paths:
        report.fail(str(directory), "no finished segments to check")
        return report

    finished = {}
    for event in events or []:
        if event.get("event_type") == "recording_finished" and event.get("path"):
            finished[Path(event["path"]).name] = event

    # 每个录像器（目录）独立成序列，按起始时间与序号排序。
    groups: Dict[Path, List[Segment]] = {}
    for path in paths:
        seg = Segment(path, finished=finished.get(path.name))
        try:
            seg.probe = probe(path, decode)
        except (RuntimeError, ValueError, OSError) as error:
            seg.probe_error = str(error)
        groups.setdefault(path.parent, []).append(seg)

    all_segments = []
    for parent, segs in sorted(groups.items()):
        segs.sort(key=lambda s: (s.start_ms or 0, s.path.name))
        report.info(f"{parent}: {len(segs)} segment(s)")
        for seg in segs:
            p = seg.probe
            if p is not None:
                report.info(f"  {seg.path.name}  {p.codec} {p.duration:6.2f}s  frames={len(p.pts)}  "
                            f"gop<={p.max_gop:.2f}s  max-gap={p.max_frame_gap:.3f}s")
            check_file(seg, opts, report)
        if events is not None:
            check_events(segs, events, report)
        check_sequence(segs, report)
        all_segments.extend(segs)
    if timeline is not None:
        check_timeline(sorted(all_segments, key=lambda s: s.closed_ms), timeline, opts, report)
    report.info(f"{report.failures} failure(s), {report.warnings} warning(s)")
    return report


# ---------------------------------------------------------------------------
# 命令行


def add_push_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--url", default=None, help="RTSP publish URL (default: local MediaMTX, path camera0, "
                                               "password from MEDIAMTX_PUBLISH_PASSWORD)")
    p.add_argument("--plan", default=DEFAULT_PLAN, help=f"gap plan (default: {DEFAULT_PLAN!r})")
    p.add_argument("--timeline", default=None, help="write (push/run) or read (check) the gap timeline JSON")


def add_check_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--segment-seconds", type=float, default=60.0, help="recording.segment_seconds")
    p.add_argument("--split-on-gap-ms", type=int, default=2000, help="recording.split_on_gap_ms")
    p.add_argument("--reconnect-slack", type=float, default=15.0,
                   help="seconds allowed between resume and the first new segment (RTSP reconnect)")
    p.add_argument("--redis", default=None, help="HOST[:PORT] to read recording events from")
    p.add_argument("--stream", default=None, help="recording stream (redis.recording_stream)")
    p.add_argument("--since", type=float, default=None,
                   help="only check files modified after this Unix time (default: timeline start)")
    p.add_argument("--no-decode", action="store_true", help="skip the full decode of every segment")


def options_from(args) -> Options:
    return Options(args.segment_seconds, args.split_on_gap_ms, args.reconnect_slack)


def load_events(args) -> Optional[List[dict]]:
    if not args.redis:
        return None
    if not args.stream:
        raise SystemExit("--stream is required with --redis")
    return read_recording_events(args.redis, args.stream, os.environ.get("REDIS_PASSWORD", ""))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog="\n".join(__doc__.splitlines()[1:]))
    sub = parser.add_subparsers(dest="command", required=True)
    p_push = sub.add_parser("push", help="push a video and inject gaps")
    p_push.add_argument("video")
    add_push_args(p_push)
    p_check = sub.add_parser("check", help="check recorded segments")
    p_check.add_argument("directory")
    p_check.add_argument("--timeline", default=None, help="gap timeline written by push")
    add_check_args(p_check)
    p_run = sub.add_parser("run", help="push, wait for the last segment, check")
    p_run.add_argument("video")
    p_run.add_argument("directory")
    add_push_args(p_run)
    add_check_args(p_run)
    args = parser.parse_args(argv)

    stop = {"signal": None}

    def on_signal(signum, _frame):
        stop["signal"] = signum
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, on_signal)
    try:
        if args.command in ("push", "run"):
            try:
                steps = parse_plan(args.plan)
            except ValueError as error:
                parser.error(str(error))
            timeline = push(args.video, args.url or default_url(), steps)
            if args.timeline:
                Path(args.timeline).write_text(json.dumps(timeline, indent=2) + "\n")
            if args.command == "push":
                return 0
            # 等最后一个分段在断流中收尾（split_on_gap 之后），finished 事件随之写入。
            wait = args.split_on_gap_ms / 1000 + 5
            print(f"waiting {wait:g}s for the last segment to be finalized")
            wait_seconds(wait)
        else:
            timeline = json.loads(Path(args.timeline).read_text()) if args.timeline else None
        since_ms = int(args.since * 1000) if args.since is not None else None
        report = check(Path(args.directory), options_from(args), timeline, load_events(args),
                       since_ms, decode=not args.no_decode)
        return 1 if report.failures else 0
    except KeyboardInterrupt:
        return 128 + (stop["signal"] or signal.SIGINT)
    except (RuntimeError, OSError, RespError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
