"""来源：一批导入的素材（docs/training.md §5.1）。X-AnyLabeling 直接打开来源目录标注。

sources/<来源>/source.json 记录每张图：
  sha256   导入时的内容摘要（跨来源去重；prepare 用它发现导入后被替换的图像）
  origin   原始文件；视频帧另有 time_ms
  group    采集组，决定划分：video:<视频摘要>、prefix:<文件名前缀>、dir:<来源>/<子目录>
  width/height
以及 classes：来源为哪些类别标注过（§15.1）。导入不涉及模型，新来源的 classes 为空；train update <模型>
source add 把模型的类别记入，之后只由 train update 改写。给 X-AnyLabeling 的类别与属性文件按检测模型生成
（labeling.py）。accepted：用 train accept 批量确认的记录；reviews：新增类别时撤销 checked 的复核名单（§15.2），每项
  class/from/at  新类别、只撤销含哪个类别框的图片（--from，没有时为 null）、时间
  files          被撤销的图片；revoked 在全部撤销完成后才为 true（中断后重新执行 train update 按名单继续）
  skipped        不撤销的理由（--skip-source），files 为空
  accepted       用 train accept --reason 跳过复核的记录
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from . import media, paths, xanylabeling
from .errors import TrainError

SCHEMA = 1
GROUP_BY = ("directory", "filename_prefix")
# 按序号命名的连拍或视频帧：去掉末尾的 _<数字> 得到同一采集的前缀（例如 20260808_102138_00023000）。
_SEQUENCE_SUFFIX = re.compile(r"_\d+$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_record(directory: Path) -> dict:
    path = directory / paths.SOURCE_RECORD
    if not path.is_file():
        raise TrainError(f"{directory} is not a source (no {paths.SOURCE_RECORD}); create it with train import")
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema") != SCHEMA:
        raise TrainError(f"{path}: unsupported schema {record.get('schema')!r}")
    return record


def class_list(directory: Path, record: dict) -> list[str]:
    """来源的类别列表。记录还没有 classes 时取旧版导入写出的 labels.txt（迁移：labeling.refresh 把它记入
    source.json 后删除）。"""
    value = record.get("classes")
    if value is None:
        labels = directory / paths.SOURCE_LABELS
        if not labels.is_file():
            raise TrainError(f"{directory}: neither {paths.SOURCE_RECORD} nor {paths.SOURCE_LABELS} gives the "
                             "classes of the source")
        value = [line.strip() for line in labels.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not isinstance(value, list) or not all(isinstance(c, str) and c for c in value) or \
            len(set(value)) != len(value):
        raise TrainError(f"{directory}: the classes of the source must be distinct names")
    return value


def review_classes(record: dict) -> dict[str, list[str]]:
    """图像名 → 它在哪些已生效的复核名单中（类别）。名单只用于报告与预标注，收录仍然只看 checked。"""
    result: dict[str, list[str]] = {}
    for review in record.get("reviews") or []:
        if review.get("revoked"):
            for name in review["files"]:
                result.setdefault(name, []).append(review["class"])
    return result


def review_progress(directory: Path, record: dict) -> list[tuple[dict, int, int]]:
    """每份已生效的复核名单：(名单, 已重新勾选的数量, 总数)。"""
    progress = []
    for review in record.get("reviews") or []:
        if not review.get("revoked") or review.get("skipped"):
            continue
        checked = 0
        for name in review["files"]:
            path = xanylabeling.annotation_path(directory / name)
            checked += path.is_file() and xanylabeling.read(path).checked
        progress.append((review, checked, len(review["files"])))
    return progress


def known_images(root: Path) -> dict[str, str]:
    """全部来源中图像的 sha256 → "来源/文件名"：同一张图不能进入两个来源（可能一边训练一边测试）。"""
    known = {}
    for record_path in sorted((root / paths.SOURCES).glob(f"*/{paths.SOURCE_RECORD}")):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        for name, info in record.get("images", {}).items():
            known[info["sha256"]] = f"{record_path.parent.name}/{name}"
    return known


def prefix_group(stem: str) -> str:
    return "prefix:" + _SEQUENCE_SUFFIX.sub("", stem)


class Importer:
    def __init__(self, root: Path, source: str, group_by: str = "directory",
                 every_s: float = 1.0, dedupe: int = 4, max_frames: int = 0):
        if group_by not in GROUP_BY:
            raise TrainError(f"--group-by must be one of {', '.join(GROUP_BY)}")
        self.directory = paths.source_dir(root, source)
        self.source = source
        self.group_by = group_by
        self.every_s, self.dedupe, self.max_frames = every_s, dedupe, max_frames
        self.record_path = self.directory / paths.SOURCE_RECORD
        if self.record_path.is_file():
            self.record = read_record(self.directory)   # 已有的来源：类别不变（只由 train update 改写）
        else:
            # 导入只带来素材：来源为哪些类别标注由 train update <模型> source add 决定。
            self.record = {"schema": SCHEMA, "source": source, "created": _now(), "classes": [],
                           "images": {}, "accepted": []}
        self.known = known_images(root)
        self.report: Counter = Counter()
        self.notes: list[str] = []

    def run(self, inputs: list[Path]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        for item in inputs:
            if not item.exists():
                raise TrainError(f"{item} does not exist")
            files = sorted(p for p in item.rglob("*") if p.is_file()) if item.is_dir() else [item]
            base = item if item.is_dir() else item.parent
            for path in files:
                suffix = path.suffix.lower()
                if suffix in media.IMAGE_SUFFIXES:
                    self._image(path, base)
                elif suffix in media.VIDEO_SUFFIXES:
                    self._video(path)
                elif suffix != ".json":  # 标注随同名图像导入
                    self.report["other files skipped"] += 1
        paths.write_json(self.record_path, self.record)

    def _name(self, stem: str, suffix: str, digest: str) -> str:
        name = f"{stem}{suffix}"
        if name in self.record["images"] or (self.directory / name).exists():
            name = f"{stem}_{digest[:8]}{suffix}"  # 不同目录中的同名文件
        return name

    def _admit(self, file: Path, digest: str) -> bool:
        if digest in self.known:
            self.report["duplicates skipped"] += 1
            if len(self.notes) < 20:
                self.notes.append(f"{file} is already imported as {self.known[digest]}")
            return False
        return True

    def _add(self, file: Path, name: str, digest: str, size: tuple[int, int], info: dict) -> None:
        self.record["images"][name] = {"sha256": digest, "width": size[0], "height": size[1], **info}
        self.known[digest] = f"{self.source}/{name}"

    def _image(self, path: Path, base: Path) -> None:
        digest = paths.sha256(path)
        if not self._admit(path, digest):
            return
        try:
            size = media.image_size(path)
        except TrainError as error:
            self.report["unreadable images skipped"] += 1
            self.notes.append(str(error))
            return
        name = self._name(path.stem, path.suffix.lower(), digest)
        if self.group_by == "filename_prefix":
            group = prefix_group(path.stem)
        else:
            relative = path.parent.relative_to(base).as_posix()
            group = f"dir:{self.source}/{relative}"
        paths.link_or_copy(path, self.directory / name)
        self._add(path, name, digest, size, {"origin": str(path), "group": group})
        self.report["images imported"] += 1
        annotation = xanylabeling.annotation_path(path)
        if annotation.is_file():
            self._annotation(annotation, name)

    def _annotation(self, source: Path, image_name: str) -> None:
        """已有的标注随图像导入：imagePath 改为来源中的文件名，其余原样保留（包括 checked）。"""
        target = xanylabeling.annotation_path(self.directory / image_name)
        try:
            xanylabeling.read(source)
        except TrainError as error:
            self.report["unreadable annotations skipped"] += 1
            self.notes.append(str(error))
            return
        shutil.copy2(source, target)
        xanylabeling.update(target, imagePath=image_name)
        self.report["annotations imported"] += 1

    def _video(self, video: Path) -> None:
        video_digest = paths.sha256(video)
        group = f"video:{video_digest[:16]}"
        kept, last = 0, None
        with tempfile.TemporaryDirectory(prefix=".dsf-frames-", dir=self.directory) as temporary:
            for frame, time_ms in media.extract_frames(video, self.every_s, Path(temporary)):
                if self.max_frames and kept >= self.max_frames:
                    break
                fingerprint = media.dhash(frame)
                if last is not None and self.dedupe >= 0 and media.hamming(fingerprint, last) <= self.dedupe:
                    self.report["similar frames skipped"] += 1
                    continue
                digest = paths.sha256(frame)
                if not self._admit(frame, digest):
                    continue
                last = fingerprint
                name = self._name(f"{video.stem}_{time_ms:08d}", ".jpg", digest)
                frame.replace(self.directory / name)
                self._add(frame, name, digest, media.image_size(self.directory / name),
                          {"origin": str(video), "time_ms": time_ms, "group": group})
                kept += 1
        self.report["video frames imported"] += kept
        self.report["videos"] += 1


def accept(root: Path, source: str, files: list[str], everything: bool = False, reason: str = "") -> int:
    """把指定图像的标注标为已检查（checked: true），并记录这次批量确认（docs/training.md §5.2）。
    复核名单中还没有重新勾选的图片要有人为新类别看过，批量确认它们必须给出理由，理由记入名单（§15.2）。"""
    directory = paths.source_dir(root, source)
    record = read_record(directory)
    if everything == bool(files):
        raise TrainError("name the images to accept, or pass --all")
    names = sorted(record["images"]) if everything else []
    for value in files:
        candidates = [n for n in record["images"] if value in (n, Path(n).stem, Path(n).stem + ".json")]
        if not candidates:
            raise TrainError(f"{value} is not an image of source {source}")
        names += candidates
    targets = [(name, xanylabeling.annotation_path(directory / name)) for name in dict.fromkeys(names)]
    missing = [name for name, path in targets if not path.is_file()]
    if missing and not everything:
        raise TrainError(f"these images have no annotation yet, label them first: {', '.join(missing[:10])}")
    pending = [(name, path) for name, path in targets if path.is_file() and not xanylabeling.read(path).checked]
    reviewing = review_classes(record)
    waiting = [name for name, _ in pending if name in reviewing]
    if waiting and not reason:
        classes = sorted({c for name in waiting for c in reviewing[name]})
        raise TrainError(f"{len(waiting)} of these images wait for review of {', '.join(classes)} "
                         f"({', '.join(waiting[:5])}{' …' if len(waiting) > 5 else ''}): label the new classes in "
                         "them and check them in X-AnyLabeling, or give --reason to accept them as they are")
    accepted = []
    for name, path in pending:
        xanylabeling.update(path, checked=True)
        accepted.append(name)
    if accepted:
        now = _now()
        record.setdefault("accepted", []).append({"at": now, "images": accepted})
        for review in record.get("reviews") or []:
            skipped = [name for name in waiting if review.get("revoked") and name in review["files"]]
            if skipped:
                review.setdefault("accepted", []).append({"at": now, "reason": reason, "images": skipped})
        paths.write_json(directory / paths.SOURCE_RECORD, record)
    return len(accepted)
