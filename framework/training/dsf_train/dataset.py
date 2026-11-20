"""train prepare：从来源中已检查的标注生成不可变的数据集（docs/training.md §6）。

datasets/<模型>/<数据集标识>/
  manifest.json   每个样本的来源、图像、sha256、尺寸、采集组、划分与解读后的对象；标识 = 它的摘要
  view/<后端>/     后端需要的目录格式（硬链接到来源图像）
素材、标注与划分参数都不变时标识不变，已有的数据集直接复用。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from . import backends, lock, paths, sources, split, tasks, xanylabeling
from .errors import TrainError

SCHEMA = 1
MAX_LISTED = 30


@dataclass
class Result:
    dataset_id: str
    directory: Path
    reused: bool
    manifest: dict
    skipped: Counter = field(default_factory=Counter)  # (来源, 原因) → 数量
    ignored_shapes: Counter = field(default_factory=Counter)  # 不收入的形状：原因 → 数量
    notes: list[str] = field(default_factory=list)


def prepare(repo: Path, root: Path, spec: dict) -> Result:
    """类别依赖（§15.1）先于读取标注检查：模型的类别必须都在每个来源中，来源的类别必须与 sources.lock.json
    的记录相同（记录中没有的来源在这里补记）。"""
    task = tasks.get(spec["task"])
    backend = backends.get(spec["train"]["backend"] if isinstance(spec.get("train"), dict) else None, task.NAME)
    data = spec["data"]
    if not data["sources"]:
        raise TrainError(f"data.sources of {spec['id']} is empty: add the sources you imported to model.yaml")
    source_classes = lock.current(root, spec)
    problems = lock.missing_classes(spec, source_classes)
    if problems:
        raise TrainError("\n".join(problems))
    notes = lock.check(repo, root, spec, record_new=True, now=source_classes)
    params = data["split"]
    declared_ignore = set(data.get("ignore_classes") or [])
    samples, problems = [], []
    skipped, ignored = Counter(), Counter()
    for source in data["sources"]:
        directory = paths.source_dir(root, source)
        record = sources.read_record(directory)
        # 来源中有、模型不用的类别（其他模型的类别，或本模型 retire 的类别）在读取时忽略。
        ignore = declared_ignore | (set(source_classes[source]) - set(task.source_labels(spec)))
        reviewing = sources.review_classes(record)
        for name, info in sorted(record["images"].items()):
            image = directory / name
            annotation_path = xanylabeling.annotation_path(image)
            where = f"{source}/{name}"
            if not image.is_file():
                problems.append(f"{where}: the image is missing")
                continue
            annotation = xanylabeling.read(annotation_path) if annotation_path.is_file() else None
            reason = skip_reason(name, annotation, reviewing)
            if reason:
                skipped[(source, reason)] += 1
                continue
            if paths.sha256(image) != info["sha256"]:
                problems.append(f"{where}: the image changed after it was imported")
                continue
            if (annotation.width, annotation.height) != (info["width"], info["height"]):
                problems.append(f"{where}: the annotation is for a {annotation.width}x{annotation.height} image, "
                                f"the image is {info['width']}x{info['height']}")
                continue
            objects, found, other_shapes = task.samples(annotation, spec, ignore)
            problems += [f"{where}: {problem}" for problem in found]
            ignored.update(other_shapes)
            samples.append({"source": source, "image": name, "sha256": info["sha256"],
                            "width": info["width"], "height": info["height"], "group": info["group"],
                            "split": split.assign(info["group"], params["seed"], params["val"], params["test"]),
                            "objects": objects})
    if problems:
        listed = "\n  ".join(problems[:MAX_LISTED])
        more = f"\n  … and {len(problems) - MAX_LISTED} more" if len(problems) > MAX_LISTED else ""
        raise TrainError(f"{len(problems)} problems in the checked annotations; fix them in X-AnyLabeling:\n  "
                         f"{listed}{more}")
    if not samples:
        raise TrainError("no checked images in " + ", ".join(data["sources"]) +
                         ": mark finished images as checked in X-AnyLabeling (or train accept)")

    # 清单只含决定训练结果的内容（不含时间与路径），同样的输入总得到同样的标识。
    manifest = {"schema": SCHEMA, "model": spec["id"], "task": task.NAME, "classes": spec["classes"],
                **task.manifest_fields(spec),
                "split": {"seed": params["seed"], "val": params["val"], "test": params["test"]},
                "samples": samples}
    text = json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=1) + "\n"
    dataset_id = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    models_root = root / paths.DATASETS / spec["id"]
    directory = models_root / dataset_id
    reused = (directory / paths.MANIFEST).is_file()
    if not reused:
        models_root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{dataset_id}.", dir=models_root))
        try:
            (staging / paths.MANIFEST).write_text(text, encoding="utf-8")
            backend.write_view(task.NAME, manifest, root, staging / paths.VIEW / backend.NAME,
                               directory / paths.VIEW / backend.NAME)
            os.rename(staging, directory)  # 数据集要么完整出现，要么不出现
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    (models_root / paths.LATEST).write_text(dataset_id + "\n", encoding="utf-8")
    return Result(dataset_id, directory, reused, manifest, skipped, ignored, notes)


def skip_reason(name: str, annotation, reviewing: dict[str, list[str]]) -> str:
    """图片不进入数据集的原因；收入时为空。新增类别时撤销的图片单独报告：它们以前检查过，只差新类别（§15.2）。"""
    if annotation is None:
        return "not labeled"
    if not annotation.checked:
        return f"waiting for review of {', '.join(reviewing[name])}" if name in reviewing else "not checked"
    return ""


def skipped_now(root: Path, source: str) -> Counter:
    """来源中现在不会进入数据集的图片，按原因计数（发布记录用，§8）。"""
    directory = paths.source_dir(root, source)
    record = sources.read_record(directory)
    reviewing = sources.review_classes(record)
    counts = Counter()
    for name in record["images"]:
        path = xanylabeling.annotation_path(directory / name)
        reason = skip_reason(name, xanylabeling.read(path) if path.is_file() else None, reviewing)
        if reason:
            counts[reason] += 1
    return counts


def locate(root: Path, spec: dict, dataset_id: str | None = None) -> tuple[str, Path, dict]:
    """找到一个已准备的数据集（默认最近一次 prepare），返回 (标识, 目录, manifest)。
    数据集必须是按当前规格的类别准备的：类别号在训练、评估与部署之间必须是同一套。"""
    models_root = root / paths.DATASETS / spec["id"]
    if dataset_id is None:
        latest = models_root / paths.LATEST
        if not latest.is_file():
            raise TrainError(f"no dataset of {spec['id']} yet: run train prepare {spec['id']} first")
        dataset_id = latest.read_text(encoding="utf-8").strip()
    directory = models_root / dataset_id
    if not dataset_id or "/" in dataset_id or not (directory / paths.MANIFEST).is_file():
        raise TrainError(f"dataset {dataset_id!r} of {spec['id']} does not exist under {models_root}")
    manifest = json.loads((directory / paths.MANIFEST).read_text(encoding="utf-8"))
    if manifest["classes"] != spec["classes"]:
        raise TrainError(f"dataset {dataset_id} was prepared with other classes than {spec['id']}/model.yaml "
                         f"has now: run train prepare {spec['id']} again")
    return dataset_id, directory, manifest


def report(result: Result, spec: dict) -> list[str]:
    """给人看的摘要：各划分的样本、组与实例数，跳过的原因，以及需要注意的分布问题。"""
    task = tasks.get(spec["task"])
    manifest = result.manifest
    state = "reused" if result.reused else "created"
    lines = [f"  note: {note}" for note in result.notes]
    lines.append(f"dataset {result.dataset_id} ({state}): {result.directory}")
    by_split = {name: [s for s in manifest["samples"] if s["split"] == name] for name in split.SPLITS}
    for name, items in by_split.items():
        negatives = sum(1 for s in items if not s["objects"])
        groups = len({s["group"] for s in items})
        lines.append(f"  {name:<5} {len(items):6d} images ({negatives} without objects), {groups} groups")
    counts = {name: task.statistics(items, spec) for name, items in by_split.items()}
    width = max(display_width(c) for c in [*spec["classes"], "class"])
    lines.append(f"  {pad('class', width)}  {'train':>6} {'val':>6} {'test':>6}")
    for name in spec["classes"]:
        lines.append(f"  {pad(name, width)}  " + " ".join(f"{counts[s][name]:6d}" for s in split.SPLITS))
    for (source, reason), count in sorted(result.skipped.items()):
        lines.append(f"  skipped {count} images of {source}: {reason}")
    for reason, count in sorted(result.ignored_shapes.items()):
        lines.append(f"  ignored {count} {reason}")
    missing = [name for name in spec["classes"] if counts["test"][name] == 0]
    if missing:
        listed = ", ".join(missing[:5]) + (f" and {len(missing) - 5} more" if len(missing) > 5 else "")
        lines.append(f"  warning: {len(missing)} classes have no test instances ({listed}); "
                     "their metrics cannot be evaluated")
    if not by_split["val"]:
        lines.append("  warning: the val split is empty; training needs validation images")
    return lines


def display_width(text: str) -> int:
    """终端中的显示宽度：中文等全角字符占两列。"""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def pad(text: str, width: int) -> str:
    return text + " " * (width - display_width(text))
