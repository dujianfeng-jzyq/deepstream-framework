"""train prelabel：用已有模型给来源中还没有标注的图片写预标注（docs/training.md §5.4）。

预标注与人工标注走同一条规则：写出的文件一律 checked: false，人在 X-AnyLabeling 中修正并勾选后才进入
数据集。已有 JSON 的图片永不改动，唯一的例外是按类别预标注（--class，§15.4）：只在这个类别的复核名单中
仍未勾选的标注上追加候选框，已有的形状不动。sources/<来源>/prelabel.json 记录追溯信息：
  batches        每次预标注所用的模型（发布或 run）、权重 sha256、置信度阈值、时间，按类别时还有 class
  images         图像名 → 所属批次与写入的形状；人确认后与之对比，统计修正量
  class_images   图像名 → 类别 → 所属批次与追加的形状（按类别预标注）
它不影响收录。
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from dsf_deploy import models

from . import backends, paths, runs, sources, tasks, xanylabeling
from .errors import TrainError

SCHEMA = 1


@dataclass
class Result:
    model: dict                                   # 所用模型：{"release"?, "run", "weights_sha256"}
    written: int = 0
    boxes: int = 0
    skipped: Counter = field(default_factory=Counter)


def _usable(classes: list[str], spec: dict, label: str | None) -> str:
    """预标注模型的类别能否用于这个规格（按名称）；不能时返回原因。按类别预标注时模型只需认识这个类别，
    否则它的类别都要在规格中：新增类别之后，当前发布的模型仍可预标注旧类别。"""
    if label is not None:
        return "" if label in classes else f"it does not know {label}"
    unknown = [name for name in classes if name not in spec["classes"]]
    return f"it has classes that {spec['id']}/model.yaml does not ({', '.join(unknown)})" if unknown else ""


def weights(repo: Path, root: Path, spec: dict, run: str | None,
            label: str | None = None) -> tuple[Path, dict, dict]:
    """选择预标注的模型，返回 (权重, 模型记录, 该 run 的 run.json)。默认是 release.json 中的当前模型：
    它的权重按 sha256 在训练根目录的 run 中找到；--run 可以用任何完成的 run，包括未发布的。"""
    origin = {}
    if run is None:
        release_path = paths.model_dir(repo, spec["id"]) / models.RELEASE
        if not release_path.is_file():
            raise TrainError(f"{spec['id']} has no published model yet; prelabel with a run: --run <run>")
        release = json.loads(release_path.read_text(encoding="utf-8"))
        run = release["run"]["id"]
        origin["release"] = release["release"]
        reason = _usable(release.get("classes") or [], spec, label)
        if reason:
            raise TrainError(f"the published model of {spec['id']} cannot prelabel: {reason}; "
                             "prelabel with a run: --run <run>")
    try:
        run_dir, record = runs.load(root, spec["id"], run)
    except TrainError:
        if origin:
            raise TrainError(f"the published model was trained by run {run}, which is not in this training root "
                             f"({paths.run_dir(root, spec['id'], run)}); prelabel with a run: --run <run>") from None
        raise
    if record["status"] != "completed":
        raise TrainError(f"run {run} is {record['status']}; only completed runs can prelabel")
    reason = _usable(record["classes"], spec, label)
    if reason:
        raise TrainError(f"run {run} cannot prelabel: {reason}")
    best = run_dir / runs.TRAIN / backends.get(record["backend"]["name"], record["task"]).BEST
    digest = paths.sha256(best)
    if origin and digest != release["run"]["weights_sha256"]:
        raise TrainError(f"the weights of run {run} ({best}) differ from the published model's sha256")
    return best, {**origin, "run": run, "weights_sha256": digest}, record


def prelabel(repo: Path, root: Path, spec: dict, source: str, run: str | None, conf: float,
             device: str | None = None, label: str | None = None) -> Result:
    """给来源中还没有 JSON 的图片写预标注；给出 label 时改为按类别预标注（_prelabel_class）。中断时已写出的
    文件与它们的记录都保留，再次运行只处理剩下的。"""
    if not 0 < conf < 1:
        raise TrainError(f"--conf must be between 0 and 1, not {conf}")
    if label is not None:
        return _prelabel_class(repo, root, spec, source, run, conf, device, label)
    task = tasks.get(spec["task"])
    directory = paths.source_dir(root, source)
    images = sources.read_record(directory)["images"]
    best, model, record = weights(repo, root, spec, run)
    backend = backends.get(record["backend"]["name"], task.NAME)
    options = dict(record["options"])
    if device is not None:
        options["device"] = device
    result = Result(model)
    pending = []
    for name in sorted(images):
        if not (directory / name).is_file():
            result.skipped["image missing"] += 1
        elif xanylabeling.annotation_path(directory / name).is_file():
            result.skipped["already labeled"] += 1
        else:
            pending.append(name)
    if not pending:
        return result

    log_path, log = _log(directory, source)
    batch = len(log["batches"])
    log["batches"].append({"at": _now(), "model": spec["id"], **model, "conf": conf, "images": 0})
    try:
        # 推理期间后端可能切换当前目录：这里只使用绝对路径。NMS 不区分类别：按类别分别抑制时，同一个目标
        # 被相邻候选判成不同类别会留下几乎重合的两个框，人要逐个删掉，修正量统计也会偏低。
        outputs = backend.predict(task.NAME, best, [directory / n for n in pending], options, conf,
                                  root / paths.PRETRAINED, agnostic=True)
        for name, predictions in zip(pending, outputs):
            info = images[name]
            shapes = task.prelabel_shapes(predictions, info["width"], info["height"], record["classes"])
            if not xanylabeling.create(xanylabeling.annotation_path(directory / name), name,
                                       info["width"], info["height"], shapes):
                result.skipped["labeled meanwhile"] += 1  # 推理期间人在工具中保存了这张图
                continue
            log["images"][name] = {"batch": batch, "shapes": [{"label": s["label"], "points": s["points"]}
                                                              for s in shapes]}
            result.written += 1
            result.boxes += len(shapes)
    finally:
        log["batches"][batch]["images"] = result.written
        if result.written:
            paths.write_json(log_path, log)
    return result


def _log(directory: Path, source: str) -> tuple[Path, dict]:
    log_path = directory / paths.PRELABEL_RECORD
    log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.is_file() else \
        {"schema": SCHEMA, "source": source, "batches": [], "images": {}}
    if log.get("schema") != SCHEMA:
        raise TrainError(f"{log_path}: unsupported schema {log.get('schema')!r}")
    return log_path, log


def _prelabel_class(repo: Path, root: Path, spec: dict, source: str, run: str | None, conf: float,
                    device: str | None, label: str) -> Result:
    """按类别预标注（§15.4）：只处理这个类别的复核名单中仍未勾选的图片，在已有标注上追加它的候选框。
    选中时记下标注文件的 sha256，写回前比对：人在工具中保存过的图片跳过。"""
    if label not in spec["classes"]:
        raise TrainError(f"{label} is not a class of {spec['id']}")
    task = tasks.get(spec["task"])
    directory = paths.source_dir(root, source)
    source_record = sources.read_record(directory)
    if label not in sources.class_list(directory, source_record):
        raise TrainError(f"source {source} has no class {label}: add it with train update {spec['id']} add {label}")
    listed = sorted({name for review in source_record.get("reviews") or []
                     if review.get("revoked") and review["class"] == label for name in review["files"]})
    if not listed:
        raise TrainError(f"source {source} has no review list for {label}: --class prelabels only the images "
                         f"that train update {spec['id']} add unchecked")
    best, model, record = weights(repo, root, spec, run, label)
    backend = backends.get(record["backend"]["name"], task.NAME)
    options = dict(record["options"])
    if device is not None:
        options["device"] = device
    result = Result(model)
    pending = []
    for name in listed:
        path = xanylabeling.annotation_path(directory / name)
        if not (directory / name).is_file():
            result.skipped["image missing"] += 1
        elif not path.is_file():
            result.skipped["annotation missing"] += 1
        elif xanylabeling.read(path).checked:
            result.skipped["already checked"] += 1
        else:
            pending.append((name, paths.sha256(path)))
    if not pending:
        return result

    log_path, log = _log(directory, source)
    batch = len(log["batches"])
    log["batches"].append({"at": _now(), "model": spec["id"], **model, "conf": conf, "class": label, "images": 0})
    images = source_record["images"]
    try:
        # 按类别分别做 NMS：不让别的类别的高分框压掉新类别的候选。
        outputs = backend.predict(task.NAME, best, [directory / name for name, _ in pending], options, conf,
                                  root / paths.PRETRAINED, agnostic=False)
        for (name, digest), predictions in zip(pending, outputs):
            path = xanylabeling.annotation_path(directory / name)
            wanted = [item for item in predictions if record["classes"][item["class"]] == label]
            shapes = task.prelabel_shapes(wanted, images[name]["width"], images[name]["height"], record["classes"])
            shapes = task.added_shapes(shapes, xanylabeling.read(path), label)
            if not shapes:
                result.skipped["no new candidates"] += 1
                continue
            if not xanylabeling.append_shapes(path, shapes, digest):
                result.skipped["saved meanwhile"] += 1
                continue
            log.setdefault("class_images", {}).setdefault(name, {})[label] = {
                "batch": batch, "shapes": [{"label": s["label"], "points": s["points"]} for s in shapes]}
            result.written += 1
            result.boxes += len(shapes)
    finally:
        log["batches"][batch]["images"] = result.written
        if result.written:
            paths.write_json(log_path, log)
    return result


def review(root: Path, spec: dict, source: str) -> list[str]:
    """预标注之后人改了多少：每个批次中已检查图片的预测框保留、删除与补充的数量。没有预标注时为空。"""
    directory = paths.source_dir(root, source)
    log_path = directory / paths.PRELABEL_RECORD
    if not log_path.is_file():
        return []
    task = tasks.get(spec["task"])
    log = json.loads(log_path.read_text(encoding="utf-8"))
    totals = [Counter() for _ in log["batches"]]
    written = [(name, None, item) for name, item in log["images"].items()]
    written += [(name, label, item) for name, items in (log.get("class_images") or {}).items()
                for label, item in items.items()]
    for name, label, item in written:
        path = xanylabeling.annotation_path(directory / name)
        if not path.is_file():
            continue
        annotation = xanylabeling.read(path)
        if annotation.checked:
            totals[item["batch"]].update(task.corrections(item["shapes"], annotation, label), checked=1)
    lines = []
    for batch, counts in zip(log["batches"], totals):
        model = f"release {batch['release']}" if batch.get("release") else f"run {batch['run']}"
        if batch.get("class"):
            model += f" for {batch['class']}"
        predicted = counts["kept"] + counts["removed"]
        kept = f"{counts['kept'] / predicted:.0%}" if predicted else "-"
        lines.append(f"  prelabel of {source} by {model} at {batch['at']}: {counts['checked']} of "
                     f"{batch['images']} images checked; {counts['kept']} of {predicted} predicted boxes kept "
                     f"({kept}), {counts['removed']} removed, {counts['added']} added")
    return lines


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
