"""train prelabel：用已有模型给来源中还没有标注的图片写预标注（docs/training.md §5.4）。

预标注与人工标注走同一条规则：写出的文件一律 checked: false，人在 X-AnyLabeling 中修正并勾选后才进入
数据集。已有 JSON 的图片永不改动。sources/<来源>/prelabel.json 记录追溯信息：
  batches   每次预标注所用的模型（发布或 run）、权重 sha256、置信度阈值与时间
  images    图像名 → 所属批次与写入的形状；人确认后与之对比，统计修正量
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


def weights(repo: Path, root: Path, spec: dict, run: str | None) -> tuple[Path, dict, dict]:
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
        if release.get("classes") != spec["classes"]:
            raise TrainError(f"the published model of {spec['id']} has other classes than model.yaml; "
                             "prelabel with a run trained on the current classes: --run <run>")
    try:
        run_dir, record = runs.load(root, spec["id"], run)
    except TrainError:
        if origin:
            raise TrainError(f"the published model was trained by run {run}, which is not in this training root "
                             f"({paths.run_dir(root, spec['id'], run)}); prelabel with a run: --run <run>") from None
        raise
    if record["status"] != "completed":
        raise TrainError(f"run {run} is {record['status']}; only completed runs can prelabel")
    if record["classes"] != spec["classes"]:
        raise TrainError(f"run {run} was trained with other classes than {spec['id']}/model.yaml has now")
    best = run_dir / runs.TRAIN / backends.get(record["backend"]["name"], record["task"]).BEST
    digest = paths.sha256(best)
    if origin and digest != release["run"]["weights_sha256"]:
        raise TrainError(f"the weights of run {run} ({best}) differ from the published model's sha256")
    return best, {**origin, "run": run, "weights_sha256": digest}, record


def prelabel(repo: Path, root: Path, spec: dict, source: str, run: str | None, conf: float,
             device: str | None = None) -> Result:
    """给来源中还没有 JSON 的图片写预标注。中断时已写出的文件与它们的记录都保留，再次运行只处理剩下的。"""
    if not 0 < conf < 1:
        raise TrainError(f"--conf must be between 0 and 1, not {conf}")
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

    log_path = directory / paths.PRELABEL_RECORD
    log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.is_file() else \
        {"schema": SCHEMA, "source": source, "batches": [], "images": {}}
    if log.get("schema") != SCHEMA:
        raise TrainError(f"{log_path}: unsupported schema {log.get('schema')!r}")
    batch = len(log["batches"])
    log["batches"].append({"at": _now(), "model": spec["id"], **model, "conf": conf, "images": 0})
    try:
        # 推理期间后端可能切换当前目录：这里只使用绝对路径。
        for name, predictions in zip(pending, backend.predict(task.NAME, best, [directory / n for n in pending],
                                                              options, conf, root / paths.PRETRAINED)):
            info = images[name]
            shapes = task.prelabel_shapes(predictions, info["width"], info["height"], spec)
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


def review(root: Path, spec: dict, source: str) -> list[str]:
    """预标注之后人改了多少：每个批次中已检查图片的预测框保留、删除与补充的数量。没有预标注时为空。"""
    directory = paths.source_dir(root, source)
    log_path = directory / paths.PRELABEL_RECORD
    if not log_path.is_file():
        return []
    task = tasks.get(spec["task"])
    log = json.loads(log_path.read_text(encoding="utf-8"))
    totals = [Counter() for _ in log["batches"]]
    for name, item in log["images"].items():
        path = xanylabeling.annotation_path(directory / name)
        if not path.is_file():
            continue
        annotation = xanylabeling.read(path)
        if annotation.checked:
            totals[item["batch"]].update(task.corrections(item["shapes"], annotation), checked=1)
    lines = []
    for batch, counts in zip(log["batches"], totals):
        model = f"release {batch['release']}" if batch.get("release") else f"run {batch['run']}"
        predicted = counts["kept"] + counts["removed"]
        kept = f"{counts['kept'] / predicted:.0%}" if predicted else "-"
        lines.append(f"  prelabel of {source} by {model} at {batch['at']}: {counts['checked']} of "
                     f"{batch['images']} images checked; {counts['kept']} of {predicted} predicted boxes kept "
                     f"({kept}), {counts['removed']} removed, {counts['added']} added")
    return lines


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
