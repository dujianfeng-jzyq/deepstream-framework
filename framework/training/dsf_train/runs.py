"""train run / train eval：在准备好的数据集上训练，在测试划分上评估（docs/training.md §7）。

runs/<模型>/<run>/
  run.json             规格摘要、数据集、源码版本、后端版本、起点权重、生效的训练参数、状态与评估摘要
  model.yaml           训练时规格的副本
  train/               后端的输出：检查点（best、last）、曲线与日志
  eval/<数据集标识>/    该数据集测试划分上的 metrics.json 与后端的图表
训练完成后 train/ 不再改变；train eval 只在 eval/ 下增加结果。
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from dsf_deploy import models, naming

from . import backends, dataset, paths, tasks
from .errors import TrainError

SCHEMA = 1
RECORD = "run.json"
SPEC_COPY = "model.yaml"
TRAIN = "train"
EVAL = "eval"
METRICS = "metrics.json"
# train.weights 以它开头时指另一次完成的 run 的最佳权重：在已有模型上用新数据继续训练。
RUN_WEIGHTS = "run:"


def start(repo: Path, root: Path, spec: dict, dataset_id: str | None, overrides: dict, command: list[str]) -> str:
    """开始一次训练，返回 run 标识。参数与起点权重的问题在创建 run 目录之前报告。"""
    task = tasks.get(spec["task"])
    backend = backends.get(spec["train"]["backend"], task.NAME)
    problems = backend.check_options(overrides)
    if problems:
        raise TrainError("--option: " + "; ".join(problems))
    dataset_id, directory, manifest = dataset.locate(root, spec, dataset_id)
    if not any(sample["split"] == "val" for sample in manifest["samples"]):
        raise TrainError(f"the val split of dataset {dataset_id} is empty; training needs validation images "
                         "(import more capture groups, or raise data.split.val)")
    options = {**(spec["train"].get("options") or {}), **overrides}
    # 没有给出训练尺寸时按部署契约的输入尺寸训练。
    options.setdefault("imgsz", max(spec["input"]["width"], spec["input"]["height"]))
    weights, weights_sha256 = _start_weights(root, spec)

    spec_text = (paths.model_dir(repo, spec["id"]) / models.SPEC).read_text(encoding="utf-8")
    now = datetime.now(timezone.utc)
    digest = hashlib.sha256(json.dumps([spec_text, dataset_id, overrides], ensure_ascii=False,
                                       sort_keys=True).encode("utf-8")).hexdigest()
    run = f"{now:%Y%m%d-%H%M%S}-{digest[:6]}"
    run_dir = paths.run_dir(root, spec["id"], run)
    try:
        run_dir.mkdir(parents=True)
    except FileExistsError:
        raise TrainError(f"{run_dir} already exists; start the run again") from None
    (run_dir / SPEC_COPY).write_text(spec_text, encoding="utf-8")
    record = {
        "schema": SCHEMA, "model": spec["id"], "run": run, "task": task.NAME, "classes": spec["classes"],
        "spec_sha256": hashlib.sha256(spec_text.encode("utf-8")).hexdigest(),
        "dataset": dataset_id, "source_revision": naming.source_revision(repo),
        "backend": {"name": backend.NAME, **backend.version()},
        "weights": {"start": spec["train"]["weights"], "start_sha256": weights_sha256, "best_sha256": ""},
        "options": options, "overrides": overrides, "command": command,
        "status": "running", "error": "", "created_at": _now(), "resumed_at": [], "finished_at": "",
        "evaluations": {},
    }
    # 训练可能持续数小时：先告诉人 run 在哪里、中断后怎样继续。
    print(f"run {run}: {run_dir}\n  if it stops early, continue with: train run {spec['id']} --resume {run}",
          flush=True)
    _train(root, run_dir, record, backend, directory, weights, resume=False)
    return run


def resume(root: Path, spec: dict, run: str) -> None:
    """从中断或失败时的最近检查点继续；参数与数据集沿用这次 run 的记录。"""
    run_dir, record = load(root, spec["id"], run)
    if record["status"] == "completed":
        raise TrainError(f"run {run} is already completed")
    backend = backends.get(record["backend"]["name"], record["task"])
    last = run_dir / TRAIN / backend.LAST
    if not last.is_file():
        raise TrainError(f"run {run} has no checkpoint to resume from ({last}); start a new run")
    dataset_id, directory, _ = dataset.locate(root, spec, record["dataset"])
    record["resumed_at"].append(_now())
    _train(root, run_dir, record, backend, directory, last, resume=True)


def _train(root: Path, run_dir: Path, record: dict, backend, directory: Path, weights: Path, resume: bool) -> None:
    """运行后端，并把结果如实记入 run.json：中断与失败也留下记录，之后可以续训。"""
    record.update(status="running", error="")
    paths.write_json(run_dir / RECORD, record)
    data = directory / paths.VIEW / backend.NAME / "data.yaml"
    try:
        best = backend.train(record["task"], data, weights, record["options"], run_dir / TRAIN, resume,
                             root / paths.PRETRAINED)
        if not best.is_file():
            raise TrainError(f"the backend finished without writing {best}")
    except KeyboardInterrupt:
        record.update(status="interrupted", finished_at=_now())
        paths.write_json(run_dir / RECORD, record)
        raise
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}", finished_at=_now())
        paths.write_json(run_dir / RECORD, record)
        raise
    if not record["weights"]["start_sha256"] and weights.is_file():
        record["weights"]["start_sha256"] = paths.sha256(weights)  # 预训练权重在训练时才下载
    record["weights"]["best_sha256"] = paths.sha256(best)
    record.update(status="completed", finished_at=_now())
    paths.write_json(run_dir / RECORD, record)


def _start_weights(root: Path, spec: dict) -> tuple[Path, str]:
    """train.weights → (权重路径, sha256)。预训练权重的文件名在 pretrained/ 下查找，没有时由后端下载，
    sha256 在训练后补记。"""
    value = spec["train"]["weights"]
    if value.startswith(RUN_WEIGHTS):
        run = value[len(RUN_WEIGHTS):]
        run_dir, record = load(root, spec["id"], run)
        if record["status"] != "completed":
            raise TrainError(f"train.weights: run {run} is {record['status']}, not completed")
        best = run_dir / TRAIN / backends.get(record["backend"]["name"], record["task"]).BEST
        return best, paths.sha256(best)
    path = Path(value)
    if path.is_absolute():
        if not path.is_file():
            raise TrainError(f"train.weights: {path} does not exist")
        return path, paths.sha256(path)
    if len(path.parts) != 1:
        raise TrainError(f"train.weights: {value!r} must be a pretrained file name, run:<run> or an absolute path")
    path = root / paths.PRETRAINED / value
    return path, paths.sha256(path) if path.is_file() else ""


def load(root: Path, model_id: str, run: str) -> tuple[Path, dict]:
    run_dir = paths.run_dir(root, model_id, run)
    if not (run_dir / RECORD).is_file():
        raise TrainError(f"run {run} of {model_id} does not exist under {run_dir.parent}")
    return run_dir, json.loads((run_dir / RECORD).read_text(encoding="utf-8"))


def evaluate(root: Path, spec: dict, run: str, dataset_id: str | None = None) -> dict:
    """在数据集（默认最近一次 prepare）的测试划分上评估一次完成的 run，写入 eval/<数据集>/metrics.json。
    换用新数据集评估旧 run，是比较新旧模型的方式：测试集只增不换（§6）。"""
    run_dir, record = load(root, spec["id"], run)
    if record["status"] != "completed":
        raise TrainError(f"run {run} is {record['status']}; only completed runs can be evaluated")
    if record["classes"] != spec["classes"]:
        raise TrainError(f"run {run} was trained with other classes than {spec['id']}/model.yaml has now")
    dataset_id, directory, manifest = dataset.locate(root, spec, dataset_id)
    test = [sample for sample in manifest["samples"] if sample["split"] == "test"]
    if not test:
        raise TrainError(f"the test split of dataset {dataset_id} is empty; nothing to evaluate on "
                         "(import more capture groups, or raise data.split.test)")
    task = tasks.get(record["task"])
    backend = backends.get(record["backend"]["name"], task.NAME)
    output = run_dir / EVAL / dataset_id
    result = backend.evaluate(task.NAME, run_dir / TRAIN / backend.BEST,
                              directory / paths.VIEW / backend.NAME / "data.yaml", "test", record["options"],
                              output / backend.NAME, root / paths.PRETRAINED)
    instances = task.statistics(test, spec)
    classes = []
    for index, name in enumerate(spec["classes"]):
        found = result["classes"].get(index)
        classes.append({"name": name, "instances": instances[name],
                        **{metric: (found or {}).get(metric) for metric in task.METRICS}})
    document = {"schema": SCHEMA, "run": run, "dataset": dataset_id, "split": "test", "images": len(test),
                "groups": len({sample["group"] for sample in test}), "metrics": result["metrics"],
                "classes": classes, "backend": {"name": backend.NAME, **backend.version()},
                "evaluated_at": _now()}
    paths.write_json(output / METRICS, document)
    record["evaluations"][dataset_id] = result["metrics"]
    paths.write_json(run_dir / RECORD, record)
    return document


def test_split_empty(root: Path, spec: dict, run: str) -> bool:
    _, record = load(root, spec["id"], run)
    _, _, manifest = dataset.locate(root, spec, record["dataset"])
    return not any(sample["split"] == "test" for sample in manifest["samples"])


def report(document: dict) -> list[str]:
    """评估结果的摘要：总体指标与每类指标；测试划分中没有实例的类别无法评估，单独列出。"""
    metrics = document["metrics"]
    names = list(metrics)
    lines = [f"evaluated run {document['run']} on the test split of dataset {document['dataset']} "
             f"({document['images']} images, {document['groups']} groups)",
             "  " + "  ".join(f"{name} {metrics[name]:.4f}" for name in names)]
    width = max(dataset.display_width(c["name"]) for c in [*document["classes"], {"name": "class"}])
    lines.append(f"  {dataset.pad('class', width)}  {'inst':>5} " + " ".join(f"{name:>9}" for name in names))
    missing = []
    for item in document["classes"]:
        if not item["instances"]:
            missing.append(item["name"])
            continue
        values = " ".join(f"{item[name]:9.4f}" if item[name] is not None else f"{'-':>9}" for name in names)
        lines.append(f"  {dataset.pad(item['name'], width)}  {item['instances']:5d} {values}")
    if missing:
        lines.append(f"  not evaluated (no test instances): {', '.join(missing)}")
    return lines


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
