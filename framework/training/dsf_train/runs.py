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
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from dsf_deploy import models, naming

from . import backends, dataset, paths, spec as specs, tasks
from .errors import TrainError

SCHEMA = 1
RECORD = "run.json"
SPEC_COPY = "model.yaml"
TRAIN = "train"
EVAL = "eval"
METRICS = "metrics.json"
# 这台训练机默认使用的设备（例如 1，或 0,1），由 scripts/train.sh 传入容器。设备属于机器而不是 run：
# 评估与推理按 --device、TRAIN_DEVICE、run 训练时的设备依次取第一个，训练时 model.yaml 与 --option 没有
# 给出 device 才用它（记入 run.json 的 options）。
DEVICE_ENV = "TRAIN_DEVICE"

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
    if machine_device():
        options.setdefault("device", machine_device())
    weights, weights_sha256 = _start_weights(root, spec)
    class_weights = _class_weights(spec, task, manifest)

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
        "options": options, "overrides": overrides, "command": command, "class_weights": class_weights,
        "status": "running", "error": "", "created_at": _now(), "resumed_at": [], "finished_at": "",
        "evaluations": {},
    }
    # 训练可能持续数小时：先告诉人 run 在哪里、中断后怎样继续。
    print(f"run {run}: {run_dir}\n  if it stops early, continue with: train run {spec['id']} --resume {run}",
          flush=True)
    if class_weights:
        print("  class weights: " + ", ".join(f"{name} {weight:.3g}" for name, weight in
                                              zip(spec["classes"], class_weights)), flush=True)
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
    data = backend.data_path(record["task"], directory / paths.VIEW / backend.NAME)
    try:
        best = backend.train(record["task"], data, weights, record["options"], run_dir / TRAIN, resume,
                             root / paths.PRETRAINED, class_weights=record.get("class_weights"))
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


def _class_weights(spec: dict, task, manifest: dict) -> list[float] | None:
    """train.class_weights → 按类别号排列的损失权重；没有设置时 None（不加权）。mapping 中没有的类别权重为 1。"""
    value = spec["train"].get("class_weights")
    if value is None:
        return None
    if value != specs.BALANCED:
        return [float(value.get(name, 1.0)) for name in spec["classes"]]
    counts = task.statistics([s for s in manifest["samples"] if s["split"] == "train"], spec)
    present = [name for name in spec["classes"] if counts[name]]
    total = sum(counts[name] for name in present)
    return [total / (len(present) * counts[name]) if counts[name] else 1.0 for name in spec["classes"]]


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


def machine_device() -> str | None:
    return os.environ.get(DEVICE_ENV, "").strip() or None


def inference_options(options: dict, device: str | None = None) -> dict:
    """评估与推理用的参数：run 的参数，设备换成 device 或这台机器的默认设备（都没有时沿用训练时的设备，
    训练时的 GPU 可能已被别的任务占用）。"""
    device = device or machine_device()
    return {**options, **({"device": device} if device else {})}


def evaluate(root: Path, spec: dict, run: str, dataset_id: str | None = None, device: str | None = None) -> dict:
    """在数据集（默认最近一次 prepare）的测试划分上评估一次完成的 run，写入 eval/<数据集>/metrics.json。
    换用新数据集评估旧 run，是比较新旧模型的方式：测试集只增不换（§6）。device 只改变这次评估用的设备。"""
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
                              backend.data_path(task.NAME, directory / paths.VIEW / backend.NAME), "test",
                              inference_options(record["options"], device),
                              output / backend.NAME, root / paths.PRETRAINED)
    sweep = _thresholds(root, spec, task, backend, run_dir, directory, record, device)
    slices = _slices(root, spec, task, backend, run_dir, manifest, test, record, device, output)
    instances = task.statistics(test, spec)
    classes = []
    for index, name in enumerate(spec["classes"]):
        found = result["classes"].get(index)
        classes.append({"name": name, "instances": instances[name],
                        **{metric: (found or {}).get(metric) for metric in task.METRICS}})
    document = {"schema": SCHEMA, "run": run, "dataset": dataset_id, "split": "test", "images": len(test),
                "groups": len({sample["group"] for sample in test}), "metrics": result["metrics"],
                "classes": classes, **({"task": task.NAME, "thresholds": sweep} if sweep is not None else {}),
                **({"slices": slices} if slices else {}),
                "backend": {"name": backend.NAME, **backend.version()}, "evaluated_at": _now()}
    paths.write_json(output / METRICS, document)
    record["evaluations"][dataset_id] = result["metrics"]
    paths.write_json(run_dir / RECORD, record)
    return document


def _slices(root: Path, spec: dict, task, backend, run_dir: Path, manifest: dict, test: list[dict],
            record: dict, device: str | None, output: Path) -> dict:
    """evaluate.slices 中的分片：按来源时，测试划分有两个以上来源才逐个来源评估（只有一个来源时就是总体）。
    每个来源写一份只含它的测试图片的视图，评估后删除。返回 {"source": {来源: {"images", "metrics", "classes"}}}。"""
    wanted = (spec.get("evaluate") or {}).get("slices") or []
    by_source: dict[str, list[dict]] = {}
    for sample in test:
        by_source.setdefault(sample["source"], []).append(sample)
    if "source" not in wanted or len(by_source) < 2:
        return {}
    result = {}
    for source, samples in sorted(by_source.items()):
        base = output / "slices" / f"source-{source}"
        view = base / paths.VIEW / backend.NAME
        shutil.rmtree(view, ignore_errors=True)
        backend.write_view(task.NAME, {**task.manifest_fields(spec), "classes": record["classes"],
                                       "samples": samples}, root, view, view)
        evaluated = backend.evaluate(task.NAME, run_dir / TRAIN / backend.BEST, backend.data_path(task.NAME, view),
                                     "test", inference_options(record["options"], device), base / backend.NAME,
                                     root / paths.PRETRAINED)
        shutil.rmtree(base / paths.VIEW, ignore_errors=True)
        instances = task.statistics(samples, spec)
        result[source] = {"images": len(samples), "metrics": evaluated["metrics"],
                          "classes": [{"name": name, "instances": instances[name],
                                       **{m: (evaluated["classes"].get(index) or {}).get(m) for m in task.METRICS}}
                                      for index, name in enumerate(spec["classes"]) if instances[name]]}
    return {"source": result}


def _thresholds(root: Path, spec: dict, task, backend, run_dir: Path, directory: Path, record: dict,
                device: str | None) -> list[dict] | None:
    """任务有部署时的置信度阈值（分类的 classifier-threshold）时，在测试划分上逐个裁剪推理并扫描阈值。"""
    if not hasattr(task, "thresholds"):
        return None
    found = backend.crops(directory / paths.VIEW / backend.NAME, "test")
    predictions = backend.predict(task.NAME, run_dir / TRAIN / backend.BEST, [path for path, _ in found],
                                  inference_options(record["options"], device), 0.0, root / paths.PRETRAINED,
                                  agnostic=False)
    outcomes = [(actual, prediction["probs"]) for (_, actual), [prediction] in zip(found, predictions)]
    return task.thresholds(outcomes, spec["classes"])


def evaluate_as(root: Path, spec: dict, run: str, dataset_id: str, mapping: dict[str, str],
                device: str | None = None) -> dict:
    """在按当前类别准备的数据集上评估类别不同的 run（当前发布，§15.6）。mapping 是 run 的类别名 → 规格的类别名
    （共有类别）：测试划分的对象换成 run 的类别号，不在 mapping 中的对象去掉，写一份只含测试划分的视图评估。
    每类指标按规格的类别名给出；结果按数据集与 mapping 缓存在 eval/<数据集>-as-<摘要>/。"""
    run_dir, record = load(root, spec["id"], run)
    if record["status"] != "completed":
        raise TrainError(f"run {run} is {record['status']}; only completed runs can be evaluated")
    dataset_id, _, manifest = dataset.locate(root, spec, dataset_id)
    test = [sample for sample in manifest["samples"] if sample["split"] == "test"]
    if not test:
        raise TrainError(f"the test split of dataset {dataset_id} is empty; nothing to evaluate on")
    task = tasks.get(record["task"])
    backend = backends.get(record["backend"]["name"], task.NAME)
    key = json.dumps(sorted(mapping.items()), ensure_ascii=False)
    output = run_dir / EVAL / f"{dataset_id}-as-{hashlib.sha256(key.encode('utf-8')).hexdigest()[:8]}"
    if (output / METRICS).is_file():
        return json.loads((output / METRICS).read_text(encoding="utf-8"))
    to_run = {spec["classes"].index(new): record["classes"].index(old) for old, new in mapping.items()}
    samples = [{**sample, "objects": [{**item, "class": to_run[item["class"]]} for item in sample["objects"]
                                      if item["class"] in to_run]} for sample in test]
    view = output / paths.VIEW / backend.NAME
    shutil.rmtree(view, ignore_errors=True)
    backend.write_view(task.NAME, {**task.manifest_fields(spec), "classes": record["classes"], "samples": samples},
                       root, view, view)
    result = backend.evaluate(task.NAME, run_dir / TRAIN / backend.BEST, backend.data_path(task.NAME, view), "test",
                              inference_options(record["options"], device),
                              output / backend.NAME, root / paths.PRETRAINED)
    shutil.rmtree(output / paths.VIEW, ignore_errors=True)
    instances = task.statistics(test, spec)
    classes = []
    for old, new in mapping.items():
        found = result["classes"].get(record["classes"].index(old)) or {}
        classes.append({"name": new, "run_name": old, "instances": instances[new],
                        **{metric: found.get(metric) for metric in task.METRICS}})
    document = {"schema": SCHEMA, "run": run, "dataset": dataset_id, "split": "test", "images": len(test),
                "mapping": mapping, "metrics": result["metrics"], "classes": classes,
                "backend": {"name": backend.NAME, **backend.version()}, "evaluated_at": _now()}
    paths.write_json(output / METRICS, document)
    return document


def errors(root: Path, spec: dict, run: str, dataset_id: str | None = None, split_name: str = "test",
           device: str | None = None) -> dict:
    """二级分类模型在一个划分上分错的裁剪，按错误预测的置信度从高到低：置信度高的错误最可能是标错了，
    逐个在 X-AnyLabeling 中核对。写入 eval/<数据集>/errors-<划分>.json。训练划分上的结果偏乐观
    （模型见过这些样本），仍能找出与模型强烈不一致的标注。"""
    task = tasks.get(spec["task"])
    if task.NAME != "classify":
        raise TrainError(f"--errors lists misclassified crops of classify models; {spec['id']} is a {task.NAME} model")
    run_dir, record = load(root, spec["id"], run)
    if record["status"] != "completed":
        raise TrainError(f"run {run} is {record['status']}; only completed runs can be evaluated")
    if record["classes"] != spec["classes"]:
        raise TrainError(f"run {run} was trained with other classes than {spec['id']}/model.yaml has now")
    dataset_id, directory, manifest = dataset.locate(root, spec, dataset_id)
    backend = backends.get(record["backend"]["name"], task.NAME)
    found = backend.crops(directory / paths.VIEW / backend.NAME, split_name)
    if not found:
        raise TrainError(f"the {split_name} split of dataset {dataset_id} has no crops")
    samples = {Path(sample["source"] + "__" + sample["image"]).stem: sample for sample in manifest["samples"]}
    options = inference_options(record["options"], device)
    wrong = []
    predictions = backend.predict(task.NAME, run_dir / TRAIN / backend.BEST, [path for path, _ in found], options,
                                  0.0, root / paths.PRETRAINED, agnostic=False)
    for (path, actual), [prediction] in zip(found, predictions):
        if prediction["class"] == actual:
            continue
        stem, _, number = path.stem.rpartition("__")
        sample = samples[stem]
        item = sample["objects"][int(number)]
        wrong.append({"source": sample["source"], "image": sample["image"], "box": item["box"], "split": split_name,
                      "parent": item.get("parent"), "label": spec["classes"][actual],
                      "predicted": spec["classes"][prediction["class"]], "confidence": round(prediction["score"], 4),
                      "crop": str(path)})
    wrong.sort(key=lambda e: -e["confidence"])
    document = {"schema": SCHEMA, "run": run, "dataset": dataset_id, "split": split_name, "crops": len(found),
                "errors": wrong, "evaluated_at": _now()}
    paths.write_json(run_dir / EVAL / dataset_id / f"errors-{split_name}.json", document)
    return document


def errors_report(document: dict, root: Path, limit: int = 40) -> list[str]:
    errors_ = document["errors"]
    lines = [f"{len(errors_)} of {document['crops']} crops in the {document['split']} split are misclassified by run "
             f"{document['run']} (most confident first; check these labels in X-AnyLabeling):"]
    for entry in errors_[:limit]:
        box = ", ".join(f"{v:.0f}" for v in entry["box"])
        lines.append(f"  labeled {entry['label']}, predicted {entry['predicted']} {entry['confidence']:.3f}  "
                     f"{paths.source_dir(root, entry['source']) / entry['image']}  [{box}] {entry['parent'] or ''}")
    if len(errors_) > limit:
        lines.append(f"  … and {len(errors_) - limit} more")
    return lines


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
    if document.get("thresholds"):
        lines += tasks.get(document["task"]).threshold_lines(document["thresholds"])
    by_source = (document.get("slices") or {}).get("source") or {}
    if by_source:
        width = max(dataset.display_width(name) for name in [*by_source, "source"])
        lines.append("  by source (evaluate.slices):")
        lines.append(f"    {dataset.pad('source', width)}  {'images':>6} " + " ".join(f"{n:>9}" for n in names))
        for source, item in by_source.items():
            values = " ".join(f"{item['metrics'][n]:9.4f}" for n in names)
            lines.append(f"    {dataset.pad(source, width)}  {item['images']:6d} {values}")
    return lines


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
