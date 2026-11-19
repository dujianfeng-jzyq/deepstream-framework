"""train publish：门槛 → 导出 → 验证 → 写入模型目录（docs/training.md §8）。顺序不可调换：

1. 门槛：evaluate.gate.min 的下限，以及与 release.json 中当前模型在同一测试划分上的退化比较（类别变化后
   在共有类别上比较，§15.6）。
2. 导出：后端导出 ONNX 并在输出端追加到 output_layout 的转换；labels.txt 按任务的格式生成。
3. 验证：ONNX 签名对照输入尺寸与输出布局（与打包时同一套规则）；原生推理与 onnxruntime 在测试样本上
   逐行比较，不一致即失败。
4. 写入：先在暂存目录组装完整的新模型目录并通过打包时的模型校验，再替换 models/<id>/ 中变化的文件，
   release.json 最后写入。前三步与组装中的任何失败都不改动模型目录。

导出物与检查结果留在 runs/<模型>/<run>/export/；提交由人在审阅 release.json 后完成。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from dsf_deploy import layout, models, naming

from . import backends, dataset, lock, media, onnxio, paths, runs, sources, tasks
from .errors import TrainError

EXPORT = "export"
EXPORT_RECORD = "export.json"
ONNX = "model.onnx"
LABELS = "labels.txt"
# 一致性比较用的测试样本数：按 sha256 排序取前几张，同一数据集每次取同一批。
CONSISTENCY_IMAGES = 16
_ENGINE_KEY = re.compile(r"^(\s*model-engine-file\s*=\s*)([^#\s]+)(.*)$")


@dataclass
class Result:
    release: str
    directory: Path
    gate: dict
    metrics: dict
    consistency: dict
    engines: list[tuple[str, str, str]] = field(default_factory=list)  # (配置, 旧路径, 新路径)
    changed: list[str] = field(default_factory=list)


def publish(repo: Path, root: Path, spec: dict, run: str, dataset_id: str | None = None,
            accept_metrics: bool = False, device: str | None = None) -> Result:
    task = tasks.get(spec["task"])
    run_dir, record = runs.load(root, spec["id"], run)
    if record["status"] != "completed":
        raise TrainError(f"run {run} is {record['status']}; only completed runs can be published")
    if record["classes"] != spec["classes"]:
        raise TrainError(f"run {run} was trained with other classes than {spec['id']}/model.yaml has now")
    backend = backends.get(record["backend"]["name"], task.NAME)
    output_layout = spec["output_layout"]
    if output_layout not in task.LAYOUTS or output_layout not in backend.LAYOUTS.get(task.NAME, ()):
        raise TrainError(f"output_layout {output_layout} cannot be exported for task {task.NAME} by backend "
                         f"{backend.NAME} (supported: {', '.join(backend.LAYOUTS.get(task.NAME, ())) or 'none'})")
    best = run_dir / runs.TRAIN / backend.BEST
    if not best.is_file() or paths.sha256(best) != record["weights"]["best_sha256"]:
        raise TrainError(f"{best} is missing or differs from the best weights recorded in run {run}")
    directory = paths.model_dir(repo, spec["id"])
    current = _current_release(directory)
    # 预检：nvinfer 配置中的 engine 路径能否改成按发布命名，在花时间评估与导出之前报告。
    _engine_configs(directory, spec, current, "preflight")

    # 1. 门槛
    dataset_id, _, _ = dataset.locate(root, spec, dataset_id)
    evaluation = _metrics(root, spec, run, dataset_id, device)
    gate = check_gate(repo, root, spec, evaluation, current, accept_metrics, device)

    # 2. 导出
    settings = spec["export"]
    width, height = spec["input"]["width"], spec["input"]["height"]
    export_dir = run_dir / EXPORT
    shutil.rmtree(export_dir, ignore_errors=True)
    export_dir.mkdir(parents=True)
    try:
        backend.export(task.NAME, best, output_layout, (width, height), settings, export_dir / ONNX,
                       root / paths.PRETRAINED)
    except ValueError as error:
        raise TrainError(f"export failed: {error}") from error
    (export_dir / LABELS).write_text(task.labels_text(spec["classes"]), encoding="utf-8")

    # 3. 验证
    onnx = {"opset": settings["opset"], "batch": settings["batch"], "dynamic": settings["dynamic"],
            **onnxio.signature(export_dir / ONNX)}
    problems = models.check_signature(spec, onnx)
    consistency = {"passed": False, "reason": "not compared: " + "; ".join(problems)} if problems else \
        _consistency(root, spec, task, backend, best, export_dir / ONNX, dataset_id, settings)
    paths.write_json(export_dir / EXPORT_RECORD, {
        "run": run, "dataset": dataset_id, "settings": settings, "onnx": onnx, "consistency": consistency,
        "exported_at": _now()})
    if problems:
        raise TrainError("the exported ONNX does not match model.yaml:\n  " + "\n  ".join(problems))
    if not consistency["passed"]:
        raise TrainError(f"the exported ONNX does not reproduce the backend's own inference: "
                         f"{_consistency_text(consistency)}; see {export_dir / EXPORT_RECORD}")

    # 4. 写入
    onnx_sha256 = paths.sha256(export_dir / ONNX)
    if current and (current.get("artifacts") or {}).get(f"{models.ARTIFACTS}/{ONNX}") == onnx_sha256:
        # 新的发布标识会改变 engine 路径，让目标机为同一个模型重建 engine。
        raise TrainError(f"the exported ONNX is identical to the published release {current['release']}; "
                         "nothing to publish")
    release_id = f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{onnx_sha256[:6]}"
    configs = _engine_configs(directory, spec, current, release_id)
    release = {
        "schema": models.SCHEMA, "model": spec["id"], "release": release_id, "published_at": _now(),
        "published_from": naming.source_revision(repo),
        "run": {"id": run, "dataset": record["dataset"], "source_revision": record["source_revision"],
                "backend": record["backend"], "weights_sha256": record["weights"]["best_sha256"]},
        "evaluated_on": dataset_id, "metrics": evaluation["metrics"],
        "class_metrics": [item for item in evaluation["classes"] if item["instances"]],
        "gate": gate, "classes": spec["classes"], "input": spec["input"], "output_layout": output_layout,
        **({"class_changes": class_changes(repo, spec, current)} if current and current.get("classes") !=
           spec["classes"] else {}),
        "training_data": training_data(root, spec, record["dataset"]),
        "onnx": onnx, "consistency": consistency,
        "artifacts": {f"{models.ARTIFACTS}/{ONNX}": onnx_sha256,
                      f"{models.ARTIFACTS}/{LABELS}": paths.sha256(export_dir / LABELS)},
    }
    changed = _install(directory, spec, release, export_dir, {name: text for name, (text, _, _) in configs.items()})
    return Result(release_id, directory, gate, evaluation["metrics"], consistency,
                  [(name, old, new) for name, (_, old, new) in configs.items()], changed)


def candidates(root: Path, spec: dict, dataset_id: str | None = None,
               repo: Path | None = None) -> list[str]:
    """不给 run 时列出可以发布的 run，不发布任何东西：发布是人的判断，最新的 run 常常只是试跑。
    指标取各 run 在数据集（默认最近一次 prepare）测试划分上的评估；覆盖参数让试跑一眼可见。"""
    task = tasks.get(spec["task"])
    try:
        dataset_id, _, _ = dataset.locate(root, spec, dataset_id)
    except TrainError:
        if dataset_id is not None:
            raise
        dataset_id = None
    current = _current_release(paths.model_dir(repo, spec["id"])) if repo is not None else None
    published = ((current or {}).get("run") or {}).get("id")
    directory = root / paths.RUNS / spec["id"]
    records = []
    for path in sorted(directory.glob(f"*/{runs.RECORD}")) if directory.is_dir() else []:
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except ValueError:
            continue
    completed = [r for r in records if r.get("status") == "completed"]
    others = [r for r in records if r.get("status") != "completed"]
    if not completed:
        lines = [f"no completed runs of {spec['id']} under {directory}; train one with: train run {spec['id']}"]
    else:
        where = f"the test split of dataset {dataset_id}" if dataset_id else "no prepared dataset"
        names = list(task.METRICS)
        width = max(len(r["run"]) for r in completed) + 2
        lines = [f"completed runs of {spec['id']} ({where}); nothing was published:",
                 f"    {'run':<{width}}" + "".join(f"{n:>10}" for n in names) + "  finished          overrides"]
        unevaluated = False
        for record in sorted(completed, key=lambda r: r.get("finished_at", ""), reverse=True):
            metrics = (record.get("evaluations") or {}).get(dataset_id) if dataset_id else None
            unevaluated = unevaluated or (dataset_id is not None and metrics is None)
            values = "".join(f"{metrics[n]:>10.4f}" if metrics and metrics.get(n) is not None else f"{'-':>10}"
                             for n in names)
            mark = "* " if record["run"] == published else "  "
            overrides = " ".join(f"{k}={v}" for k, v in (record.get("overrides") or {}).items()) or "-"
            if record.get("classes") != spec["classes"]:
                overrides += "  (other classes than model.yaml: cannot be published)"
            finished = (record.get("finished_at") or "")[:16].replace("T", " ")
            lines.append(f"  {mark}{record['run']:<{width}}{values}  {finished:<16}  {overrides}")
        if published and any(record["run"] == published for record in completed):
            lines.append(f"  * the run of the published release {current['release']}")
        if unevaluated:
            lines.append(f"  - not evaluated on this dataset yet: train eval {spec['id']} <run>")
        lines.append(f"publish one with: train publish {spec['id']} <run>")
    if others:
        states = ", ".join(f"{r.get('run')} ({r.get('status')})" for r in others)
        lines.append(f"  not publishable: {states}")
    return lines


def check_gate(repo: Path, root: Path, spec: dict, evaluation: dict, current: dict | None, accept_metrics: bool,
               device: str | None = None) -> dict:
    """发布门槛。未通过时报错并列出原因；--accept-metrics 只放行“没有可对照的东西”，不放行未达标的指标。"""
    task = tasks.get(spec["task"])
    gate = (spec.get("evaluate") or {}).get("gate") or {}
    metrics = evaluation["metrics"]
    result = {"dataset": evaluation["dataset"], "min": {}, "regression": None, "accepted_without_comparison": False}
    failures = []
    for name, bound in (gate.get("min") or {}).items():
        value = metrics[name]
        passed = _better_or_equal(task, name, value, bound)
        result["min"][name] = {"bound": bound, "value": value, "passed": passed}
        if not passed:
            failures.append(f"{name} {value:.4f} does not reach evaluate.gate.min {bound}")

    regression = gate.get("regression")
    unavailable = ""
    if regression is None:
        result["regression"] = {"skipped": "evaluate.gate.regression is not configured"}
    elif current is None:
        result["regression"] = {"skipped": "there is no published model yet"}
    else:
        unavailable = _comparable(root, spec, current)
        metric, max_drop = regression["metric"], regression["max_drop"]
        old_run = current["run"]["id"]
        shared = {}
        if not unavailable and current["classes"] == spec["classes"]:
            old = _metrics(root, spec, old_run, evaluation["dataset"], device)["metrics"][metric]
            new = metrics[metric]
        elif not unavailable:
            # 类别变了：两个模型都只在共有类别上比较，取这些类别的每类指标的平均（§15.6）。
            mapping, excluded = class_mapping(repo, root, spec, current)
            previous = runs.evaluate_as(root, spec, old_run, evaluation["dataset"], mapping, device)
            old_values = {item["name"]: item[metric] for item in previous["classes"]}
            new_values = {item["name"]: item[metric] for item in evaluation["classes"] if item["instances"]}
            names = [n for n in spec["classes"] if old_values.get(n) is not None and new_values.get(n) is not None]
            if names:
                old = sum(old_values[n] for n in names) / len(names)
                new = sum(new_values[n] for n in names) / len(names)
                shared = {"classes": names, "excluded": excluded,
                          "renamed": {o: n for o, n in mapping.items() if o != n}}
            else:
                unavailable = "no class it shares with model.yaml has test instances"
        if unavailable:
            result["regression"] = {"skipped": unavailable}
        else:
            drop = old - new if task.METRICS[metric] == "higher" else new - old
            passed = drop <= max_drop
            result["regression"] = {"metric": metric, "max_drop": max_drop, "release": current["release"],
                                    "run": old_run, "published": old, "value": new, "drop": drop,
                                    "passed": passed, **shared}
            if not passed:
                over = f" (mean over the {len(shared['classes'])} shared classes)" if shared else ""
                failures.append(f"{metric}{over} {new:.4f} is {drop:.4f} below the published release "
                                f"{current['release']} ({old:.4f}) on the same test split; "
                                f"evaluate.gate.regression allows {max_drop}")
    if failures:
        raise TrainError("the run does not pass the publish gate:\n  " + "\n  ".join(failures))
    compared = bool(result["min"]) or "passed" in result["regression"]
    if unavailable and not accept_metrics:
        raise TrainError(f"cannot compare with the published release {current['release']}: {unavailable}. "
                         "Review the metrics and pass --accept-metrics to publish without the comparison")
    if not compared and not accept_metrics:
        raise TrainError("nothing to check the metrics against: evaluate.gate.min is empty and there is no "
                         "published model to compare with. Review the metrics and pass --accept-metrics "
                         "(or set evaluate.gate.min)")
    result["accepted_without_comparison"] = bool(unavailable) or not compared
    return result


def _better_or_equal(task, name: str, value: float, bound: float) -> bool:
    return value >= bound if task.METRICS[name] == "higher" else value <= bound


def _comparable(root: Path, spec: dict, current: dict) -> str:
    """当前发布能否在本训练根目录中重新评估；不能时返回原因。类别不同也能比较（在共有类别上）。"""
    run = (current.get("run") or {}).get("id") or ""
    try:
        run_dir, record = runs.load(root, spec["id"], run)
    except TrainError:
        return f"its run {run!r} is not in this training root"
    if record["status"] != "completed" or record["classes"] != current.get("classes"):
        return f"its run {run} is {record['status']} or has other classes than the release"
    best = run_dir / runs.TRAIN / backends.get(record["backend"]["name"], record["task"]).BEST
    if not best.is_file() or paths.sha256(best) != current["run"].get("weights_sha256"):
        return f"the weights of its run {run} differ from the published sha256"
    return ""


def class_mapping(repo: Path, root: Path, spec: dict, current: dict) -> tuple[dict[str, str], list[str]]:
    """当前发布的类别名 → 规格中的类别名（共有类别，按发布以来的改名对应），以及不计入比较的类别：
    add --from A 新增的类别以前被标成 A，旧模型会把新类别的目标报成 A，比较对它不公平。"""
    mapping = _survivors(repo, spec, current)
    added = set(spec["classes"]) - set(mapping.values())
    excluded = set()
    for source in spec["data"]["sources"]:
        record = sources.read_record(paths.source_dir(root, source))
        excluded |= {r["from"] for r in record.get("reviews") or [] if r.get("from") and r["class"] in added}
    return {old: new for old, new in mapping.items() if new not in excluded}, sorted(excluded)


def class_changes(repo: Path, spec: dict, current: dict) -> dict:
    """相对当前发布的类别变化：改名（按 train update 的记录）、删除与新增。"""
    survivors = _survivors(repo, spec, current)
    return {"renamed": {old: new for old, new in survivors.items() if old != new},
            "removed": [old for old in current["classes"] if old not in survivors],
            "added": [name for name in spec["classes"] if name not in survivors.values()]}


def _survivors(repo: Path, spec: dict, current: dict) -> dict[str, str]:
    """当前发布中仍在规格里的类别：旧名 → 现在的名称。改过名的按记录对应；名称被另一个类别改名占用的旧类别
    不算留下（它已被删除，同名的是另一个类别）。"""
    names = lock.renamed(repo, spec["id"])
    result = {}
    for old in current["classes"]:
        new = names.get(old, old if old not in names.values() else None)
        if new in spec["classes"]:
            result[old] = new
    return result


def training_data(root: Path, spec: dict, dataset_id: str) -> dict:
    """训练所用数据集的按来源统计（§8）。收入的图片、负样本与每类实例数从 manifest 计算；跳过数与未完成的
    复核名单是发布时来源的状态：它们说明这次发布时还有哪些素材没有进入。"""
    task = tasks.get(spec["task"])
    _, _, manifest = dataset.locate(root, spec, dataset_id)
    names = list(dict.fromkeys([*spec["data"]["sources"], *(s["source"] for s in manifest["samples"])]))
    result = {}
    for source in names:
        samples = [s for s in manifest["samples"] if s["source"] == source]
        counts = task.statistics(samples, spec)
        entry = {"images": len(samples), "negatives": sum(1 for s in samples if not s["objects"]),
                 "instances": {name: counts[name] for name in spec["classes"]}}
        directory = paths.source_dir(root, source)
        if (directory / paths.SOURCE_RECORD).is_file():
            entry["skipped"] = dict(sorted(dataset.skipped_now(root, source).items()))
            entry["reviews"] = [{"class": review["class"], "from": review.get("from"), "at": review["at"],
                                 "checked": checked, "total": total}
                                for review, checked, total in sources.review_progress(
                                    directory, sources.read_record(directory)) if checked < total]
        result[source] = entry
    return result


def _metrics(root: Path, spec: dict, run: str, dataset_id: str, device: str | None) -> dict:
    """run 在数据集测试划分上的评估；已有结果直接使用（权重与数据集都不可变，结果相同）。"""
    run_dir, _ = runs.load(root, spec["id"], run)
    path = run_dir / runs.EVAL / dataset_id / runs.METRICS
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return runs.evaluate(root, spec, run, dataset_id, device)


def _consistency(root: Path, spec: dict, task, backend, best: Path, onnx_path: Path, dataset_id: str,
                 settings: dict) -> dict:
    """原生推理与 ONNX 在同一批测试样本上的输出比较。固定 batch 时按 batch 分组，最后一组用最后一张补齐。"""
    import numpy
    _, _, manifest = dataset.locate(root, spec, dataset_id)
    samples = sorted((s for s in manifest["samples"] if s["split"] == "test"), key=lambda s: s["sha256"])
    samples = samples[:CONSISTENCY_IMAGES]
    width, height = spec["input"]["width"], spec["input"]["height"]
    images = [media.letterbox(paths.source_dir(root, s["source"]) / s["image"], width, height) for s in samples]
    size = len(images) if settings["dynamic"] else settings["batch"]
    native, exported = [], []
    for start in range(0, len(images), size):
        chunk = images[start:start + size]
        chunk += [chunk[-1]] * (size - len(chunk))
        batch = numpy.stack(chunk).astype(numpy.float32)
        native.append(backend.reference(task.NAME, best, spec["output_layout"], batch))
        exported.append(onnxio.run(onnx_path, batch))
    summary = task.compare(spec["output_layout"], numpy.concatenate(native), numpy.concatenate(exported))
    return {**summary, "images": len(images)}


def _consistency_text(summary: dict) -> str:
    if "reason" in summary:
        return summary["reason"]
    return (f"{summary['candidates']} candidates compared on {summary.get('images', '?')} images, max box "
            f"difference {summary['max_box_diff']:.4f} px, max confidence difference "
            f"{summary['max_conf_diff']:.5f}, {summary['class_mismatches']} class mismatches")


def _current_release(directory: Path) -> dict | None:
    path = directory / models.RELEASE
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError as error:
        raise TrainError(f"{path}: {error}") from error


def _engine_configs(directory: Path, spec: dict, current: dict | None,
                    release_id: str) -> dict[str, tuple[str, str, str]]:
    """把各 nvinfer 配置的 model-engine-file 改为 /data/engines/<id>/<发布>/<文件>，返回 {配置: (新内容, 旧, 新)}。
    新旧发布的 engine 并存，回滚不需要重建。还没有发布段的路径插入一段，已有的替换当前发布的那一段；
    看不出哪一段是发布标识时报错，而不是猜。"""
    prefix = f"{layout.MOUNTS['CONTAINER_DATA_PATH']}/engines/{spec['id']}/"
    previous = (current or {}).get("release")
    updates = {}
    for config in sorted(directory.glob("*.txt")):
        lines = config.read_text(encoding="utf-8").splitlines(keepends=True)
        section = None
        for index, line in enumerate(lines):
            stripped = line.split("#", 1)[0].strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                section = stripped[1:-1].strip()
                continue
            match = _ENGINE_KEY.match(line.rstrip("\n"))
            if section != "property" or not match:
                continue
            old = match.group(2)
            parts = old[len(prefix):].split("/") if old.startswith(prefix) else []
            if not parts or ".." in parts or not all(parts):
                raise TrainError(f"{config.name}: model-engine-file must be a file under {prefix}, got {old}")
            if len(parts) == 1:
                parts = [release_id, *parts]
            elif previous and parts[0] == previous:
                parts = [release_id, *parts[1:]]
            else:
                raise TrainError(f"{config.name}: cannot tell which directory of model-engine-file={old} is the "
                                 f"release (the published release is {previous or 'none'}); use "
                                 f"{prefix}<file> and publish names the release directory")
            new = prefix + "/".join(parts)
            lines[index] = f"{match.group(1)}{new}{match.group(3)}" + ("\n" if line.endswith("\n") else "")
            updates[config.name] = ("".join(lines), old, new)
    return updates


def _install(directory: Path, spec: dict, release: dict, export_dir: Path, configs: dict[str, str]) -> list[str]:
    """先组装完整的新模型目录并通过打包时的模型校验，再把变化的文件逐个原子替换进模型目录，release.json 最后
    写入：校验失败时模型目录不变。返回变化的文件。"""
    staging_root = Path(tempfile.mkdtemp(prefix=".publish-", dir=export_dir))
    try:
        staged = staging_root / spec["id"]
        shutil.copytree(directory, staged, ignore=shutil.ignore_patterns(models.ARTIFACTS, models.RELEASE))
        (staged / models.ARTIFACTS).mkdir()
        for name in (ONNX, LABELS):
            shutil.copy2(export_dir / name, staged / models.ARTIFACTS / name)
        for name, text in configs.items():
            (staged / name).write_text(text, encoding="utf-8")
        paths.write_json(staged / models.RELEASE, release)
        renames_cleared = lock.clear_renamed(staged)  # 改名记录只覆盖到这次发布
        problems = models.check_model(staged)
        if problems:
            raise TrainError(f"the new contents of {directory} fail the model check, nothing was changed:\n  " +
                             "\n  ".join(problems))
        targets = [f"{models.ARTIFACTS}/{ONNX}", f"{models.ARTIFACTS}/{LABELS}", *configs,
                   *([lock.FILE] if renames_cleared else []), models.RELEASE]
        changed = []
        for relative in targets:
            source, target = staged / relative, directory / relative
            if target.is_file() and paths.sha256(target) == paths.sha256(source):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.publish")
            shutil.copy2(source, temporary)
            os.replace(temporary, target)
            changed.append(relative)
        published = {directory / relative for relative in release["artifacts"]}
        for path in sorted(p for p in (directory / models.ARTIFACTS).rglob("*") if p.is_file()):
            if path not in published:  # 上一次发布留下、本次不再发布的产物
                path.unlink()
                changed.append(str(path.relative_to(directory)))
        return changed
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


def report(result: Result) -> list[str]:
    gate = result.gate
    lines = [f"published release {result.release} to {result.directory}",
             "  " + "  ".join(f"{name} {value:.4f}" for name, value in result.metrics.items()) +
             f"  (test split of dataset {gate['dataset']})"]
    for name, item in gate["min"].items():
        lines.append(f"  gate min: {name} {item['value']:.4f} >= {item['bound']}")
    regression = gate["regression"]
    if "passed" in regression:
        lines.append(f"  gate regression: {regression['metric']} {regression['value']:.4f} vs release "
                     f"{regression['release']} {regression['published']:.4f} (drop {regression['drop']:.4f}, "
                     f"allowed {regression['max_drop']})")
    else:
        lines.append(f"  gate regression skipped: {regression['skipped']}")
    if regression.get("classes"):
        excluded = f"; not compared: {', '.join(regression['excluded'])}" if regression.get("excluded") else ""
        lines.append(f"  compared on the shared classes: {', '.join(regression['classes'])}{excluded}")
    if gate["accepted_without_comparison"]:
        lines.append("  metrics accepted without comparison (--accept-metrics)")
    lines.append(f"  export consistency: {_consistency_text(result.consistency)}")
    for config, old, new in result.engines:
        lines.append(f"  {config}: model-engine-file {old} -> {new}")
    if not result.engines:
        lines.append(f"  note: no nvinfer config in {result.directory} yet; write one with "
                     f"onnx-file={models.ARTIFACTS}/{ONNX} and labelfile-path={models.ARTIFACTS}/{LABELS}")
    lines.append(f"  changed: {', '.join(result.changed) or 'nothing'}")
    lines.append(f"  review {models.RELEASE} and the diff, then commit models/{result.directory.name}/")
    return lines


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
