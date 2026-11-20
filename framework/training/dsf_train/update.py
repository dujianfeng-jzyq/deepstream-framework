"""train update <id> <操作>：修改模型的类别，连同来源与其他模型（docs/training.md §15.3）。

每次运行先完成全部检查（冲突、可写），再写入。写入顺序保证中断后重新执行同一命令即可完成：
标注文件 → 来源记录与 labels.txt → sources.lock.json → 其他模型的 model.yaml → 本模型的 model.yaml（add 例外，
见 add）。标注先改，
来源记录还写着旧名称时 prepare 遇到新名称报“未知类别”，不会在不一致的状态下训练；重新执行时已改过的文件
没有可改的形状，跳过；本模型的规格最后改，它改完即全部完成。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import lock, paths, sources, spec as specs, xanylabeling
from .errors import TrainError


@dataclass
class Result:
    lines: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)  # 标注与 labels.txt 被改写的来源


def _models(repo: Path) -> dict[str, dict]:
    """仓库中全部模型的规格。任何一个规格不成立都报错：影响范围算不全时不能改写共用的来源。"""
    return specs.load_all(repo)


def _labels(spec: dict) -> list[str]:
    """模型的类别中哪些是来源中的标签：二级模型的类别是框上属性的值，不是标签。"""
    return [] if spec.get("cascade") else spec["classes"]


def _cascade_lines(models: dict[str, dict], parents: set[str], name: str, renamed_to: str | None = None) -> list[str]:
    """触发类别中有 name 的二级模型：cascade.classes 由人改写（自动改写尚未实现），父模型类别号的变化由打包时
    的级联校验对照 operate-on-class-ids。"""
    lines = []
    for model_id, spec in sorted(models.items()):
        cascade = spec.get("cascade") or {}
        if cascade.get("parent") in parents and name in cascade.get("classes", []):
            change = f"rename it to {renamed_to}" if renamed_to else "remove it"
            lines.append(f"  second-stage model {model_id} triggers on {name}: {change} in its cascade.classes "
                         "(not rewritten automatically) and check operate-on-class-ids in its nvinfer config")
    return lines


def _users(models: dict[str, dict], names: set[str]) -> list[str]:
    """依赖这些来源中任何一个的模型。"""
    return [model_id for model_id, spec in models.items() if names & set(spec["data"]["sources"])]


def _require_confirmed(repo: Path, root: Path, spec: dict, translate) -> None:
    """操作的起点必须是确认过的状态：否则改写后的差异里混着人还没看过的变化。记录按这次操作换算后与来源
    相同也算：上一次执行在改写来源之后中断了。"""
    recorded = lock.read(repo, spec["id"])
    now = lock.current(root, spec)
    stale = [source for source, names in now.items()
             if source not in recorded or names not in (recorded[source], translate(recorded[source]))]
    if stale:
        raise TrainError(f"the classes of {', '.join(stale)} are not confirmed for {spec['id']}; review them "
                         f"with train update {spec['id']} and confirm with --confirm first")


def _annotations(root: Path, source: str) -> list[Path]:
    directory = paths.source_dir(root, source)
    record = sources.read_record(directory)
    return [path for path in (xanylabeling.annotation_path(directory / name) for name in sorted(record["images"]))
            if path.is_file()]


def _check_writable(files: list[Path]) -> None:
    denied = [str(path) for path in files if not os.access(path, os.W_OK) or not os.access(path.parent, os.W_OK)]
    if denied:
        raise TrainError("cannot write " + ", ".join(denied[:10]) + (" …" if len(denied) > 10 else ""))


def _rewrite_source(root: Path, source: str, mapping: dict[str, str], classes: list[str]) -> int:
    """改写来源的标注与预标注记录中的类别名，再写来源的类别与 labels.txt。返回改写的标注数。"""
    directory = paths.source_dir(root, source)
    count = sum(xanylabeling.relabel(path, mapping) for path in _annotations(root, source))
    log_path = directory / paths.PRELABEL_RECORD
    if log_path.is_file():
        # 预标注的形状与人确认后的标注按类别配对（修正量统计），名称必须一起改。
        log = json.loads(log_path.read_text(encoding="utf-8"))
        for item in log.get("images", {}).values():
            for shape in item.get("shapes", []):
                shape["label"] = mapping.get(shape["label"], shape["label"])
        paths.write_json(log_path, log)
    record = sources.read_record(directory)
    if record.get("classes") != classes:
        record["classes"] = classes
        paths.write_json(directory / paths.SOURCE_RECORD, record)
    sources.write_labels(directory, classes)
    return count


def _source_files(root: Path, names: list[str]) -> list[Path]:
    """改写这些来源要写的文件（已存在的）；目录本身也要可写，原子替换在目录中新建临时文件。"""
    files = []
    for source in names:
        directory = paths.source_dir(root, source)
        files += [directory / paths.SOURCE_RECORD, directory / paths.SOURCE_LABELS,
                  directory / paths.PRELABEL_RECORD, *_annotations(root, source)]
    return [path for path in files if path.exists()]


def _refresh_locks(repo: Path, root: Path, models: dict[str, dict], changed: dict[str, list[str]],
                   translate) -> tuple[list[str], list[str]]:
    """被改写的来源的锁记录：记录按 translate 换算后等于来源的新类别时自动更新（这次修改不改变它对这个模型的
    含义），否则留给人确认。返回 (自动更新的模型, 需要确认的模型)。"""
    updated, pending = [], []
    for model_id in _users(models, set(changed)):
        recorded = lock.read(repo, model_id)
        entries = {source: recorded[source] for source in changed if source in recorded}
        new = {source: translate(model_id, names) for source, names in entries.items()}
        good = {source: names for source, names in new.items() if names == changed[source]}
        if good and any(recorded[source] != names for source, names in good.items()):
            lock.write(repo, model_id, {**recorded, **good})
            updated.append(model_id)
        if len(good) != len(entries):
            pending.append(model_id)
    return updated, pending


def rename(repo: Path, root: Path, spec: dict, old: str, new: str) -> Result:
    """原位改名。同一个类别名在共用来源的模型之间含义相同，所以依赖这些来源且含旧名称的模型一起改名，
    它们的其他来源也一起改写（按共用来源传递）。"""
    if old not in spec["classes"]:
        raise TrainError(f"{old} is not a class of {spec['id']}")
    if not new.strip() or new != new.strip():
        raise TrainError(f"invalid class name {new!r}")
    if new in spec["classes"]:
        raise TrainError(f"{spec['id']} already has a class {new}")

    def renamed(names: list[str]) -> list[str]:
        return [new if name == old else name for name in names]

    _require_confirmed(repo, root, spec, renamed)
    models = _models(repo)
    scope_models, scope_sources = {spec["id"]}, set(spec["data"]["sources"])
    while True:
        grow = {model_id for model_id in _users(models, scope_sources) if old in _labels(models[model_id])}
        if grow <= scope_models:
            break
        scope_models |= grow
        for model_id in grow:
            scope_sources |= set(models[model_id]["data"]["sources"])
    conflicts = [f"model {model_id} already has a class {new}" for model_id in sorted(scope_models)
                 if new in models[model_id]["classes"]]
    before = {}
    for source in sorted(scope_sources):
        directory = paths.source_dir(root, source)
        before[source] = names = sources.class_list(directory, sources.read_record(directory))
        if old in names and new in names:
            conflicts.append(f"source {source} already has a class {new}")
    if conflicts:
        raise TrainError(f"cannot rename {old} to {new}:\n  " + "\n  ".join(conflicts))
    _check_writable(_source_files(root, sorted(scope_sources)) +
                    [specs.path(repo, model_id) for model_id in scope_models])
    result = Result([f"renamed {old} to {new}"], sorted(scope_sources))
    after = {source: renamed(names) for source, names in before.items()}
    for source in sorted(scope_sources):
        count = _rewrite_source(root, source, {old: new}, after[source])
        result.lines.append(f"  source {source}: {count} annotations rewritten")
    updated, pending = _refresh_locks(repo, root, models, after, lambda _, names: renamed(names))
    for model_id in sorted(scope_models - {spec["id"]}) + [spec["id"]]:
        lock.record_rename(repo, model_id, old, new)  # 中断后重新执行时再记一次，结果相同
        specs.write_classes(repo, model_id, renamed(models[model_id]["classes"]))
        result.lines.append(f"  model {model_id}: class {models[model_id]['classes'].index(old)} renamed")
    result.lines += _lock_lines(updated, pending)
    result.lines += _cascade_lines(models, scope_models, old, new)
    result.lines += _republish_lines(repo, sorted(scope_models), sources_changed=True)
    return result


def retire(repo: Path, root: Path, spec: dict, name: str, into: str | None = None) -> Result:
    """从模型中删除一个类别，后面的类别号前移。只 retire 时来源不变（其他模型可能还在用它，本模型读取时
    忽略它）；--into 把来源中的这个类别并入另一个类别，其他模型仍把它当作独立类别时拒绝。"""
    classes = spec["classes"]
    if name not in classes:
        raise TrainError(f"{name} is not a class of {spec['id']}")
    if len(classes) == 1:
        raise TrainError(f"{name} is the only class of {spec['id']}")
    if into is not None and (into == name or into not in classes):
        raise TrainError(f"--into must name another class of {spec['id']}")
    remaining = [c for c in classes if c != name]

    def merged(names: list[str]) -> list[str]:
        return [c for c in names if c != name] if into is not None else names

    _require_confirmed(repo, root, spec, merged)
    if into is None:
        specs.write_classes(repo, spec["id"], remaining)
        return Result([f"retired {name} from {spec['id']} (the sources keep it; {spec['id']} ignores it)",
                       *_renumber_lines(classes, name), *_cascade_lines(_models(repo), {spec["id"]}, name),
                       *_republish_lines(repo, [spec["id"]])])

    models = _models(repo)
    own = set(spec["data"]["sources"])
    others = [model_id for model_id in _users(models, own) if model_id != spec["id"]]
    conflicts = [f"model {model_id} has {name} as a class of its own: merge it there first "
                 f"(train update {model_id} retire {name} --into …), or only retire {name} here"
                 for model_id in others if name in _labels(models[model_id])]
    if conflicts:
        raise TrainError(f"cannot merge {name} into {into} in the sources of {spec['id']}:\n  " +
                         "\n  ".join(conflicts))
    before = {}
    for source in sorted(own):
        directory = paths.source_dir(root, source)
        before[source] = sources.class_list(directory, sources.read_record(directory))
    _check_writable(_source_files(root, sorted(own)) + [specs.path(repo, spec["id"])])
    after = {source: [c for c in names if c != name] for source, names in before.items()}
    result = Result([f"merged {name} into {into} in {spec['id']} and its sources"], sorted(own))
    for source in sorted(own):
        count = _rewrite_source(root, source, {name: into}, after[source])
        result.lines.append(f"  source {source}: {count} annotations rewritten")
    # 合并改变了来源的标注：只有本模型的记录自动更新，其他模型的数据变了（原来忽略的框成了 into），需要人确认。
    updated, pending = _refresh_locks(
        repo, root, models, after, lambda model_id, names: merged(names) if model_id == spec["id"] else names)
    specs.write_classes(repo, spec["id"], remaining)
    result.lines += _renumber_lines(classes, name)
    result.lines += _lock_lines(updated, pending)
    result.lines += _cascade_lines(models, {spec["id"]}, name)
    result.lines += _republish_lines(repo, [spec["id"]], sources_changed=True)
    return result


def add(repo: Path, root: Path, spec: dict, name: str, from_class: str | None = None,
        skip: dict[str, str] | None = None) -> Result:
    """追加一个类别。已有这个类别的来源不变（之前已为它标注过）；其余来源追加类别，并把已检查的图片（--from 时
    只是含 from_class 框的）改为未检查、记入复核名单：画面中可能有新类别的目标，不能再当作背景。skip 中的来源
    （来源 → 理由）只追加类别、不撤销。

    来源记录先写（名单、类别与 revoked: false 一次原子写入），再改标注，最后标记 revoked：中断后来源已有这个
    类别，但名单未完成，重新执行时按名单继续撤销。"""
    skip = skip or {}
    if not name.strip() or name != name.strip():
        raise TrainError(f"invalid class name {name!r}")
    if name in spec["classes"]:
        raise TrainError(f"{spec['id']} already has a class {name}")
    if from_class is not None and from_class not in spec["classes"]:
        raise TrainError(f"--from must name a class of {spec['id']}")
    own = spec["data"]["sources"]
    unknown = sorted(set(skip) - set(own))
    if unknown:
        raise TrainError(f"--skip-source: {', '.join(unknown)} is not a source of {spec['id']}")
    if not own:
        raise TrainError(f"data.sources of {spec['id']} is empty")

    def added(names: list[str]) -> list[str]:
        return names if name in names else [*names, name]

    _require_confirmed(repo, root, spec, added)
    models = _models(repo)
    plans = {}  # 来源 → 这次的复核名单（None：来源已有这个类别且名单已完成）
    for source in own:
        directory = paths.source_dir(root, source)
        record = sources.read_record(directory)
        unfinished = [r for r in record.get("reviews") or [] if r["class"] == name and not r.get("revoked")]
        if unfinished:
            plans[source] = unfinished[-1]
        elif name in sources.class_list(directory, record):
            plans[source] = None
        elif source in skip:
            plans[source] = {"class": name, "from": from_class, "at": _now(), "files": [], "skipped": skip[source],
                             "revoked": False}
        else:
            plans[source] = {"class": name, "from": from_class, "at": _now(),
                             "files": _checked(directory, record, from_class), "revoked": False}
    _check_writable(_source_files(root, [s for s, plan in plans.items() if plan is not None]) +
                    [specs.path(repo, spec["id"])])

    result = Result([f"added {name} to {spec['id']} as class {len(spec['classes'])}"],
                    sorted(source for source, plan in plans.items() if plan is not None))
    after = {}
    for source, plan in plans.items():
        directory = paths.source_dir(root, source)
        record = sources.read_record(directory)
        after[source] = added(sources.class_list(directory, record))
        if plan is None:
            result.lines.append(f"  source {source}: already has {name}, unchanged")
            continue
        reviews = record.setdefault("reviews", [])
        if plan not in reviews:
            reviews.append(plan)
        record["classes"] = after[source]
        paths.write_json(directory / paths.SOURCE_RECORD, record)
        sources.write_labels(directory, after[source])
        for image in plan["files"]:
            path = xanylabeling.annotation_path(directory / image)
            if path.is_file():
                xanylabeling.update(path, checked=False)
        record = sources.read_record(directory)
        next(r for r in record["reviews"] if r == plan)["revoked"] = True
        paths.write_json(directory / paths.SOURCE_RECORD, record)
        if plan.get("skipped"):
            result.lines.append(f"  source {source}: class added, nothing unchecked ({plan['skipped']})")
        else:
            which = f"with {from_class} boxes " if from_class else ""
            result.lines.append(f"  source {source}: class added, {len(plan['files'])} checked images {which}"
                                f"unchecked for review")
    changed = {source: names for source, names in after.items() if plans[source] is not None}
    updated, pending = _refresh_locks(
        repo, root, models, changed, lambda model_id, names: added(names) if model_id == spec["id"] else names)
    specs.write_classes(repo, spec["id"], [*spec["classes"], name])
    result.lines += _lock_lines(updated, pending)
    result.lines += _republish_lines(repo, [spec["id"]], sources_changed=bool(changed))
    if any(plan and plan["files"] for plan in plans.values()):
        result.lines.append(f"  next: label {name} in a few dozen of these images and check them, train v0, then "
                            f"train prelabel {spec['id']} --source <source> --class {name} --run <v0>")
    return result


def _checked(directory: Path, record: dict, from_class: str | None) -> list[str]:
    """来源中已检查的图片；from_class 时只取含这个类别形状的。"""
    found = []
    for image in sorted(record["images"]):
        path = xanylabeling.annotation_path(directory / image)
        if not path.is_file():
            continue
        annotation = xanylabeling.read(path)
        if annotation.checked and (from_class is None or any(s.label == from_class for s in annotation.shapes)):
            found.append(image)
    return found


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _renumber_lines(classes: list[str], name: str) -> list[str]:
    index = classes.index(name)
    if index == len(classes) - 1:
        return []
    return [f"  classes after {name} moved down by one ({classes[index + 1]} is now class {index}); downstream "
            "that identifies classes by number must follow"]


def _lock_lines(updated: list[str], pending: list[str]) -> list[str]:
    lines = [f"  updated {lock.FILE} of {model_id}" for model_id in updated]
    lines += [f"  {model_id} must review the changed sources: train update {model_id} --confirm"
              for model_id in pending]
    return lines


def _republish_lines(repo: Path, model_ids: list[str], sources_changed: bool = False) -> list[str]:
    from dsf_deploy import models as deploy_models
    published = [m for m in model_ids if (paths.model_dir(repo, m) / deploy_models.RELEASE).is_file()]
    lines = []
    if published:
        lines.append(f"  {', '.join(published)}: the published classes no longer match model.yaml, so packaging "
                     "fails until a run on the new classes is published; services using them must review the change")
    if sources_changed:
        lines.append("  reopen the changed sources in X-AnyLabeling and load labels.txt again")
    return lines
