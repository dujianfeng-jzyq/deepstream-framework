"""来源 → 模型的类别依赖（docs/training.md §15.1）。

models/<id>/sources.lock.json 记录模型确认时每个来源的类别列表（有序，顺序即类别号），由工具写入，随模型提交
与发布。来源的类别与记录不同时 prepare、run、publish 拒绝，打印差异；人看过差异、判断模型是否需要修改之后，
用 train update <id> --confirm 重写记录。记录里没有的来源（第一次训练，或新加进 data.sources）由 prepare 按
来源当前的类别补记并提示：加入来源本身就是对 model.yaml 的一次人工修改。
"""

from __future__ import annotations

import json
from pathlib import Path

from dsf_deploy import classes as class_diff

from . import paths, sources
from .errors import TrainError

SCHEMA = 1
FILE = "sources.lock.json"


def path(repo: Path, model_id: str) -> Path:
    return paths.model_dir(repo, model_id) / FILE


def read(repo: Path, model_id: str) -> dict[str, list[str]]:
    """来源 → 确认过的类别列表；还没有记录时为空。"""
    lock_path = path(repo, model_id)
    if not lock_path.is_file():
        return {}
    try:
        document = json.loads(lock_path.read_text(encoding="utf-8"))
    except ValueError as error:
        raise TrainError(f"{lock_path}: {error}") from error
    recorded = document.get("sources") if isinstance(document, dict) else None
    if document.get("schema") != SCHEMA or not isinstance(recorded, dict) or \
            not all(isinstance(v, list) and all(isinstance(c, str) for c in v) for v in recorded.values()):
        raise TrainError(f"{lock_path}: not a sources lock of schema {SCHEMA}; it is written by train, "
                         f"restore it from git or confirm again with: train update {model_id} --confirm")
    return recorded


def write(repo: Path, model_id: str, recorded: dict[str, list[str]]) -> None:
    paths.write_json(path(repo, model_id), {"schema": SCHEMA, "sources": dict(sorted(recorded.items()))})


def current(root: Path, spec: dict) -> dict[str, list[str]]:
    """data.sources 中每个来源当前的类别列表。"""
    result = {}
    for source in spec["data"]["sources"]:
        directory = paths.source_dir(root, source)
        result[source] = sources.class_list(directory, sources.read_record(directory))
    return result


def missing_classes(spec: dict, source_classes: dict[str, list[str]]) -> list[str]:
    """模型的类别必须都在每个来源中，否则这个来源的图片无法为它作证（缺的类别在图上是背景）。"""
    problems = []
    for source, names in source_classes.items():
        missing = [name for name in spec["classes"] if name not in names]
        if missing:
            problems.append(f"source {source} has no class {', '.join(missing)}: its images cannot show them. "
                            f"Add classes with train update {spec['id']} add <class>, not by editing model.yaml")
    return problems


def check(repo: Path, root: Path, spec: dict, record_new: bool = False,
          now: dict[str, list[str]] | None = None) -> list[str]:
    """来源的类别与记录不同时报错并打印差异。record_new 时补记记录中没有的来源（prepare），返回给人看的提示；
    否则没有记录的来源同样报错（run、publish 不写入模型目录）。now 是已经读出的 current(root, spec)。"""
    recorded = read(repo, spec["id"])
    now = current(root, spec) if now is None else now
    changed, unrecorded = [], []
    for source, names in now.items():
        if source not in recorded:
            unrecorded.append(source)
        elif recorded[source] != names:
            changed.append(f"source {source}:\n    " + "\n    ".join(class_diff.diff(recorded[source], names)))
    problems = []
    if changed:
        problems.append(f"the classes of these sources changed since {spec['id']} confirmed them "
                        f"({path(repo, spec['id'])}):\n  " + "\n  ".join(changed) +
                        f"\n  review whether {spec['id']} needs changes, then confirm with: "
                        f"train update {spec['id']} --confirm")
    if unrecorded and not record_new:
        problems.append(f"{path(repo, spec['id'])} does not record the classes of {', '.join(unrecorded)}: "
                        f"run train prepare {spec['id']} first")
    if problems:
        raise TrainError("\n".join(problems))
    stale = sorted(set(recorded) - set(now))
    if not record_new or (not unrecorded and not stale):
        return []
    write(repo, spec["id"], {source: recorded.get(source, now[source]) for source in now})
    notes = [f"recorded the classes of source {source} in {path(repo, spec['id'])} (commit it with the model)"
             for source in unrecorded]
    notes += [f"dropped source {source} from {path(repo, spec['id'])}: it is no longer in data.sources"
              for source in stale]
    return notes


def confirm(repo: Path, root: Path, spec: dict) -> list[str]:
    """按来源当前的类别重写记录，返回每个来源相对旧记录的差异。模型的类别不在某个来源中时拒绝：
    确认它也无法训练。"""
    now = current(root, spec)
    problems = missing_classes(spec, now)
    if problems:
        raise TrainError("\n".join(problems))
    recorded = read(repo, spec["id"])
    lines = []
    for source, names in now.items():
        if source not in recorded:
            lines.append(f"  source {source}: recorded")
        elif recorded[source] != names:
            lines.append(f"  source {source}:\n    " + "\n    ".join(class_diff.diff(recorded[source], names)))
        else:
            lines.append(f"  source {source}: unchanged")
    for source in sorted(set(recorded) - set(now)):
        lines.append(f"  source {source}: dropped (no longer in data.sources)")
    if recorded != now:
        write(repo, spec["id"], now)
    return lines


def status(repo: Path, root: Path, spec: dict) -> list[str]:
    """train update <id> 不带操作时的状态：类别，以及每个来源与记录的差异。"""
    lines = [f"{spec['id']}: {len(spec['classes'])} classes"]
    lines += [f"  {index:3d} {name}" for index, name in enumerate(spec["classes"])]
    recorded = read(repo, spec["id"])
    now = current(root, spec)
    for source, names in now.items():
        if source not in recorded:
            state = "not recorded yet (train prepare records it)"
        elif recorded[source] != names:
            state = "changed since confirmed:\n    " + "\n    ".join(class_diff.diff(recorded[source], names))
        else:
            state = "confirmed"
        lines.append(f"  source {source}: {state}")
    lines += [f"  {problem}" for problem in missing_classes(spec, now)]
    return lines
