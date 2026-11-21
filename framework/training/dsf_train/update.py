"""train update <id> <操作>：修改模型的类别与来源，连同来源的标注与其他模型（docs/training.md §15.3）。

每次运行先完成全部检查（冲突、可写），再写入。写入顺序保证中断后重新执行同一命令即可完成：
标注文件 → 来源记录 → 其他模型的 model.yaml → 本模型的 model.yaml（add 例外，见 add）。标注先改，来源记录还写着
旧名称时 prepare 遇到新名称报“未知类别”，不会在不一致的状态下训练；重新执行时已改过的文件
没有可改的形状，跳过；本模型的规格最后改，它改完即全部完成。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import paths, sources, spec as specs, xanylabeling
from .errors import TrainError


@dataclass
class Result:
    lines: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)  # 要重新生成标注工具文件的来源


def _models(repo: Path) -> dict[str, dict]:
    """仓库中全部模型的规格。任何一个规格不成立都报错：影响范围算不全时不能改写共用的来源。"""
    return specs.load_all(repo)


def _labels(spec: dict) -> list[str]:
    """模型的类别中哪些是来源中的标签：二级模型的类别是框上属性的值，不是标签。"""
    return [] if spec.get("cascade") else spec["classes"]


def _children(models: dict[str, dict], parents: set[str], name: str) -> list[str]:
    """以这些模型为父、触发类别中有 name 的二级模型。"""
    return [model_id for model_id, spec in sorted(models.items())
            if (spec.get("cascade") or {}).get("parent") in parents and name in spec["cascade"].get("classes", [])]


def _rewrite_cascades(repo: Path, models: dict[str, dict], parents: set[str], name: str, new: str) -> list[str]:
    """父类别 name 改名或并入 new：改写触发它的二级模型的 cascade.classes。父模型的类别号随之变化，二级
    模型 nvinfer 配置中的 operate-on-class-ids 由人对照（打包时的级联校验发现不符）。"""
    lines = []
    for model_id in _children(models, parents, name):
        try:
            classes = specs.write_cascade_classes(repo, model_id, {name: new})
        except TrainError as error:
            lines.append(f"  second-stage model {model_id} triggers on {name}: change it to {new} in its "
                         f"cascade.classes by hand ({error})")
            continue
        lines.append(f"  second-stage model {model_id}: cascade.classes {name} -> {new} ({', '.join(classes)}); "
                     "check operate-on-class-ids in its nvinfer config after the parent is republished")
    return lines


def _users(models: dict[str, dict], names: set[str]) -> list[str]:
    """依赖这些来源中任何一个的模型。"""
    return [model_id for model_id, spec in models.items() if names & set(spec["data"]["sources"])]


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
    """改写来源的标注与预标注记录中的类别名，再写来源的类别。返回改写的标注数。"""
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
    return count


def _source_files(root: Path, names: list[str]) -> list[Path]:
    """改写这些来源要写的文件（已存在的）；目录本身也要可写，原子替换在目录中新建临时文件。"""
    files = []
    for source in names:
        directory = paths.source_dir(root, source)
        files += [directory / paths.SOURCE_RECORD,
                  directory / paths.PRELABEL_RECORD, *_annotations(root, source)]
    return [path for path in files if path.exists()]


def _shared_lines(models: dict[str, dict], spec_id: str, unchecked: dict[str, int]) -> list[str]:
    """来源中被撤销 checked 的图片对共用来源的其他模型同样不进入数据集，直到重新勾选。"""
    lines = []
    for model_id in sorted(_users(models, {s for s, count in unchecked.items() if count})):
        if model_id != spec_id:
            names = [s for s in models[model_id]["data"]["sources"] if unchecked.get(s)]
            lines.append(f"  note: {model_id} also uses {', '.join(names)}: the unchecked images leave its dataset "
                         "too until they are checked again")
    return lines


def _publish_rename_note(repo: Path, spec: dict, name: str) -> list[str]:
    """发布时同一类别号上一删一增读作改名（publish.class_changes）。上次发布以来删去了最后一个类别、现在又在
    这个位置追加新类别时两者无法区分，提醒人。"""
    from dsf_deploy import models as deploy_models
    path = paths.model_dir(repo, spec["id"]) / deploy_models.RELEASE
    if not path.is_file():
        return []
    published = json.loads(path.read_text(encoding="utf-8")).get("classes") or []
    index = len(spec["classes"])
    if index < len(published) and published[index] not in spec["classes"]:
        return [f"  note: class {index} was {published[index]} in the current release, so publish compares {name} "
                f"with it as if {published[index]} had been renamed to {name}"]
    return []


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
    for model_id in sorted(scope_models - {spec["id"]}) + [spec["id"]]:
        specs.write_classes(repo, model_id, renamed(models[model_id]["classes"]))
        result.lines.append(f"  model {model_id}: class {models[model_id]['classes'].index(old)} renamed")
    result.lines += _rewrite_cascades(repo, models, scope_models, old, new)
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

    models = _models(repo)
    children = _children(models, {spec["id"]}, name)
    if into is None and children:
        # 父模型不再输出这个类别，二级模型却仍作用于它：先由人决定二级模型改作用于哪些类别。
        raise TrainError(f"second-stage model {', '.join(children)} triggers on {name}: remove {name} from its "
                         f"cascade.classes first (or merge: train update {spec['id']} retire {name} --into …)")
    if into is None:
        specs.write_classes(repo, spec["id"], remaining)
        return Result([f"retired {name} from {spec['id']} (the sources keep it; {spec['id']} ignores it)",
                       *_renumber_lines(classes, name), *_republish_lines(repo, [spec["id"]])])

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
    specs.write_classes(repo, spec["id"], remaining)
    result.lines += _renumber_lines(classes, name)
    # 同名类别在共用来源的模型之间含义相同：合并是说 name 就是 into，其他有 into 的模型的数据随之变多。
    result.lines += [f"  note: {model_id} also has {into}: its {into} boxes in the shared sources now include the "
                     f"former {name} boxes" for model_id in others if into in _labels(models[model_id])]
    result.lines += _rewrite_cascades(repo, models, {spec["id"]}, name, into)
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
        _extend(directory, after[source], [plan])
        if plan.get("skipped"):
            result.lines.append(f"  source {source}: class added, nothing unchecked ({plan['skipped']})")
        else:
            which = f"with {from_class} boxes " if from_class else ""
            result.lines.append(f"  source {source}: class added, {len(plan['files'])} checked images {which}"
                                f"unchecked for review")
    specs.write_classes(repo, spec["id"], [*spec["classes"], name])
    result.lines += _shared_lines(models, spec["id"], {s: len(plan["files"]) for s, plan in plans.items() if plan})
    result.lines += _publish_rename_note(repo, spec, name)
    result.lines += _republish_lines(repo, [spec["id"]], sources_changed=any(plans.values()))
    if any(plan and plan["files"] for plan in plans.values()):
        result.lines.append(f"  next: label {name} in a few dozen of these images and check them, train v0, then "
                            f"train prelabel {spec['id']} --source <source> --class {name} --run <v0>")
    return result


def _extend(directory: Path, classes: list[str], plans: list[dict]) -> None:
    """来源记下新的类别与复核名单（一次原子写入，revoked: false），再撤销名单中图片的 checked，最后标记
    revoked：中断后来源已有这些类别，名单未完成，重新执行时按名单继续撤销。"""
    record = sources.read_record(directory)
    reviews = record.setdefault("reviews", [])
    for plan in plans:
        if plan not in reviews:
            reviews.append(plan)
    record["classes"] = classes
    paths.write_json(directory / paths.SOURCE_RECORD, record)
    for plan in plans:
        for image in plan["files"]:
            path = xanylabeling.annotation_path(directory / image)
            # 检查不通过的保留原状态：它本来就不进入数据集，人修正时也要为新类别看一遍（它在名单中）。
            if path.is_file() and xanylabeling.read(path).checked:
                xanylabeling.update(path, checked=False)
    record = sources.read_record(directory)
    for plan in plans:
        next(r for r in record["reviews"] if r == plan)["revoked"] = True
    paths.write_json(directory / paths.SOURCE_RECORD, record)


def add_source(repo: Path, root: Path, spec: dict, source: str, reason: str = "") -> Result:
    """把来源加入模型的 data.sources。来源从此为模型的类别标注：
    - 还没有类别的来源（刚导入）记下模型的类别。随导入带来的已勾选标注视为按这些类别标注过（与 train accept
      一样，确认的责任在执行它的人），输出中给出数量。
    - 已为其他类别标注过的来源追加缺少的类别；已勾选的图片可能有这些类别的目标，与 add 一样改为未勾选、记入
      复核名单（每个类别一份）。给出 reason 时只追加类别、不撤销，理由记入名单。
    - 二级模型不改变来源的类别：来源必须已为它的 cascade.classes 标注（先把来源加入父模型）。
    写入顺序：来源记录与标注 → model.yaml；中断后重新执行同一命令即可完成。"""
    directory = paths.source_dir(root, source)
    record = sources.read_record(directory)
    if source in spec["data"]["sources"]:
        raise TrainError(f"{source} is already a source of {spec['id']}")
    have = sources.class_list(directory, record)
    unfinished = [r for r in record.get("reviews") or [] if not r.get("revoked")]
    lines = []
    if spec.get("cascade"):
        if reason:
            raise TrainError("--reason goes with adding a source to a detection model")
        missing = [name for name in spec["cascade"]["classes"] if name not in have]
        if missing:
            parent = spec["cascade"]["parent"]
            raise TrainError(f"source {source} is not labeled for {', '.join(missing)} (cascade.classes of "
                             f"{spec['id']}): add it to the parent first: train update {parent} source add {source}")
        classes, plans = have, []
        lines.append(f"  source {source}: {spec['id']} reads {spec['data']['from_parent']['attribute']} on its "
                     f"{', '.join(spec['cascade']['classes'])} boxes")
    else:
        missing = [name for name in spec["classes"] if name not in have]
        classes, plans = [*have, *missing], []
        if not have:
            checked = len(_checked(directory, record, None))
            lines.append(f"  source {source}: labeled for the {len(classes)} classes of {spec['id']} from now on")
            if checked:
                lines.append(f"  note: {checked} checked annotations came with the import; they count as labeled for "
                             "these classes (uncheck the ones that are not)")
        elif missing:
            files = _checked(directory, record, None)
            now = _now()
            plans = [{"class": name, "from": None, "at": now, "files": [] if reason else files,
                      **({"skipped": reason} if reason else {}), "revoked": False} for name in missing]
            done = "nothing unchecked" if reason else f"{len(files)} checked images unchecked for review"
            lines.append(f"  source {source}: added {', '.join(missing)}; {done}")
        else:
            lines.append(f"  source {source}: already labeled for the classes of {spec['id']}")
    _check_writable(_source_files(root, [source]) + [specs.path(repo, spec["id"])])
    if classes != have or plans or unfinished:
        _extend(directory, classes, plans + unfinished)
    names = [*spec["data"]["sources"], source]
    specs.write_sources(repo, spec["id"], names)
    lines += _shared_lines(_models(repo), spec["id"], {source: len(plans[0]["files"]) if plans else 0})
    return Result([f"added source {source} to {spec['id']} (data.sources: {', '.join(names)})", *lines], [source])


def remove_source(repo: Path, root: Path, spec: dict, source: str) -> Result:
    """把来源移出模型的 data.sources。来源本身（素材、标注、类别）不变：其他模型可能还在用它，以后也可以再
    加回来。"""
    if source not in spec["data"]["sources"]:
        raise TrainError(f"{source} is not a source of {spec['id']}")
    names = [name for name in spec["data"]["sources"] if name != source]
    specs.write_sources(repo, spec["id"], names)
    return Result([f"removed source {source} from {spec['id']} (data.sources: {', '.join(names) or 'none'}); the "
                   "source itself is unchanged"], [source])


# 二级模型：类别是父类别框上一个属性的取值（§12）。改动的是标注中的属性值，不是来源的类别，所以来源的类别不变。


def _value_scope(repo: Path, root: Path, spec: dict) -> tuple[str, set[str], list[Path]]:
    """(属性, 触发类别, 来源中的标注文件)。同一来源中同名属性是同一个属性：另一个二级模型也在这些来源上
    声明它时拒绝，两边的取值必须一起改（暂不支持）。"""
    attribute = spec["data"]["from_parent"]["attribute"]
    own = set(spec["data"]["sources"])
    others = [model_id for model_id, other in sorted(_models(repo).items())
              if model_id != spec["id"] and other.get("cascade")
              and other["data"]["from_parent"]["attribute"] == attribute and own & set(other["data"]["sources"])]
    if others:
        raise TrainError(f"{', '.join(others)} also read attribute {attribute} in the sources of {spec['id']}: one "
                         "attribute has one set of values in a source, so change them together by hand")
    files = [path for source in sorted(own) for path in _annotations(root, source)]
    _check_writable(files + [specs.path(repo, spec["id"])])
    return attribute, set(spec["cascade"]["classes"]), files


def _value_lines(repo: Path, spec: dict, sources_names: list[str]) -> list[str]:
    return _republish_lines(repo, [spec["id"]]) + [
        f"  reopen {', '.join(sources_names)} in X-AnyLabeling and load the .attributes.json of "
        f"{spec['cascade']['parent']} again"]


def rename_value(repo: Path, root: Path, spec: dict, old: str, new: str) -> Result:
    """二级模型的类别改名：来源中属性取值 old 改为 new（所有框：同一来源中同名属性含义相同），再改规格。"""
    ignored = spec["data"].get("ignore_classes") or []
    if old not in spec["classes"]:
        raise TrainError(f"{old} is not a class of {spec['id']} (a value of "
                         f"{spec['data']['from_parent']['attribute']})")
    if not new.strip() or new != new.strip():
        raise TrainError(f"invalid class name {new!r}")
    if new in spec["classes"] or new in ignored:
        raise TrainError(f"{spec['id']} already has {new} in classes or data.ignore_classes")
    attribute, _, files = _value_scope(repo, root, spec)
    count = sum(xanylabeling.rewrite_attribute(path, attribute, {old: new}) for path in files)
    specs.write_classes(repo, spec["id"], [new if name == old else name for name in spec["classes"]])
    return Result([f"renamed {old} to {new} in {spec['id']}", f"  {count} boxes with {attribute}={old} rewritten",
                   *_value_lines(repo, spec, spec["data"]["sources"])], list(spec["data"]["sources"]))


def retire_value(repo: Path, root: Path, spec: dict, name: str, into: str | None = None) -> Result:
    """二级模型删除一个类别，后面的类别号前移。--into 把来源中的这个取值改为另一个类别；否则取值留在标注中，
    记入 data.ignore_classes（这样的框不收入，与“未知”一样）。"""
    classes = spec["classes"]
    if name not in classes:
        raise TrainError(f"{name} is not a class of {spec['id']}")
    if len(classes) <= 2:
        raise TrainError(f"{spec['id']} would have one class left: a classifier needs two")
    if into is not None and (into == name or into not in classes):
        raise TrainError(f"--into must name another class of {spec['id']}")
    attribute, _, files = _value_scope(repo, root, spec)
    remaining = [c for c in classes if c != name]
    if into is None:
        ignored = spec["data"].get("ignore_classes") or []
        if name not in ignored:
            specs.write_data_list(repo, spec["id"], "ignore_classes", [*ignored, name])
        specs.write_classes(repo, spec["id"], remaining)
        lines = [f"retired {name} from {spec['id']}: boxes with {attribute}={name} stay as they are and are left out "
                 "(data.ignore_classes)"]
    else:
        count = sum(xanylabeling.rewrite_attribute(path, attribute, {name: into}) for path in files)
        specs.write_classes(repo, spec["id"], remaining)
        lines = [f"merged {name} into {into} in {spec['id']}", f"  {count} boxes with {attribute}={name} rewritten"]
    return Result([*lines, *_renumber_lines(classes, name), *_value_lines(repo, spec, spec["data"]["sources"])],
                  list(spec["data"]["sources"]))


def add_value(repo: Path, root: Path, spec: dict, name: str, from_class: str | None = None) -> Result:
    """二级模型追加一个类别（类别号在最后）。--from 给出新类别以前被标成的取值：触发类别框上的这个取值被删去，
    这些框成为没有取值、不进入数据集，由 train prelabel <id> 提议、人在复核集中核对后写回；图片的 checked 不变，
    父模型照常使用它们。没有 --from 时标注不变。值在 data.ignore_classes 中时移出，那些框从此收入。"""
    ignored = spec["data"].get("ignore_classes") or []
    if not name.strip() or name != name.strip():
        raise TrainError(f"invalid class name {name!r}")
    if name in spec["classes"]:
        raise TrainError(f"{spec['id']} already has a class {name}")
    if from_class is not None and from_class not in spec["classes"] and from_class not in ignored:
        raise TrainError(f"--from must name a class of {spec['id']} or a value in data.ignore_classes")
    attribute, parents, files = _value_scope(repo, root, spec)
    lines = [f"added {name} to {spec['id']} as class {len(spec['classes'])}"]
    if from_class is not None:
        count = sum(xanylabeling.rewrite_attribute(path, attribute, {from_class: None}, parents) for path in files)
        lines.append(f"  {count} boxes with {attribute}={from_class} lost the value; next: train prelabel {spec['id']} "
                     f"--source <source> proposes values for them with the current model, and you correct them to "
                     f"{name} where it applies in the review")
    if name in ignored:
        specs.write_data_list(repo, spec["id"], "ignore_classes", [v for v in ignored if v != name])
        lines.append(f"  {name} is no longer in data.ignore_classes: its boxes are samples from now on")
    specs.write_classes(repo, spec["id"], [*spec["classes"], name])
    lines += _publish_rename_note(repo, spec, name)
    return Result([*lines, *_value_lines(repo, spec, spec["data"]["sources"])], list(spec["data"]["sources"]))


def _checked(directory: Path, record: dict, from_class: str | None) -> list[str]:
    """来源中人看过的图片（检查通过或不通过）；from_class 时只取含这个类别形状的。"""
    found = []
    for image in sorted(record["images"]):
        path = xanylabeling.annotation_path(directory / image)
        if not path.is_file():
            continue
        annotation = xanylabeling.read(path)
        if (annotation.checked or annotation.failed) and (from_class is None or any(s.label == from_class for s in annotation.shapes)):
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


def _republish_lines(repo: Path, model_ids: list[str], sources_changed: bool = False) -> list[str]:
    from dsf_deploy import models as deploy_models
    published = [m for m in model_ids if (paths.model_dir(repo, m) / deploy_models.RELEASE).is_file()]
    lines = []
    if published:
        lines.append(f"  {', '.join(published)}: the published classes no longer match model.yaml, so packaging "
                     "fails until a run on the new classes is published; services using them must review the change")
    if sources_changed:
        lines.append("  reopen the changed sources in X-AnyLabeling and load the .classes.txt and .attributes.json "
                     "again")
    return lines


def status(root: Path, spec: dict) -> list[str]:
    """train update <id> 不带操作时：类别，每个来源的复核进度，以及模型的类别不在来源中的问题。"""
    lines = [f"{spec['id']}: {len(spec['classes'])} classes"]
    lines += [f"  {index:3d} {name}" for index, name in enumerate(spec["classes"])]
    now = sources.model_sources(root, spec)
    for source in now:
        directory = paths.source_dir(root, source)
        record = sources.read_record(directory)
        lines.append(f"  source {source}: {len(record['images'])} images")
        for review, checked, total in sources.review_progress(directory, record):
            scope = f" (images with {review['from']})" if review.get("from") else ""
            lines.append(f"    review of {review['class']}{scope} since {review['at']}: {checked} of {total} "
                         "checked again")
    lines += [f"  {problem}" for problem in sources.missing_classes(spec, now)]
    return lines
