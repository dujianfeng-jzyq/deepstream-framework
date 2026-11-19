"""模型目录的契约校验：规格（model.yaml）、发布事实（release.json）、产物（artifacts/）与
nvinfer 配置彼此一致。打包前执行，不一致的模型不会进入发布包；训练工具发布时写出同一格式。

  models/<id>/model.yaml     规格，人工维护：类别、输入尺寸、输出布局、训练与导出参数
  models/<id>/release.json   发布事实，由发布工具写入：产物的 sha256、导出的 ONNX 签名、来源与指标
  models/<id>/artifacts/     已发布的产物，只能是 release.json 列出的文件
  models/<id>/*.txt          nvinfer 配置，人工维护的运行参数；与产物相关的键在这里检查

engine 是目标机生成的派生物，必须写到 /data/engines/<id>/ 下：数据目录跨发布与清洁安装保留，
模型目录对所有容器只读。

还没有发布的模型（只有 model.yaml）可以存在，但不能有引用它的 nvinfer 配置。

服务 → 模型的类别依赖（docs/training.md §15.1）：services/<服务>/models.lock.json 记录服务确认时各模型发布的
类别列表，由 lock-models 写入。与模型的 release.json 不同时校验失败并打印差异；没有这个文件的服务不检查。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import classes as class_diff
from . import layout
from .errors import DeployError

SPEC = "model.yaml"
RELEASE = "release.json"
ARTIFACTS = "artifacts"
SCHEMA = 1
# 已知输出布局的最后一维（每个候选的字段数）。未列出的布局不做形状检查：
# 新的解析插件可以引入新布局而不修改框架，代价是导出形状只由插件自己的测试保证。
OUTPUT_FIELDS = {"xyxy-confidence-class_id": 6}
SERVICE_LOCK = "models.lock.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def model_dirs(root: Path) -> list[Path]:
    models = root / layout.MODELS
    return sorted(p for p in models.iterdir() if p.is_dir()) if models.is_dir() else []


def check(root: Path) -> None:
    """校验 root/models 下的全部模型与各服务的模型锁记录；有任何问题时一次列出全部问题。"""
    problems = [f"{directory.relative_to(root)}: {problem}"
                for directory in model_dirs(root) for problem in check_model(directory)]
    problems += check_service_locks(root)
    if problems:
        raise DeployError("model check failed:\n  " + "\n  ".join(problems))


def published_classes(root: Path, model: str) -> list[str] | None:
    """模型发布的类别（release.json）；还没有发布时为 None。"""
    path = root / layout.MODELS / model / RELEASE
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8")).get("classes")
    except ValueError as error:
        raise DeployError(f"{path}: {error}") from error
    return value if isinstance(value, list) else None


def read_service_lock(path: Path) -> dict[str, list[str]]:
    if not path.is_file():
        return {}
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as error:
        raise DeployError(f"{path}: {error}") from error
    recorded = document.get("models") if isinstance(document, dict) else None
    if document.get("schema") != SCHEMA or not isinstance(recorded, dict) or \
            not all(isinstance(v, list) and all(isinstance(c, str) for c in v) for v in recorded.values()):
        raise DeployError(f"{path}: not a models lock of schema {SCHEMA}; it is written by dsf-deploy.py lock-models")
    return recorded


def check_service_locks(root: Path) -> list[str]:
    """服务记录的模型类别与模型发布的类别一致。发布包里没有 services/（只带服务的 compose 与模板），这里只在
    开发机打包时检查。"""
    problems = []
    for path in sorted((root / layout.SERVICES).glob(f"*/{SERVICE_LOCK}")):
        relative = path.relative_to(root)
        service = path.parent.name
        try:
            recorded = read_service_lock(path)
        except DeployError as error:
            problems.append(str(error))
            continue
        for model, names in sorted(recorded.items()):
            published = published_classes(root, model)
            if published is None:
                problems.append(f"{relative}: model {model} has not been published")
            elif published != names:
                problems.append(f"{relative}: the published classes of {model} changed since {service} confirmed "
                                f"them:\n    " + "\n    ".join(class_diff.diff(names, published)) +
                                f"\n    review {service}, then: framework/deploy/dsf-deploy.py lock-models {service}")
    return problems


def lock_models(root: Path, service: str, models: list[str]) -> list[str]:
    """把模型当前发布的类别记入 services/<服务>/models.lock.json，返回每个模型相对旧记录的差异。不给模型时
    重新确认记录中已有的全部模型。"""
    directory = root / layout.SERVICES / service
    if not service or "/" in service or not (directory / layout.UNIT_COMPOSE).is_file():
        raise DeployError(f"{service!r} is not a service (no {layout.SERVICES}/{service}/{layout.UNIT_COMPOSE})")
    path = directory / SERVICE_LOCK
    recorded = read_service_lock(path)
    targets = models or sorted(recorded)
    if not targets:
        raise DeployError(f"{path.relative_to(root)} does not exist yet: name the models {service} uses")
    lines, updated = [], dict(recorded)
    for model in targets:
        published = published_classes(root, model)
        if published is None:
            raise DeployError(f"model {model} has not been published (no {layout.MODELS}/{model}/{RELEASE})")
        if model not in recorded:
            lines.append(f"  {model}: recorded ({len(published)} classes)")
        elif recorded[model] != published:
            lines.append(f"  {model}:\n    " + "\n    ".join(class_diff.diff(recorded[model], published)))
        else:
            lines.append(f"  {model}: unchanged")
        updated[model] = published
    if updated != recorded or not path.is_file():
        path.write_text(json.dumps({"schema": SCHEMA, "models": dict(sorted(updated.items()))}, ensure_ascii=False,
                                   indent=2) + "\n", encoding="utf-8")
    return lines


def check_model(directory: Path) -> list[str]:
    spec_path = directory / SPEC
    if not spec_path.is_file():
        return [f"missing {SPEC}"]
    from .render import _yaml  # PyYAML 只在需要时导入（与渲染相同）
    try:
        spec = _yaml().load(spec_path.read_text(encoding="utf-8")) or {}
    except Exception as error:
        return [f"{SPEC}: {error}"]
    problems = check_spec(directory.name, spec)
    if problems:
        return problems  # 规格本身不成立时，与发布和配置的对照没有意义

    release_path = directory / RELEASE
    release = None
    if release_path.is_file():
        try:
            release = json.loads(release_path.read_text(encoding="utf-8"))
        except ValueError as error:
            return [f"{RELEASE}: {error}"]
        problems += _check_release(directory, spec, release)
    elif (directory / ARTIFACTS).exists():
        problems.append(f"{ARTIFACTS}/ exists but there is no {RELEASE}")
    for config in sorted(directory.glob("*.txt")):
        problems += _check_nvinfer(directory, config, spec, release)
    return problems


def check_spec(name: str, spec: dict) -> list[str]:
    if not isinstance(spec, dict):
        return [f"{SPEC} must be a mapping"]
    problems = []
    if spec.get("schema") != SCHEMA:
        problems.append(f"{SPEC}: schema must be {SCHEMA}")
    if spec.get("id") != name:
        problems.append(f"{SPEC}: id must equal the directory name {name!r}")
    classes = spec.get("classes")
    if not isinstance(classes, list) or not classes or \
            not all(isinstance(item, str) and item.strip() for item in classes):
        problems.append(f"{SPEC}: classes must be a non-empty list of names")
    elif len(set(classes)) != len(classes):
        problems.append(f"{SPEC}: class names must be unique")
    size = spec.get("input")
    if not isinstance(size, dict) or not all(isinstance(size.get(k), int) and size[k] > 0
                                             for k in ("width", "height")):
        problems.append(f"{SPEC}: input must give positive integer width and height")
    if not isinstance(spec.get("output_layout"), str) or not spec["output_layout"]:
        problems.append(f"{SPEC}: output_layout is required")
    return problems


def _check_release(directory: Path, spec: dict, release: dict) -> list[str]:
    if not isinstance(release, dict):
        return [f"{RELEASE} must be an object"]
    problems = []
    if release.get("schema") != SCHEMA:
        problems.append(f"{RELEASE}: schema must be {SCHEMA}")
    if release.get("model") != spec["id"]:
        problems.append(f"{RELEASE}: model must be {spec['id']!r}")
    # 规格在发布之后改动了契约：已发布的产物不再符合，必须重新训练并发布。
    for key in ("classes", "input", "output_layout"):
        if release.get(key) != spec[key]:
            problems.append(f"{key} in {SPEC} differs from the published release; "
                            "retrain and publish, or restore the published value")

    artifacts = release.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        return problems + [f"{RELEASE}: artifacts must list the published files"]
    root = directory / ARTIFACTS
    for relative, digest in sorted(artifacts.items()):
        path = directory / relative
        if Path(relative).is_absolute() or root.resolve() not in path.resolve().parents:
            problems.append(f"{RELEASE}: artifact {relative} must be inside {ARTIFACTS}/")
        elif not path.is_file():
            problems.append(f"missing artifact {relative}")
        elif sha256(path) != digest:
            problems.append(f"{relative} does not match its sha256 in {RELEASE}")
    listed = {(directory / relative).resolve() for relative in artifacts}
    if root.is_dir():
        for path in sorted(p for p in root.rglob("*") if p.is_file() and p.resolve() not in listed):
            problems.append(f"{path.relative_to(directory)} is not a published artifact")

    labels = directory / ARTIFACTS / "labels.txt"
    if labels.is_file() and labels.read_text(encoding="utf-8").splitlines() != spec["classes"]:
        problems.append(f"{ARTIFACTS}/labels.txt must list the classes of {SPEC}, one per line, in order")
    problems += check_signature(spec, release.get("onnx"))
    return problems


def check_signature(spec: dict, onnx: object) -> list[str]:
    """导出的 ONNX 签名（发布工具从模型读出）与输入尺寸、输出布局对得上。"""
    if not isinstance(onnx, dict) or not onnx.get("inputs") or not onnx.get("outputs"):
        return [f"{RELEASE}: onnx must record the exported inputs and outputs"]
    problems = []
    dynamic = bool(onnx.get("dynamic"))
    shape = onnx["inputs"][0].get("shape") or []
    expected = [None, 3, spec["input"]["height"], spec["input"]["width"]]
    if len(shape) != 4 or shape[1:] != expected[1:]:
        problems.append(f"exported input shape {shape} does not match input {spec['input']} (NCHW, 3 channels)")
    elif not dynamic and shape[0] != onnx.get("batch"):
        problems.append(f"exported input batch {shape[0]} differs from the recorded batch {onnx.get('batch')}")
    fields = OUTPUT_FIELDS.get(spec["output_layout"])
    if fields is not None:
        outputs = onnx["outputs"]
        last = (outputs[0].get("shape") or [None])[-1] if len(outputs) == 1 else None
        if last != fields:
            problems.append(f"{spec['output_layout']} needs one output whose last dimension is {fields}")
    return problems


def _properties(config: Path) -> dict[str, str] | None:
    """nvinfer 配置的 [property] 段；没有该段（例如标签文件）返回 None。"""
    section, values, found = None, {}, False
    for raw in config.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            found = found or section == "property"
        elif section == "property" and "=" in line:
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    return values if found else None


def _check_nvinfer(directory: Path, config: Path, spec: dict, release: dict | None) -> list[str]:
    properties = _properties(config)
    if properties is None:
        return []
    name = config.name
    if release is None:
        return [f"{name} configures a model that has not been published ({RELEASE} is missing)"]
    published = {(directory / relative).resolve() for relative in release.get("artifacts") or {}}
    problems = []
    for key in ("onnx-file", "labelfile-path"):
        value = properties.get(key)
        if value and (config.parent / value).resolve() not in published:
            problems.append(f"{name}: {key}={value} is not a published artifact")
    if "onnx-file" not in properties:
        problems.append(f"{name}: onnx-file is required")
    engines = f"{layout.MOUNTS['CONTAINER_DATA_PATH']}/engines/{spec['id']}/"
    engine = properties.get("model-engine-file", "")
    if not engine.startswith(engines) or ".." in Path(engine).parts:
        problems.append(f"{name}: model-engine-file must be a file under {engines} "
                        f"(the models directory is read-only), got {engine or 'nothing'}")
    classes = properties.get("num-detected-classes")
    if classes is not None and classes != str(len(spec["classes"])):
        problems.append(f"{name}: num-detected-classes={classes} but {SPEC} has {len(spec['classes'])} classes")
    onnx = release.get("onnx") or {}
    batch = properties.get("batch-size", "1")
    if not onnx.get("dynamic") and batch != str(onnx.get("batch")):
        problems.append(f"{name}: batch-size={batch} but the ONNX was exported with a fixed batch of "
                        f"{onnx.get('batch')}")
    return problems
