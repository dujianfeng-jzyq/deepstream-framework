"""命令行入口：init / import / accept / prelabel / prepare / run / eval / review / publish / update。由 framework/training/dsf-train.py 调用，
通常经 scripts/train.sh 在训练镜像中运行。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dsf_deploy.errors import DeployError

from . import dataset, labeling, paths, prelabel, publish, review, runs, sources, spec, update
from .errors import TrainError


def _classes(args: argparse.Namespace) -> list[str]:
    classes = list(args.class_ or [])
    if args.classes_file:
        classes += [line.strip() for line in Path(args.classes_file).read_text(encoding="utf-8").splitlines()
                    if line.strip()]
    return classes


def _options(values: list[str]) -> dict:
    """--option KEY=VALUE：值按 YAML 标量解读（epochs=3 是整数，device=0,1 是字符串）。"""
    from dsf import yamlio
    options = {}
    for value in values:
        key, sep, text = value.partition("=")
        if not sep or not key:
            raise TrainError(f"--option {value!r}: expected KEY=VALUE")
        try:
            options[key] = yamlio.load(text)
        except Exception as error:
            raise TrainError(f"--option {value!r}: {error}") from error
    return options


def _print(lines: list[str]) -> None:
    if lines:
        print("\n".join(lines))


def _prepare(repo: Path, root: Path, model: dict) -> str:
    """train prepare，以及 train run 训练之前：返回数据集标识。"""
    # 先于读取标注：prepare 因标注问题失败时，改正它们所需的属性文件已经是最新的。
    _print(labeling.refresh(repo, root, model["data"]["sources"]))
    for name, count in review.pending(root, model["data"]["sources"]):
        print(f"  warning: review {name} has {count} edited annotations that are not applied: "
              f"train review apply {name}")
    result = dataset.prepare(repo, root, model)
    print("\n".join(dataset.report(result, model)))
    for source in model["data"]["sources"]:
        for line in prelabel.review(root, model, source):
            print(line)
    return result.dataset_id


def _related_sources(repo: Path, model_id: str) -> list[str]:
    """模型、它的父模型与二级模型的来源：模型的类别或来源变化后，这些来源中的标注工具文件要重新生成。"""
    models, _ = labeling._specs(repo)
    model = models.get(model_id)
    if model is None:
        return []
    parent = (model.get("cascade") or {}).get("parent")
    related = [m for m, other in models.items()
               if m in (model_id, parent) or (other.get("cascade") or {}).get("parent") == model_id]
    return [source for m in sorted(related) for source in models[m]["data"]["sources"]]


def _evaluate(root: Path, model: dict, run: str, dataset_id: str | None = None, device: str | None = None) -> None:
    print("\n".join(runs.report(runs.evaluate(root, model, run, dataset_id, device))))
    print(f"  details: {paths.run_dir(root, model['id'], run) / runs.EVAL}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="train", description="Prepare data for and train the models in models/.")
    parser.add_argument("--repo", type=Path, default=None, help="project root (defaults to this checkout)")
    parser.add_argument("--train-root", help="training root (defaults to $TRAIN_ROOT or ~/<project>/training)")
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("init", help="create models/<id>/model.yaml for a new model")
    p.add_argument("model")
    p.add_argument("--task", required=True)
    p.add_argument("--class", dest="class_", action="append", help="a class name (repeat in class-number order)")
    p.add_argument("--classes-file", help="a text file with one class name per line")
    p.add_argument("--display-name", default="")
    p.add_argument("--parent", default="", help="classify: the first-stage model whose boxes are classified")
    p.add_argument("--parent-class", dest="parent_classes", action="append",
                   help="classify: a parent class whose boxes are classified (repeat)")
    p.add_argument("--attribute", default="", help="classify: the box attribute whose values are the classes")

    p = commands.add_parser("import", help="import images and videos (with their annotations) into a source")
    p.add_argument("source", help="source name under <train-root>/sources/ (created or extended)")
    p.add_argument("paths", nargs="+", type=Path, help="image or video files, or directories of them")
    p.add_argument("--every", type=float, default=1.0, help="seconds between extracted video frames (default 1)")
    p.add_argument("--dedupe", type=int, default=4,
                   help="drop a video frame whose hash differs from the last kept frame by at most this many "
                        "of 64 bits (default 4, -1 keeps every frame)")
    p.add_argument("--max-frames", type=int, default=0, help="at most this many frames per video (0: no limit)")
    p.add_argument("--group-by", choices=sources.GROUP_BY, default="directory",
                   help="capture group of images: their directory, or the file name without a trailing _<number>")

    p = commands.add_parser("accept", help="mark annotations as checked in bulk (e.g. labels made before checking)")
    p.add_argument("source")
    p.add_argument("images", nargs="*", help="image or annotation names of the source")
    p.add_argument("--all", action="store_true", help="every labeled image of the source")
    p.add_argument("--reason", default="",
                   help="accept images that wait for review of a new class anyway (recorded in the review list)")

    p = commands.add_parser("prelabel", help="write unchecked annotations for the unlabeled images of a source; a "
                            "second-stage model fills its attribute on the parent boxes")
    p.add_argument("model")
    p.add_argument("--source", required=True)
    p.add_argument("--run", help="prelabel with this completed run (defaults to the published model)")
    p.add_argument("--conf", type=float, default=0.25, help="confidence threshold (default 0.25)")
    p.add_argument("--device", help="inference device, e.g. 0 or cpu (defaults to TRAIN_DEVICE, then the run's device)")
    p.add_argument("--class", dest="class_", metavar="CLASS",
                   help="add CLASS candidates to the images that wait for review of CLASS (existing shapes stay)")

    p = commands.add_parser("prepare", help="build the dataset of a model from its checked annotations")
    p.add_argument("model")

    p = commands.add_parser("run", help="prepare the dataset, train a model on it, then evaluate it on the test split")
    p.add_argument("model")
    p.add_argument("--dataset", help="train on this prepared dataset instead of preparing one")
    p.add_argument("--option", action="append", default=[], metavar="KEY=VALUE",
                   help="override a train.options value for this run, e.g. epochs=3 or device=1 (recorded in run.json)")
    p.add_argument("--resume", metavar="RUN", help="continue an interrupted or failed run from its last checkpoint")

    p = commands.add_parser("eval", help="evaluate a completed run on the test split of a dataset")
    p.add_argument("model")
    p.add_argument("run")
    p.add_argument("--dataset", help="dataset id (defaults to the latest prepared, by train prepare or run)")
    p.add_argument("--errors", action="store_true",
                   help="classify: list the misclassified crops instead (most confident first)")
    p.add_argument("--split", choices=("test", "val", "train"), action="append",
                   help="split for --errors (repeat for several; default test)")
    p.add_argument("--device", help="inference device, e.g. 1 or cpu (defaults to TRAIN_DEVICE, then the run's device)")

    p = commands.add_parser("review", help="list reviews, or apply the labels fixed in one back to its sources")
    p.add_argument("action", nargs="?", choices=("apply",))
    p.add_argument("name", nargs="?")

    p = commands.add_parser("publish", help="gate, export and verify a run, then write it to models/<id>/")
    p.add_argument("model")
    p.add_argument("run", nargs="?", help="the run to publish; without it, list the completed runs")
    p.add_argument("--dataset", help="dataset whose test split the gate uses "
                                     "(defaults to the latest prepared, by train prepare or run)")
    p.add_argument("--accept-metrics", action="store_true",
                   help="publish although there is nothing to compare the metrics with (no evaluate.gate.min "
                        "and no comparable published model); failed gates are never accepted")
    p.add_argument("--device", help="device for the gate's evaluations (defaults to TRAIN_DEVICE, then the run's device)")

    p = commands.add_parser("update", help="change the sources or classes of a model, or show them",
                            description="Without an operation, show the classes and the state of every source.")
    p.add_argument("model")
    p.add_argument("operation", nargs="?", choices=("source", "add", "rename", "retire"),
                   help="source add|remove <source>: train on a source (labeled for the classes of the model from "
                        "then on), or stop; add <class>: append a class and uncheck the checked images of the "
                        "sources for review; rename <old> <new>: rename a class here, in the sources and in the "
                        "models sharing them; retire <class>: remove a class from this model (the sources keep it). On a "
                        "second-stage model the classes are values of its box attribute: add/rename/retire change "
                        "those values in its sources")
    p.add_argument("names", nargs="*", help="the names the operation takes")
    p.add_argument("--into", metavar="CLASS",
                   help="with retire: merge the class into CLASS in the sources too (relabels the annotations)")
    p.add_argument("--from", dest="from_class", metavar="CLASS",
                   help="with add: the new class used to be labeled as CLASS; only images with CLASS boxes are "
                        "unchecked")
    p.add_argument("--skip-source", action="append", default=[], metavar="SOURCE",
                   help="with add: add the class to SOURCE without unchecking anything (needs --reason)")
    p.add_argument("--reason", default="",
                   help="with --skip-source, or source add: why the checked images need no review for the classes "
                        "the source gains")

    args = parser.parse_args(argv)
    repo = (args.repo or paths.repo_root()).resolve()
    try:
        root = paths.train_root(repo, args.train_root)
        if args.command == "init":
            target = spec.path(repo, args.model)
            if target.exists():
                raise TrainError(f"{target} already exists")
            text = spec.skeleton(args.model, args.task, _classes(args), args.display_name, args.parent,
                                 args.parent_classes, args.attribute)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
            try:
                spec.load(repo, args.model)
            except TrainError:
                target.unlink()  # 不留下无法使用的规格：改正参数后重新 init
                raise
            if args.task == "classify":
                print(f"created {target}; train on a source of the parent with: train update {args.model} source "
                      "add <source>")
            else:
                print(f"created {target}; import material with: train import <source> <paths>, then: "
                      f"train update {args.model} source add <source>")
        elif args.command == "import":
            importer = sources.Importer(root, args.source, args.group_by, args.every, args.dedupe, args.max_frames)
            importer.run([path.expanduser().resolve() for path in args.paths])
            for note in importer.notes:
                print(f"  note: {note}")
            for key, count in sorted(importer.report.items()):
                print(f"  {key}: {count}")
            print(f"source {args.source}: {importer.directory} ({len(importer.record['images'])} images)")
            _print(labeling.refresh(repo, root, [args.source]))
        elif args.command == "accept":
            count, failed = sources.accept(root, args.source, args.images, args.all, args.reason)
            print(f"marked {count} annotations of {args.source} as checked")
            if failed:
                print(f"  left {len(failed)} images that failed the check in X-AnyLabeling as they are: fix them and "
                      "mark them passed there")
        elif args.command == "prepare":
            _prepare(repo, root, spec.load(repo, args.model))
        elif args.command == "prelabel" and spec.load(repo, args.model).get("cascade"):
            model = spec.load(repo, args.model)
            if args.class_:
                raise TrainError("--class prelabels a new class of a detection model; a second-stage model "
                                 "prelabels the attribute on the parent boxes")
            result = prelabel.prelabel_attribute(repo, root, model, args.source, args.run, args.conf, args.device)
            used = result.model
            name = f"release {used['release']} (run {used['run']})" if "release" in used else f"run {used['run']}"
            attribute = model["data"]["from_parent"]["attribute"]
            print(f"prelabeled {attribute} with {name}: {result.boxes} boxes filled, {result.unsure} left empty "
                  f"(below --conf {args.conf})")
            if result.direct:
                print(f"  wrote {result.direct} unchecked images of {args.source}: correct the boxes and {attribute} "
                      "in X-AnyLabeling and check each image")
            for reason, count in sorted(result.skipped.items()):
                print(f"  skipped {count} {reason}")
            if result.unlabeled:
                print(f"  {result.unlabeled} images of {args.source} have no annotation yet: prelabel or label the "
                      f"{model['cascade']['parent']} boxes first, then run this again")
            if result.direct:
                _print(labeling.refresh(repo, root, [args.source]))
            if result.proposals:
                review_name = f"{model['id']}-prelabel-{args.source}"
                target = review.propose(root, review_name, model["id"], used["run"], attribute, result.proposals)
                print(f"review {review_name}: {target}\n  open it in X-AnyLabeling and step through with D (the boxes "
                      f"are in {review.CHECKLIST}); correct and save, then: train review apply {review_name}\n"
                      "  apply writes back every image of the review: delete the .json of images you did not check")
            elif not result.direct and not result.unsure:
                print(f"  no boxes without {attribute}: nothing to prelabel")
        elif args.command == "prelabel":
            model = spec.load(repo, args.model)
            result = prelabel.prelabel(repo, root, model, args.source, args.run, args.conf, args.device,
                                       args.class_)
            used = result.model
            name = f"release {used['release']} (run {used['run']})" if "release" in used else f"run {used['run']}"
            kind = f"{args.class_} boxes added" if args.class_ else "boxes"
            print(f"prelabeled {result.written} images of {args.source} with {name}: {result.boxes} {kind}")
            for reason, count in sorted(result.skipped.items()):
                print(f"  skipped {count} images: {reason}")
            if result.written:
                print("  correct them in X-AnyLabeling and mark each finished image as checked; "
                      "unchecked prelabels never enter a dataset")
                children = [m for m, other in labeling._specs(repo)[0].items()
                            if (other.get("cascade") or {}).get("parent") == model["id"]
                            and args.source in other["data"]["sources"]]
                for child in sorted(children):
                    print(f"  before reviewing, fill the attributes of the second-stage models on these boxes: "
                          f"train prelabel {child} --source {args.source}")
        elif args.command == "run":
            model = spec.load(repo, args.model)
            if args.resume:
                if args.dataset or args.option:
                    raise TrainError("--resume continues with the run's own dataset and options; "
                                     "drop --dataset and --option")
                run = args.resume
                runs.resume(root, model, run)
            else:
                options = _options(args.option)
                if args.dataset:
                    dataset_id = args.dataset
                else:
                    # 数据集按内容命名：标注没变时复用同一个，所以每次训练前准备只花读取标注的时间，
                    # 又不会在勾选了新图片后悄悄用旧的数据集训练。
                    dataset_id = _prepare(repo, root, model)
                run = runs.start(repo, root, model, dataset_id, options, sys.argv[1:])
            print(f"run {run} completed: {paths.run_dir(root, model['id'], run)}")
            if runs.test_split_empty(root, model, run):
                print("  warning: the test split is empty, so the run is not evaluated "
                      "(import more capture groups, or raise data.split.test)")
            else:
                _evaluate(root, model, run)
        elif args.command == "eval":
            model = spec.load(repo, args.model)
            if args.errors:
                splits = list(dict.fromkeys(args.split or ["test"]))
                found = []
                for split_name in splits:
                    document = runs.errors(root, model, args.run, args.dataset, split_name, args.device)
                    print("\n".join(runs.errors_report(document, root, limit=10)))
                    print(f"  all of them: {paths.run_dir(root, model['id'], args.run) / runs.EVAL}/"
                          f"{document['dataset']}/errors-{split_name}.json")
                    found += document["errors"]
                if found:
                    name = f"{model['id']}-{args.run}-{'-'.join(splits)}"
                    target = review.create(root, name, model["id"], args.run, found)
                    print(f"review {name}: {target}\n  open it in X-AnyLabeling and step through with D (the boxes to "
                          f"check are in {review.CHECKLIST}); fix and save, then: train review apply {name}")
            else:
                if args.split:
                    raise TrainError("--split goes with --errors")
                _evaluate(root, model, args.run, args.dataset, args.device)
        elif args.command == "review":
            if args.action is None:
                if args.name:
                    raise TrainError("usage: train review [apply <name>]")
                print("\n".join(review.listing(root)))
            else:
                if not args.name:
                    raise TrainError("name the review to apply (train review lists them)")
                result = review.apply(root, args.name)
                print(f"review {args.name}: wrote {len(result.written)} annotations back, {result.unchanged} unchanged")
                for item in result.written:
                    print(f"  wrote {item}")
                for item in result.conflicts:
                    print(f"  conflict: {item} changed in its source since the review was made; not written "
                          "(fix it in the source directly)")
                if result.conflicts:
                    return 1
        elif args.command == "publish":
            model = spec.load(repo, args.model)
            if args.run is None:
                if args.accept_metrics or args.device:
                    raise TrainError("name the run to publish (train publish <id> without a run only lists them)")
                print("\n".join(publish.candidates(root, model, args.dataset, repo)))
                return 0
            result = publish.publish(repo, root, model, args.run, args.dataset, args.accept_metrics, args.device)
            print("\n".join(publish.report(result)))
        elif args.command == "update":
            model = spec.load(repo, args.model)
            if model.get("cascade") and args.skip_source:
                raise TrainError(f"--skip-source is for detection models: the classes of {args.model} are values of "
                                 f"the box attribute {model['data']['from_parent']['attribute']}")
            arity = {"source": 2, "add": 1, "rename": 2, "retire": 1, None: 0}[args.operation]
            adding = args.from_class or args.skip_source
            usage = ("usage: train update <id> [source add|remove <source> [--reason <text>] | add <class> "
                     "[--from <class>] [--skip-source <source> --reason <text>] | rename <old> <new> | "
                     "retire <class> [--into <class>]]")
            if len(args.names) != arity or (args.into and args.operation != "retire") or \
                    (adding and args.operation != "add") or \
                    (args.operation == "source" and args.names[0] not in ("add", "remove")) or \
                    (args.reason and args.operation not in ("add", "source")):
                raise TrainError(usage)
            if args.operation == "add" and bool(args.skip_source) != bool(args.reason):
                raise TrainError("--skip-source and --reason go together")
            if args.operation == "source" and args.reason and args.names[0] != "add":
                raise TrainError(usage)
            result = None
            if args.operation == "source" and args.names[0] == "add":
                result = update.add_source(repo, root, model, args.names[1], args.reason)
            elif args.operation == "source":
                result = update.remove_source(repo, root, model, args.names[1])
            elif args.operation == "add" and model.get("cascade"):
                result = update.add_value(repo, root, model, args.names[0], args.from_class)
            elif args.operation == "add":
                skip = {source: args.reason for source in args.skip_source}
                result = update.add(repo, root, model, args.names[0], args.from_class, skip)
            elif args.operation == "rename" and model.get("cascade"):
                result = update.rename_value(repo, root, model, *args.names)
            elif args.operation == "rename":
                result = update.rename(repo, root, model, *args.names)
            elif args.operation == "retire" and model.get("cascade"):
                result = update.retire_value(repo, root, model, args.names[0], args.into)
            elif args.operation == "retire":
                result = update.retire(repo, root, model, args.names[0], args.into)
            if result is not None:
                # 类别或来源变了：重新生成相关来源中的类别与属性文件（不再使用的随之删除）。
                print("\n".join(result.lines + labeling.refresh(
                    repo, root, [*result.sources, *_related_sources(repo, args.model)])))
            else:
                print("\n".join(update.status(root, model)))
    except (TrainError, DeployError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    return 0
