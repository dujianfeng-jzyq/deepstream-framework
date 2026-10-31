"""命令行入口：init / import / accept / prepare。由 framework/training/dsf-train.py 调用，
通常经 scripts/train.sh 在训练镜像中运行。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dsf_deploy.errors import DeployError

from . import dataset, paths, sources, spec
from .errors import TrainError


def _classes(args: argparse.Namespace) -> list[str]:
    classes = list(args.class_ or [])
    if args.classes_file:
        classes += [line.strip() for line in Path(args.classes_file).read_text(encoding="utf-8").splitlines()
                    if line.strip()]
    return classes


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

    p = commands.add_parser("import", help="import images and videos into a source for labeling")
    p.add_argument("model", help="the model whose classes go into the source's labels.txt")
    p.add_argument("paths", nargs="+", type=Path, help="image or video files, or directories of them")
    p.add_argument("--source", required=True, help="source name under <train-root>/sources/ (created or extended)")
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

    p = commands.add_parser("prepare", help="build the dataset of a model from its checked annotations")
    p.add_argument("model")

    args = parser.parse_args(argv)
    repo = (args.repo or paths.repo_root()).resolve()
    try:
        root = paths.train_root(repo, args.train_root)
        if args.command == "init":
            target = spec.path(repo, args.model)
            if target.exists():
                raise TrainError(f"{target} already exists")
            text = spec.skeleton(args.model, args.task, _classes(args), args.display_name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
            spec.load(repo, args.model)
            print(f"created {target}; import material with: train import {args.model} <paths> --source <name>")
        elif args.command == "import":
            model = spec.load(repo, args.model)
            importer = sources.Importer(root, args.source, model["classes"], args.group_by,
                                        args.every, args.dedupe, args.max_frames)
            importer.run([path.expanduser().resolve() for path in args.paths])
            for note in importer.notes:
                print(f"  note: {note}")
            for key, count in sorted(importer.report.items()):
                print(f"  {key}: {count}")
            print(f"source {args.source}: {importer.directory} ({len(importer.record['images'])} images)")
            print("  label it in X-AnyLabeling (load labels.txt as the predefined labels) and mark finished "
                  "images as checked")
            if args.source not in model["data"]["sources"]:
                print(f"  add {args.source} to data.sources in {spec.path(repo, args.model)} to train on it")
        elif args.command == "accept":
            count = sources.accept(root, args.source, args.images, args.all)
            print(f"marked {count} annotations of {args.source} as checked")
        elif args.command == "prepare":
            model = spec.load(repo, args.model)
            result = dataset.prepare(root, model)
            print("\n".join(dataset.report(result, model)))
    except (TrainError, DeployError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    return 0
