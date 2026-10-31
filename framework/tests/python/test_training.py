"""训练工具：规格、导入、X-AnyLabeling 读写、批量确认、划分、数据集准备，以及训练与评估的记录。

用真实的 Pillow 与 ffmpeg（builder 中已安装），以及 X-AnyLabeling 实际保存的样例文件。训练与评估用
假后端：这里验证 run 的记录、续训与评估的组织；Ultralytics 本身只在训练机上验证。
"""

import io
import json
import shutil
import subprocess
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from deploy_support import ROOT
from dsf_deploy import models
from dsf_train import backends, cli, dataset, paths, runs, sources, spec, split, xanylabeling
from dsf_train.backends import ultralytics
from dsf_train.errors import TrainError

SAMPLES = Path(__file__).resolve().parent / "data/x-anylabeling-4.0.6"
NEGATIVE = SAMPLES / "20260808_102138_00023000.json"
LABELED = SAMPLES / "20260808_102827_00058000.json"   # 一个“标志牌(圆牌)”矩形框，带属性
CLASSES = ["标志牌(圆牌)", "龙门架A", "防撞桶"]


def image(path: Path, color: str, size=(1920, 1080)) -> Path:
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


def annotation(path: Path, template: Path, checked: bool, **changes) -> Path:
    """以真实样例为模板写一份标注（imagePath 指向同名图像）。"""
    document = json.loads(template.read_text(encoding="utf-8"))
    document.update({"imagePath": path.with_suffix(".jpg").name, "checked": checked, **changes})
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


class TrainingTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="dsf-train-")).resolve()
        self.repo = self.tmp / "repo"
        self.root = self.tmp / "training"
        self.raw = self.tmp / "raw"
        (self.repo / "models/m").mkdir(parents=True)
        spec.path(self.repo, "m").write_text(spec.skeleton("m", "detect", CLASSES), encoding="utf-8")
        self.set_data(sources=["batch-01"], ignore_classes=["待定"])

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def set_data(self, **data):
        path = spec.path(self.repo, "m")
        text = path.read_text(encoding="utf-8")
        for key, value in data.items():
            old = next(line for line in text.splitlines() if line.startswith(f"  {key}:"))
            text = text.replace(old, f"  {key}: {json.dumps(value, ensure_ascii=False)}")
        path.write_text(text, encoding="utf-8")

    def spec(self):
        return spec.load(self.repo, "m")

    def train(self, *argv):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = cli.main(["--repo", str(self.repo), "--train-root", str(self.root), *argv])
        return code, stdout.getvalue(), stderr.getvalue()

    def import_(self, *paths_, source="batch-01", **options):
        importer = sources.Importer(self.root, source, CLASSES, **options)
        importer.run([Path(p) for p in paths_])
        return importer

    def source(self, name="batch-01"):
        return paths.source_dir(self.root, name)


class SpecTest(TrainingTestCase):
    def test_skeleton_is_a_valid_spec(self):
        model = self.spec()
        self.assertEqual(model["classes"], CLASSES)
        self.assertEqual(models.check_spec("m", model), [])  # 也满足打包时的部署契约

    def test_class_names_that_need_quoting_stay_strings(self):
        names = ["1", "yes", "a: b", " padded", "标志牌(绿底矩形)"]
        text = spec.skeleton("n", "detect", names)
        (self.repo / "models/n").mkdir()
        spec.path(self.repo, "n").write_text(text, encoding="utf-8")
        self.assertEqual(spec.load(self.repo, "n")["classes"], names)

    def test_tasks_that_are_not_implemented_are_reported_early(self):
        with self.assertRaisesRegex(TrainError, "task segment is not supported yet"):
            spec.skeleton("n", "segment", ["a"])
        with self.assertRaisesRegex(TrainError, "unknown task 'track'"):
            spec.skeleton("n", "track", ["a"])
        path = spec.path(self.repo, "m")
        path.write_text(path.read_text().replace("task: detect", "task: classify") +
                        "cascade: {parent: x, classes: [a]}\n")
        with self.assertRaisesRegex(TrainError, "task classify is not supported yet(.|\n)*cascade"):
            self.spec()

    def test_init_creates_the_spec_once(self):
        code, out, _ = self.train("init", "new", "--task", "detect", "--class", "a", "--class", "b")
        self.assertEqual(code, 0, out)
        self.assertEqual(spec.load(self.repo, "new")["classes"], ["a", "b"])
        code, _, err = self.train("init", "new", "--task", "detect", "--class", "a")
        self.assertEqual((code, "already exists" in err), (1, True))
        code, _, err = self.train("init", "other", "--task", "detect")
        self.assertIn("give at least one class", err)

    def test_split_fractions_are_checked(self):
        path = spec.path(self.repo, "m")
        path.write_text(path.read_text().replace("val: 0.15", "val: 0.9"))
        with self.assertRaisesRegex(TrainError, "val and test must be fractions whose sum is below 1"):
            self.spec()


class ReaderTest(unittest.TestCase):
    def test_reads_the_files_saved_by_x_anylabeling(self):
        negative = xanylabeling.read(NEGATIVE)
        self.assertEqual((negative.width, negative.height, negative.shapes, negative.checked), (1920, 1080, [], False))
        labeled = xanylabeling.read(LABELED)
        [shape] = labeled.shapes
        self.assertEqual((shape.label, shape.shape_type, len(shape.points)), ("标志牌(圆牌)", "rectangle", 4))
        self.assertEqual(shape.attributes, {"遮挡": "否", "脏污": "否", "破损": "否", "歪斜": "否"})

    def test_update_keeps_every_other_field(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "a.json"
            shutil.copy(LABELED, path)
            before = json.loads(path.read_text())
            xanylabeling.update(path, checked=True)
            after = json.loads(path.read_text())
            self.assertEqual(list(after), list(before))  # 键的顺序不变
            self.assertEqual(after, {**before, "checked": True})


class ImportTest(TrainingTestCase):
    def test_images_are_deduplicated_and_bring_their_annotations(self):
        image(self.raw / "cam1/20260808_102827_00000001.jpg", "red")
        image(self.raw / "cam1/20260808_102827_00000002.jpg", "blue")
        annotation(self.raw / "cam1/20260808_102827_00000001.json", LABELED, checked=True)
        shutil.copy(self.raw / "cam1/20260808_102827_00000001.jpg", self.raw / "copy.jpg")
        (self.raw / "broken.jpg").write_bytes(b"not an image")
        (self.raw / "notes.txt").write_text("x")
        importer = self.import_(self.raw)
        self.assertEqual(importer.report["images imported"], 2)
        self.assertEqual(importer.report["duplicates skipped"], 1)
        self.assertEqual(importer.report["unreadable images skipped"], 1)
        self.assertEqual(importer.report["other files skipped"], 1)
        record = sources.read_record(self.source())
        info = record["images"]["20260808_102827_00000001.jpg"]
        self.assertEqual((info["width"], info["height"], info["group"]), (1920, 1080, "dir:batch-01/cam1"))
        self.assertEqual(info["origin"], str(self.raw / "cam1/20260808_102827_00000001.jpg"))
        imported = xanylabeling.read(self.source() / "20260808_102827_00000001.json")
        self.assertEqual((imported.image, imported.checked), ("20260808_102827_00000001.jpg", True))
        self.assertEqual((self.source() / paths.SOURCE_LABELS).read_text(encoding="utf-8").splitlines(), CLASSES)
        # 同一张图不能进入另一个来源（可能一边训练一边测试）
        again = self.import_(self.raw / "cam1", source="batch-02")
        self.assertEqual((again.report["images imported"], again.report["duplicates skipped"]), (0, 2))

    def test_capture_groups(self):
        image(self.raw / "20260808_102138_00023000.jpg", "red")
        image(self.raw / "20260808_102138_00024000.jpg", "blue")
        image(self.raw / "single.jpg", "green")
        self.import_(self.raw, group_by="filename_prefix")
        groups = {n: i["group"] for n, i in sources.read_record(self.source())["images"].items()}
        self.assertEqual(groups, {"20260808_102138_00023000.jpg": "prefix:20260808_102138",
                                  "20260808_102138_00024000.jpg": "prefix:20260808_102138",
                                  "single.jpg": "prefix:single"})

    def test_same_name_from_different_directories(self):
        image(self.raw / "a/frame.jpg", "red")
        image(self.raw / "b/frame.jpg", "blue")
        self.import_(self.raw)
        names = sorted(sources.read_record(self.source())["images"])
        self.assertEqual(names[0], "frame.jpg")
        self.assertRegex(names[1], r"^frame_[0-9a-f]{8}\.jpg$")

    def test_video_frames_are_sampled_and_near_duplicates_dropped(self):
        video = self.raw / "clip.mp4"
        video.parent.mkdir(parents=True)
        # 前 3 秒静止画面，后 3 秒每秒一种颜色：静止部分只保留一帧
        colors = "color=c=gray:s=320x240:d=3,format=yuv420p[a];" \
                 "color=c=red:s=320x240:d=1[r];color=c=blue:s=320x240:d=1[b];color=c=green:s=320x240:d=1[g];" \
                 "[r][b][g]concat=n=3,format=yuv420p,drawgrid=w=40:h=40:c=white[c];[a][c]concat=n=2[v]"
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "lavfi", "-i", "nullsrc",
                        "-filter_complex", colors, "-map", "[v]", "-r", "10", str(video)], check=True)
        importer = self.import_(video)
        record = sources.read_record(self.source())
        frames = sorted(record["images"])
        self.assertEqual(frames[0], "clip_00000000.jpg")
        self.assertGreaterEqual(importer.report["similar frames skipped"], 2)
        self.assertEqual(len({record["images"][f]["group"] for f in frames}), 1)
        self.assertTrue(record["images"][frames[0]]["group"].startswith("video:"))
        self.assertEqual(record["images"]["clip_00000000.jpg"]["time_ms"], 0)
        self.assertFalse(any(p.name.startswith(".dsf-frames") for p in self.source().iterdir()))
        limited = self.import_(video, source="batch-02", dedupe=-1, max_frames=2)
        self.assertEqual(limited.report["video frames imported"], 0)  # 已全部导入过的帧按内容去重
        self.assertGreater(limited.report["duplicates skipped"], 0)


class AcceptTest(TrainingTestCase):
    def setUp(self):
        super().setUp()
        for index in range(3):
            image(self.raw / f"f_{index}.jpg", ["red", "blue", "green"][index])
        annotation(self.raw / "f_0.json", LABELED, checked=False)
        annotation(self.raw / "f_1.json", NEGATIVE, checked=False)
        self.import_(self.raw)

    def test_accept_named_images(self):
        code, out, _ = self.train("accept", "batch-01", "f_0")
        self.assertEqual((code, out.strip()), (0, "marked 1 annotations of batch-01 as checked"))
        self.assertTrue(xanylabeling.read(self.source() / "f_0.json").checked)
        self.assertFalse(xanylabeling.read(self.source() / "f_1.json").checked)
        record = sources.read_record(self.source())
        self.assertEqual(record["accepted"][0]["images"], ["f_0.jpg"])
        code, _, err = self.train("accept", "batch-01", "f_2.jpg")
        self.assertEqual(code, 1)
        self.assertIn("have no annotation yet, label them first: f_2.jpg", err)

    def test_accept_all_skips_unlabeled_images(self):
        self.assertEqual(sources.accept(self.root, "batch-01", [], everything=True), 2)
        self.assertEqual(sources.accept(self.root, "batch-01", [], everything=True), 0)
        with self.assertRaisesRegex(TrainError, "name the images to accept, or pass --all"):
            sources.accept(self.root, "batch-01", [])


class SplitTest(unittest.TestCase):
    def test_assignment_depends_only_on_the_group(self):
        groups = [f"video:{i:04d}" for i in range(4000)]
        first = [split.assign(g, 42, 0.15, 0.15) for g in groups]
        self.assertEqual(first, [split.assign(g, 42, 0.15, 0.15) for g in groups])
        for name, expected in (("train", 0.7), ("val", 0.15), ("test", 0.15)):
            self.assertAlmostEqual(first.count(name) / len(groups), expected, delta=0.03)
        self.assertNotEqual(first, [split.assign(g, 7, 0.15, 0.15) for g in groups])


class PrepareTest(TrainingTestCase):
    def setUp(self):
        super().setUp()
        image(self.raw / "a/f_0.jpg", "red")
        image(self.raw / "a/f_1.jpg", "blue")
        image(self.raw / "a/f_2.jpg", "green")
        annotation(self.raw / "a/f_0.json", LABELED, checked=True)
        annotation(self.raw / "a/f_1.json", NEGATIVE, checked=True)
        annotation(self.raw / "a/f_2.json", LABELED, checked=False)

    def prepare(self):
        return dataset.prepare(self.root, self.spec())

    def test_only_checked_images_enter_the_dataset(self):
        self.import_(self.raw)
        result = self.prepare()
        samples = {s["image"]: s for s in result.manifest["samples"]}
        self.assertEqual(sorted(samples), ["f_0.jpg", "f_1.jpg"])
        self.assertEqual(samples["f_1.jpg"]["objects"], [])  # 已检查的负样本
        [box] = samples["f_0.jpg"]["objects"]
        self.assertEqual(box["class"], 0)
        self.assertEqual(box["box"], [1345.79, 456.12, 1379.87, 491.62])
        self.assertEqual(box["attributes"]["遮挡"], "否")
        self.assertEqual(result.skipped[("batch-01", "not checked")], 1)
        lines = dataset.report(result, self.spec())
        self.assertIn("  skipped 1 images of batch-01: not checked", lines)

    def test_the_view_is_a_yolo_dataset_of_hard_links(self):
        self.import_(self.raw)
        result = self.prepare()
        view = result.directory / paths.VIEW / "ultralytics"
        split_name = result.manifest["samples"][0]["split"]
        labels = (view / "labels" / split_name / "batch-01__f_0.txt").read_text().split()
        self.assertEqual(labels[0], "0")
        self.assertAlmostEqual(float(labels[1]), (1345.79 + 1379.87) / 2 / 1920, places=5)
        self.assertEqual((view / "labels" / split_name / "batch-01__f_1.txt").read_text(), "")
        linked = view / "images" / split_name / "batch-01__f_0.jpg"
        self.assertEqual(linked.stat().st_ino, (self.source() / "f_0.jpg").stat().st_ino)
        data_yaml = (view / "data.yaml").read_text(encoding="utf-8")
        self.assertIn(f"path: {view}\n", data_yaml)
        self.assertIn('  0: "标志牌(圆牌)"\n', data_yaml)
        self.assertEqual((self.root / "datasets/m/latest").read_text().strip(), result.dataset_id)

    def test_the_identity_follows_the_content(self):
        self.import_(self.raw)
        first = self.prepare()
        again = self.prepare()
        self.assertEqual((again.dataset_id, again.reused), (first.dataset_id, True))
        document = json.loads((self.source() / "f_0.json").read_text())
        document["shapes"][0]["points"][0][0] -= 5
        (self.source() / "f_0.json").write_text(json.dumps(document, ensure_ascii=False))
        changed = self.prepare()
        self.assertNotEqual(changed.dataset_id, first.dataset_id)
        self.assertFalse(changed.reused)

    def test_new_sources_do_not_move_existing_samples(self):
        self.import_(self.raw)
        before = {s["image"]: s["split"] for s in self.prepare().manifest["samples"]}
        for index in range(20):
            path = image(self.tmp / f"more/g{index}/x_{index}.jpg", "#%06x" % (index * 4000))
            annotation(path.with_suffix(".json"), NEGATIVE, checked=True)
        self.import_(self.tmp / "more", source="batch-02")
        self.set_data(sources=["batch-01", "batch-02"])
        after = {s["image"]: s["split"] for s in self.prepare().manifest["samples"] if s["source"] == "batch-01"}
        self.assertEqual(after, before)

    def test_annotation_problems_are_listed_together(self):
        annotation(self.raw / "a/f_0.json", LABELED, checked=True,
                   shapes=[{"label": "标志牌(方牌)", "points": [[1, 1], [9, 9]], "shape_type": "rectangle"},
                           {"label": "龙门架A", "points": [[1, 1], [5000, 9]], "shape_type": "rectangle"},
                           {"label": "待定", "points": [[1, 1], [9, 9]], "shape_type": "rectangle"},
                           {"label": "防撞桶", "points": [[1, 1], [9, 1], [9, 9]], "shape_type": "polygon"}])
        self.import_(self.raw)
        with self.assertRaisesRegex(TrainError, "2 problems in the checked annotations(.|\n)*"
                                                "batch-01/f_0.jpg: unknown class '标志牌\\(方牌\\)'(.|\n)*"
                                                "batch-01/f_0.jpg: 龙门架A: box .* is outside the 1920x1080 image"):
            self.prepare()
        annotation(self.source() / "f_0.json", LABELED, checked=True,
                   shapes=[{"label": "待定", "points": [[1, 1], [9, 9]], "shape_type": "rectangle"},
                           {"label": "防撞桶", "points": [[1, 1], [9, 1], [9, 9]], "shape_type": "polygon"}])
        result = self.prepare()
        self.assertEqual(result.ignored_shapes, {"polygon": 1})  # 忽略的类别与其他任务的形状不是错误
        self.assertEqual(next(s for s in result.manifest["samples"] if s["image"] == "f_0.jpg")["objects"], [])

    def test_images_changed_after_import_are_rejected(self):
        self.import_(self.raw)
        image(self.source() / "f_1.jpg", "white")  # 链接被替换成另一张图
        with self.assertRaisesRegex(TrainError, "batch-01/f_1.jpg: the image changed after it was imported"):
            self.prepare()

    def test_nothing_checked_or_no_sources(self):
        self.set_data(sources=[])
        with self.assertRaisesRegex(TrainError, "data.sources of m is empty"):
            self.prepare()
        self.set_data(sources=["batch-01"])
        with self.assertRaisesRegex(TrainError, "batch-01 is not a source"):
            self.prepare()
        image(self.tmp / "other/x.jpg", "red")
        self.import_(self.tmp / "other")
        with self.assertRaisesRegex(TrainError, "no checked images in batch-01"):
            self.prepare()


class FakeBackend(types.ModuleType):
    """记录调用的假后端；fail 设为异常时训练在写出 last 之后抛出它。"""
    NAME = "fake"
    TASKS = ("detect",)
    BEST = ultralytics.BEST
    LAST = ultralytics.LAST
    RESERVED = ultralytics.RESERVED

    def __init__(self):
        super().__init__("fake")
        self.calls, self.fail = [], None

    write_view = staticmethod(ultralytics.write_view)
    check_options = staticmethod(ultralytics.check_options)

    def version(self):
        return {"fake": "1.0"}

    def train(self, task, data, weights, options, directory, resume, workdir):
        self.calls.append(("train", weights, dict(options), resume))
        if not weights.exists() and weights.parent == workdir:  # 模拟下载预训练权重
            workdir.mkdir(parents=True, exist_ok=True)
            weights.write_bytes(b"pretrained")
        (directory / "weights").mkdir(parents=True, exist_ok=True)
        (directory / self.LAST).write_bytes(b"last")
        if self.fail is not None:
            error, self.fail = self.fail, None
            raise error
        (directory / self.BEST).write_bytes(f"best {data}".encode())
        return directory / self.BEST

    def evaluate(self, task, weights, data, split_name, options, directory, workdir):
        self.calls.append(("evaluate", weights, data, split_name))
        directory.mkdir(parents=True, exist_ok=True)
        return {"metrics": {"map50_95": 0.5, "map50": 0.7, "precision": 0.8, "recall": 0.6},
                "classes": {0: {"map50_95": 0.5, "map50": 0.7, "precision": 0.8, "recall": 0.6}}}


class RunTest(TrainingTestCase):
    def setUp(self):
        super().setUp()
        self.backend = FakeBackend()
        backends.BACKENDS["fake"] = self.backend
        path = spec.path(self.repo, "m")
        path.write_text(path.read_text().replace("backend: ultralytics", "backend: fake"))
        # 选出三个划分各至少一组的采集组，每组一张带框、已检查的图片。
        covered, index = set(), 0
        while covered != set(split.SPLITS):
            group = f"g{index}"
            name = split.assign(f"dir:batch-01/{group}", 42, 0.15, 0.15)
            if name not in covered or index < 6:
                covered.add(name)
                image(self.raw / group / f"{group}.jpg", (index * 7 % 256, index, 0))
                annotation(self.raw / group / f"{group}.json", LABELED, checked=True)
            index += 1
        self.import_(self.raw)
        self.dataset_id = dataset.prepare(self.root, self.spec()).dataset_id

    def tearDown(self):
        del backends.BACKENDS["fake"]
        super().tearDown()

    def record(self, run):
        return runs.load(self.root, "m", run)[1]

    def only_run(self):
        [run] = [p.name for p in (self.root / "runs/m").iterdir()]
        return run

    def test_a_run_is_recorded_and_evaluated_on_the_test_split(self):
        code, out, err = self.train("run", "m", "--option", "epochs=3")
        self.assertEqual(code, 0, err)
        run = self.only_run()
        record = self.record(run)
        self.assertEqual((record["status"], record["dataset"], record["classes"]), ("completed", self.dataset_id, CLASSES))
        self.assertEqual(record["options"]["epochs"], 3)          # 覆盖值生效并记录
        self.assertEqual(record["overrides"], {"epochs": 3})
        self.assertEqual(record["options"]["patience"], 50)       # 其余来自 model.yaml
        self.assertEqual(record["backend"], {"name": "fake", "fake": "1.0"})
        pretrained = self.root / paths.PRETRAINED / "yolo11s.pt"
        self.assertEqual(record["weights"]["start_sha256"], paths.sha256(pretrained))  # 下载后补记
        self.assertEqual(record["weights"]["best_sha256"],
                         paths.sha256(self.root / "runs/m" / run / "train" / ultralytics.BEST))
        self.assertEqual((self.root / "runs/m" / run / "model.yaml").read_text(), spec.path(self.repo, "m").read_text())
        self.assertRegex(run, r"^\d{8}-\d{6}-[0-9a-f]{6}$")

        [_, (_, weights, data, split_name)] = self.backend.calls
        self.assertEqual((weights.name, split_name), ("best.pt", "test"))
        self.assertEqual(data, self.root / "datasets/m" / self.dataset_id / "view/fake/data.yaml")
        metrics = json.loads((self.root / "runs/m" / run / "eval" / self.dataset_id / "metrics.json").read_text())
        test = [s for s in json.loads((self.root / "datasets/m" / self.dataset_id / "manifest.json").read_text())
                ["samples"] if s["split"] == "test"]
        self.assertEqual(metrics["classes"][0], {"name": CLASSES[0], "instances": len(test), "map50_95": 0.5,
                                                 "map50": 0.7, "precision": 0.8, "recall": 0.6})
        self.assertEqual(metrics["classes"][1]["map50"], None)    # 测试划分中没有的类别不给指标
        self.assertEqual(self.record(run)["evaluations"][self.dataset_id]["map50"], 0.7)
        self.assertIn("not evaluated (no test instances): 龙门架A, 防撞桶", out)

    def test_interrupted_and_failed_runs_are_recorded_and_resumable(self):
        self.backend.fail = KeyboardInterrupt()
        code, out, _ = self.train("run", "m")
        self.assertEqual(code, 130)
        run = self.only_run()
        self.assertIn(f"train run m --resume {run}", out)
        self.assertEqual(self.record(run)["status"], "interrupted")
        self.assertIn("--resume continues with the run's own dataset",
                      self.train("run", "m", "--resume", run, "--option", "epochs=1")[2])
        self.backend.fail = RuntimeError("CUDA out of memory")
        with self.assertRaisesRegex(RuntimeError, "CUDA out of memory"):
            runs.resume(self.root, self.spec(), run)
        record = self.record(run)
        self.assertEqual((record["status"], record["error"]), ("failed", "RuntimeError: CUDA out of memory"))
        code, _, err = self.train("run", "m", "--resume", run)
        self.assertEqual(code, 0, err)
        record = self.record(run)
        self.assertEqual((record["status"], record["error"], len(record["resumed_at"])), ("completed", "", 2))
        resumed = [call for call in self.backend.calls if call[0] == "train"][-1]
        self.assertEqual((resumed[1], resumed[3]), (self.root / "runs/m" / run / "train" / ultralytics.LAST, True))
        code, _, err = self.train("run", "m", "--resume", run)
        self.assertIn("already completed", err)

    def test_reserved_options_and_unknown_gate_metrics_are_rejected(self):
        code, _, err = self.train("run", "m", "--option", "project=/tmp/elsewhere")
        self.assertIn("project is set by the training tool", err)
        self.assertFalse((self.root / "runs").exists())           # 预检失败不创建 run
        path = spec.path(self.repo, "m")
        text = path.read_text()
        path.write_text(text.replace("    patience: 50", "    patience: 50\n    data: other.yaml")
                        .replace("min: {}", "min: {map75: 0.5}"))
        with self.assertRaisesRegex(TrainError, "train.options: data is set(.|\n)*unknown metric 'map75'"):
            self.spec()

    def test_weights_can_continue_from_another_run(self):
        self.assertEqual(self.train("run", "m")[0], 0)
        first = self.only_run()
        path = spec.path(self.repo, "m")
        path.write_text(path.read_text().replace("weights: yolo11s.pt", f"weights: run:{first}"))
        self.assertEqual(self.train("run", "m")[0], 0)
        second = next(p.name for p in (self.root / "runs/m").iterdir() if p.name != first)
        record = self.record(second)
        best = self.root / "runs/m" / first / "train" / ultralytics.BEST
        self.assertEqual((record["weights"]["start"], record["weights"]["start_sha256"]),
                         (f"run:{first}", paths.sha256(best)))
        path.write_text(path.read_text().replace(f"run:{first}", "sub/dir.pt"))
        self.assertIn("must be a pretrained file name", self.train("run", "m")[2])

    def test_an_old_run_is_evaluated_on_a_newer_dataset(self):
        self.assertEqual(self.train("run", "m")[0], 0)
        run = self.only_run()
        image(self.tmp / "more/x/x.jpg", "white")
        annotation(self.tmp / "more/x/x.json", LABELED, checked=True)
        self.import_(self.tmp / "more")
        newer = dataset.prepare(self.root, self.spec()).dataset_id
        code, out, err = self.train("eval", "m", run)
        self.assertEqual(code, 0, err)
        self.assertEqual(sorted(self.record(run)["evaluations"]), sorted([self.dataset_id, newer]))
        self.assertIn(f"on the test split of dataset {newer}", out)

    def test_the_dataset_must_match_the_classes(self):
        path = spec.path(self.repo, "m")
        path.write_text(path.read_text().replace("  - 防撞桶\n", "  - 防撞桶\n  - 新类别\n"))
        code, _, err = self.train("run", "m")
        self.assertIn("prepared with other classes", err)
        code, _, err = self.train("run", "m", "--dataset", "nope")
        self.assertIn("dataset 'nope' of m does not exist", err)


class RepositoryModelTest(unittest.TestCase):
    def test_repository_models_are_valid_training_specs(self):
        # 构建镜像的上下文不含 models/：这里在开发机上覆盖仓库中的真实规格。
        for directory in models.model_dirs(ROOT):
            spec.load(ROOT, directory.name)


if __name__ == "__main__":
    unittest.main()
