"""训练工具：规格、导入、X-AnyLabeling 读写、批量确认、划分、数据集准备，训练与评估的记录，预标注与发布。

用真实的 Pillow、ffmpeg 与 numpy（builder 中已安装），以及 X-AnyLabeling 实际保存的样例文件。训练、评估、
推理与导出用假后端，ONNX 的签名与推理也用替身：这里验证记录、门槛、一致性判断与模型目录的写入；
Ultralytics、onnx 与 onnxruntime 本身只在训练机上验证。
"""

import io
import json
import os
import shutil
import subprocess
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from deploy_support import ROOT
from dsf_deploy import models
from dsf_train import backends, cli, dataset, media, paths, publish, runs, sources, spec, split, xanylabeling
from dsf_train.backends import ultralytics
from dsf_train.tasks import detect
from dsf_train.errors import TrainError

SAMPLES = Path(__file__).resolve().parent / "data/x-anylabeling-4.0.6"
NEGATIVE = SAMPLES / "20260808_102138_00023000.json"
LABELED = SAMPLES / "20260808_102827_00058000.json"   # 一个“标志牌(圆牌)”矩形框，带属性
CLASSES = ["标志牌(圆牌)", "龙门架A", "防撞桶"]

try:
    import numpy
except ModuleNotFoundError:  # 宿主可能没有；builder 中有，DSF_FORBID_SKIPS 使跳过计为失败
    numpy = None
needs_numpy = unittest.skipUnless(numpy is not None, "numpy not installed")


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


def new_file_mode() -> int:
    umask = os.umask(0)
    os.umask(umask)
    return 0o666 & ~umask


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
            self.assertEqual(path.stat().st_mode & 0o777, new_file_mode())  # 不因原子替换变成 0600


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
        self.during_predict = None  # 推理每张图之前调用，模拟人同时在工具中保存
        self.map50_95 = {}          # run → 评估得到的 map50_95（默认 0.5）

    LAYOUTS = ultralytics.LAYOUTS

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
        (directory / self.BEST).write_bytes(f"best {data} {directory}".encode())  # 每次 run 的权重不同
        return directory / self.BEST

    def evaluate(self, task, weights, data, split_name, options, directory, workdir):
        self.calls.append(("evaluate", weights, data, split_name, options.get("device")))
        directory.mkdir(parents=True, exist_ok=True)
        value = self.map50_95.get(weights.parents[2].name, 0.5)  # runs/m/<run>/train/weights/best.pt
        return {"metrics": {"map50_95": value, "map50": 0.7, "precision": 0.8, "recall": 0.6},
                "classes": {0: {"map50_95": value, "map50": 0.7, "precision": 0.8, "recall": 0.6}}}

    def export(self, task, weights, layout, size, settings, target, workdir):
        self.calls.append(("export", weights, layout, size, dict(settings)))
        target.write_bytes(b"onnx of " + weights.read_bytes())

    def reference(self, task, weights, layout, batch):
        """每张图 4 个候选：坐标来自图像均值，两个高分、两个低分。"""
        rows = []
        for item in batch:
            mean = float(item.mean()) * 100
            rows.append([[mean, mean, mean + 10, mean + 10, 0.9, 1.0], [1, 2, 3, 4, 0.3, 2.0],
                         [5, 6, 7, 8, 0.02, 0.0], [0, 0, 1, 1, 0.01, 1.0]])
        return numpy.array(rows, dtype=numpy.float32)


    def predict(self, task, weights, images, options, conf, workdir):
        self.calls.append(("predict", weights, list(images), dict(options), conf))
        for path in images:
            if self.during_predict:
                self.during_predict(path)
            # 一个越界的框（裁剪到图像内）与一个龙门架A。
            yield [{"class": 0, "box": [1345.8, 456.1, 1379.9, 491.6], "score": 0.9},
                   {"class": 1, "box": [-3.0, 10.0, 400.0, 1100.0], "score": 0.4}]


class RunTestCase(TrainingTestCase):
    """假后端与一个三个划分都有样本的数据集。"""
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


class RunTest(RunTestCase):
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

        [_, (_, weights, data, split_name, _)] = self.backend.calls
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


class PrelabelTest(RunTestCase):
    def setUp(self):
        super().setUp()
        self.assertEqual(self.train("run", "m")[0], 0)
        self.run_id = self.only_run()
        for index in range(3):
            image(self.tmp / "batch-02" / f"n_{index}.jpg", (0, 0, 100 + index * 40))
        annotation(self.tmp / "batch-02/n_2.json", NEGATIVE, checked=False)  # 人已经开始标的图
        self.import_(self.tmp / "batch-02", source="batch-02")
        self.directory = self.source("batch-02")

    def release(self, run, digest):
        record = {"schema": 1, "model": "m", "release": "r1", "classes": CLASSES,
                  "run": {"id": run, "weights_sha256": digest}}
        (self.repo / "models/m/release.json").write_text(json.dumps(record), encoding="utf-8")

    def test_only_unlabeled_images_get_unchecked_annotations_in_the_saved_format(self):
        code, out, err = self.train("prelabel", "m", "--source", "batch-02", "--run", self.run_id,
                                    "--conf", "0.3", "--device", "1")
        self.assertEqual(code, 0, err)
        self.assertIn(f"prelabeled 2 images of batch-02 with run {self.run_id}: 4 boxes", out)
        self.assertIn("skipped 1 images: already labeled", out)
        [(_, weights, images, options, conf)] = [c for c in self.backend.calls if c[0] == "predict"]
        self.assertEqual(weights, self.root / "runs/m" / self.run_id / "train" / ultralytics.BEST)
        self.assertEqual([p.name for p in images], ["n_0.jpg", "n_1.jpg"])
        self.assertEqual((options["device"], conf), ("1", 0.3))
        self.assertEqual(json.loads((self.directory / "n_2.json").read_text())["shapes"], [])  # 未被改动

        written = json.loads((self.directory / "n_0.json").read_text(encoding="utf-8"))
        sample = json.loads(LABELED.read_text(encoding="utf-8"))
        self.assertEqual(list(written), list(sample))                 # 字段与键的顺序同 4.0.6 保存的文件
        self.assertEqual(list(written["shapes"][0]), list(sample["shapes"][0]))
        self.assertEqual((written["checked"], written["imagePath"]), (False, "n_0.jpg"))
        self.assertEqual((self.directory / "n_0.json").stat().st_mode & 0o777, new_file_mode())
        found = xanylabeling.read(self.directory / "n_0.json")
        self.assertEqual([(s.label, s.shape_type) for s in found.shapes],
                         [("标志牌(圆牌)", "rectangle"), ("龙门架A", "rectangle")])
        self.assertEqual(found.shapes[1].points, [(0.0, 10.0), (400.0, 10.0), (400.0, 1080.0), (0.0, 1080.0)])

        log = json.loads((self.directory / paths.PRELABEL_RECORD).read_text(encoding="utf-8"))
        [batch] = log["batches"]
        self.assertEqual((batch["run"], batch["images"], batch["conf"]), (self.run_id, 2, 0.3))
        self.assertEqual(batch["weights_sha256"], paths.sha256(weights))
        self.assertEqual(sorted(log["images"]), ["n_0.jpg", "n_1.jpg"])

        code, out, _ = self.train("prelabel", "m", "--source", "batch-02", "--run", self.run_id)
        self.assertIn("prelabeled 0 images", out)                     # 永不覆盖已有标注
        self.assertIn("skipped 3 images: already labeled", out)

    def test_prelabels_enter_the_dataset_only_once_checked_and_corrections_are_counted(self):
        self.set_data(sources=["batch-01", "batch-02"])
        self.assertEqual(self.train("prelabel", "m", "--source", "batch-02", "--run", self.run_id)[0], 0)
        names = lambda: {s["image"] for s in dataset.prepare(self.root, self.spec()).manifest["samples"]
                         if s["source"] == "batch-02"}
        self.assertEqual(names(), set())
        # n_0：删掉龙门架A，补一个防撞桶，微调圆牌；n_1：原样确认。
        path = self.directory / "n_0.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        document["shapes"][0]["points"] = xanylabeling.rectangle_points([1346, 457, 1381, 492])
        document["shapes"][1] = xanylabeling.new_shape("防撞桶", "rectangle",
                                                       xanylabeling.rectangle_points([10, 10, 50, 50]))
        document["checked"] = True
        path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
        sources.accept(self.root, "batch-02", ["n_1.jpg"])
        self.assertEqual(names(), {"n_0.jpg", "n_1.jpg"})
        code, out, err = self.train("prepare", "m")
        self.assertEqual(code, 0, err)
        self.assertIn(f"prelabel of batch-02 by run {self.run_id}", out)
        self.assertIn("2 of 2 images checked; 3 of 4 predicted boxes kept (75%), 1 removed, 1 added", out)

    def test_the_default_model_is_the_published_one(self):
        code, _, err = self.train("prelabel", "m", "--source", "batch-02")
        self.assertIn("no published model yet", err)
        self.release("20990101-000000-abcdef", "0" * 64)
        code, _, err = self.train("prelabel", "m", "--source", "batch-02")
        self.assertIn("not in this training root", err)
        self.release(self.run_id, "0" * 64)
        code, _, err = self.train("prelabel", "m", "--source", "batch-02")
        self.assertIn("differ from the published model's sha256", err)
        self.assertFalse((self.directory / "n_0.json").exists())
        best = self.root / "runs/m" / self.run_id / "train" / ultralytics.BEST
        self.release(self.run_id, paths.sha256(best))
        code, out, err = self.train("prelabel", "m", "--source", "batch-02")
        self.assertEqual(code, 0, err)
        self.assertIn(f"with release r1 (run {self.run_id})", out)
        log = json.loads((self.directory / paths.PRELABEL_RECORD).read_text(encoding="utf-8"))
        self.assertEqual(log["batches"][0]["release"], "r1")

    def test_annotations_saved_during_inference_are_kept(self):
        def save(path):
            if path.name == "n_1.jpg":
                annotation(path.with_suffix(".json"), NEGATIVE, checked=True)
        self.backend.during_predict = save
        code, out, err = self.train("prelabel", "m", "--source", "batch-02", "--run", self.run_id)
        self.assertEqual(code, 0, err)
        self.assertIn("skipped 1 images: labeled meanwhile", out)
        self.assertTrue(xanylabeling.read(self.directory / "n_1.json").checked)
        log = json.loads((self.directory / paths.PRELABEL_RECORD).read_text(encoding="utf-8"))
        self.assertEqual(sorted(log["images"]), ["n_0.jpg"])

    def test_runs_that_cannot_prelabel(self):
        code, _, err = self.train("prelabel", "m", "--source", "batch-02", "--run", self.run_id, "--conf", "1.5")
        self.assertIn("--conf must be between 0 and 1", err)
        path = spec.path(self.repo, "m")
        path.write_text(path.read_text().replace("  - 防撞桶\n", "  - 防撞桶\n  - 新类别\n"))
        code, _, err = self.train("prelabel", "m", "--source", "batch-02", "--run", self.run_id)
        self.assertIn("trained with other classes", err)


class CorrectionsTest(unittest.TestCase):
    def test_boxes_are_matched_by_class_and_overlap(self):
        def shapes(*items):
            return [xanylabeling.new_shape(label, "rectangle", xanylabeling.rectangle_points(box))
                    for label, box in items]
        predicted = shapes(("a", [0, 0, 10, 10]), ("a", [20, 0, 30, 10]), ("b", [0, 20, 10, 30]))
        final = xanylabeling.Annotation(Path("x.json"), "x.jpg", 100, 100, True, {}, [
            xanylabeling.Shape("a", "rectangle", [(1, 1), (10, 10)]),          # 保留（微调）
            xanylabeling.Shape("c", "rectangle", [(0, 20), (10, 30)]),         # b 改成 c：删一个、加一个
            xanylabeling.Shape("a", "rectangle", [(60, 60), (70, 70)]),        # 补上的框
            xanylabeling.Shape("a", "polygon", [(0, 0), (5, 5), (0, 5)])])     # 其他形状不计
        self.assertEqual(dict(detect.corrections(predicted, final)), {"kept": 1, "removed": 2, "added": 2})


NVINFER = """[property]
onnx-file=artifacts/model.onnx
labelfile-path=artifacts/labels.txt
model-engine-file=/data/engines/m/model_b1_gpu0_fp16.engine  # 构建器写入的位置
batch-size=1
num-detected-classes=3
network-type=0
"""


class FakeOnnx:
    """onnxio 的替身：签名可改，推理 = 原生推理 + offset。"""
    def __init__(self, backend):
        self.backend = backend
        self.output = [1, 4, 6]
        self.offset = 0.0

    def signature(self, path):
        return {"inputs": [{"name": "input", "shape": [1, 3, 640, 640]}],
                "outputs": [{"name": "output", "shape": self.output}]}

    def run(self, path, batch):
        return self.backend.reference("detect", None, "xyxy-confidence-class_id", batch) + self.offset


@needs_numpy
class PublishTest(RunTestCase):
    def setUp(self):
        super().setUp()
        self.assertEqual(self.train("run", "m")[0], 0)
        self.run_id = self.only_run()
        self.model_dir = self.repo / "models/m"
        (self.model_dir / "nvinfer.txt").write_text(NVINFER, encoding="utf-8")
        self.onnx = FakeOnnx(self.backend)
        patcher = mock.patch.object(publish, "onnxio", self.onnx)
        patcher.start()
        self.addCleanup(patcher.stop)

    def snapshot(self):
        return {str(p.relative_to(self.model_dir)): p.read_bytes() for p in self.model_dir.rglob("*") if p.is_file()}

    def gate(self, text):
        path = spec.path(self.repo, "m")
        path.write_text(path.read_text().replace("    min: {}", f"    min: {text}"))

    def second_run(self):
        path = spec.path(self.repo, "m")
        path.write_text(path.read_text().replace("epochs: 200", "epochs: 201"))  # 另一份规格摘要
        self.assertEqual(self.train("run", "m")[0], 0)
        return max(p.name for p in (self.root / "runs/m").iterdir() if p.name != self.run_id)

    def test_first_publish_writes_a_checked_model_directory(self):
        code, _, err = self.train("publish", "m", self.run_id)
        self.assertIn("evaluate.gate.min is empty and there is no published model", err)
        self.assertFalse((self.model_dir / "release.json").exists())
        code, out, err = self.train("publish", "m", self.run_id, "--accept-metrics")
        self.assertEqual(code, 0, err)
        release = json.loads((self.model_dir / "release.json").read_text(encoding="utf-8"))
        self.assertRegex(release["release"], r"^\d{8}-\d{6}-[0-9a-f]{6}$")
        self.assertEqual((release["run"]["id"], release["evaluated_on"], release["metrics"]["map50_95"]),
                         (self.run_id, self.dataset_id, 0.5))
        self.assertTrue(release["gate"]["accepted_without_comparison"])
        self.assertTrue(release["consistency"]["passed"])
        self.assertEqual(release["consistency"]["class_mismatches"], 0)
        self.assertEqual((self.model_dir / "artifacts/labels.txt").read_text(encoding="utf-8"),
                         "".join(f"{c}\n" for c in CLASSES))
        best = self.root / "runs/m" / self.run_id / "train" / ultralytics.BEST
        self.assertEqual((self.model_dir / "artifacts/model.onnx").read_bytes(), b"onnx of " + best.read_bytes())
        engine = f"/data/engines/m/{release['release']}/model_b1_gpu0_fp16.engine"
        self.assertIn(f"model-engine-file={engine}  # 构建器写入的位置\n",
                      (self.model_dir / "nvinfer.txt").read_text(encoding="utf-8"))
        self.assertEqual(models.check_model(self.model_dir), [])     # 与打包时同一套校验
        [export] = [c for c in self.backend.calls if c[0] == "export"]
        self.assertEqual((export[2], export[3], export[4]["batch"]), ("xyxy-confidence-class_id", (640, 640), 1))
        record = json.loads((self.root / "runs/m" / self.run_id / "export/export.json").read_text())
        self.assertEqual(record["onnx"]["outputs"][0]["shape"], [1, 4, 6])
        self.assertIn(f"nvinfer.txt: model-engine-file /data/engines/m/model_b1_gpu0_fp16.engine -> {engine}", out)
        before = self.snapshot()
        code, _, err = self.train("publish", "m", self.run_id, "--accept-metrics")
        self.assertIn(f"identical to the published release {release['release']}", err)   # 不为同一模型换 engine 路径
        self.assertEqual(self.snapshot(), before)

    def test_the_minimum_gate_is_never_overridden(self):
        self.gate("{map50_95: 0.6}")
        before = self.snapshot()
        code, _, err = self.train("publish", "m", self.run_id, "--accept-metrics")
        self.assertIn("map50_95 0.5000 does not reach evaluate.gate.min 0.6", err)
        self.assertEqual(self.snapshot(), before)
        self.assertFalse(any(c[0] == "export" for c in self.backend.calls))  # 门槛在导出之前
        path = spec.path(self.repo, "m")
        path.write_text(path.read_text().replace("min: {map50_95: 0.6}", "min: {map50_95: 0.5}"))
        code, out, err = self.train("publish", "m", self.run_id)              # 有下限即无需 --accept-metrics
        self.assertEqual(code, 0, err)
        self.assertIn("gate min: map50_95 0.5000 >= 0.5", out)

    def test_a_new_run_is_compared_with_the_published_one(self):
        self.assertEqual(self.train("publish", "m", self.run_id, "--accept-metrics")[0], 0)
        first = json.loads((self.model_dir / "release.json").read_text())["release"]
        second = self.second_run()
        self.backend.map50_95[second] = 0.48                                   # 下降 0.02 > max_drop 0.01
        shutil.rmtree(self.root / "runs/m" / second / "eval")                  # train run 时按 0.5 评估过
        before = self.snapshot()
        code, _, err = self.train("publish", "m", second, "--device", "1")
        self.assertIn(f"map50_95 0.4800 is 0.0200 below the published release {first}", err)
        self.assertEqual(self.snapshot(), before)
        self.backend.map50_95[second] = 0.495
        shutil.rmtree(self.root / "runs/m" / second / "eval")                  # 重新评估（已有结果会复用）
        code, out, err = self.train("publish", "m", second)
        self.assertEqual(code, 0, err)
        release = json.loads((self.model_dir / "release.json").read_text())
        regression = release["gate"]["regression"]
        self.assertEqual((regression["release"], regression["run"], regression["passed"]), (first, self.run_id, True))
        self.assertAlmostEqual(regression["drop"], 0.005)
        self.assertFalse(release["gate"]["accepted_without_comparison"])
        # 发布段被替换为新的发布，而不是再嵌套一层。
        self.assertIn(f"model-engine-file=/data/engines/m/{release['release']}/model_b1_gpu0_fp16.engine",
                      (self.model_dir / "nvinfer.txt").read_text())
        self.assertEqual(regression["published"], 0.5)                       # 当前发布在同一测试划分上的结果
        self.assertIn(("evaluate", self.root / "runs/m" / second / "train" / ultralytics.BEST,
                       self.root / "datasets/m" / self.dataset_id / "view/fake/data.yaml", "test", "1"),
                      self.backend.calls)                                      # --device 只用于门槛的评估

    def test_a_published_model_that_cannot_be_reevaluated_needs_explicit_acceptance(self):
        self.assertEqual(self.train("publish", "m", self.run_id, "--accept-metrics")[0], 0)
        second = self.second_run()
        shutil.rmtree(self.root / "runs/m" / self.run_id)                      # 当前发布的权重不在这台训练机上
        self.gate("{map50: 0.5}")
        code, _, err = self.train("publish", "m", second)
        self.assertIn(f"its run '{self.run_id}' is not in this training root", err)
        self.assertIn("--accept-metrics", err)
        code, _, err = self.train("publish", "m", second, "--accept-metrics")
        self.assertEqual(code, 0, err)
        gate = json.loads((self.model_dir / "release.json").read_text())["gate"]
        self.assertTrue(gate["accepted_without_comparison"])
        self.assertEqual(gate["min"]["map50"]["passed"], True)

    def test_an_export_that_differs_from_native_inference_is_not_published(self):
        before = self.snapshot()
        self.onnx.offset = 0.01                                                # 置信度差 0.01 > 0.001
        code, _, err = self.train("publish", "m", self.run_id, "--accept-metrics")
        self.assertIn("does not reproduce the backend's own inference", err)
        self.assertEqual(self.snapshot(), before)
        record = json.loads((self.root / "runs/m" / self.run_id / "export/export.json").read_text())
        self.assertFalse(record["consistency"]["passed"])
        self.onnx.offset = 0.0
        self.onnx.output = [1, 4, 5]
        code, _, err = self.train("publish", "m", self.run_id, "--accept-metrics")
        self.assertIn("xyxy-confidence-class_id needs one output whose last dimension is 6", err)
        self.assertEqual(self.snapshot(), before)

    def test_contract_problems_are_found_before_anything_changes(self):
        nvinfer = self.model_dir / "nvinfer.txt"
        nvinfer.write_text(NVINFER.replace("/data/engines/m/model", "/data/engines/m/fp16/model"))
        code, _, err = self.train("publish", "m", self.run_id, "--accept-metrics")
        self.assertIn("cannot tell which directory of model-engine-file", err)
        self.assertFalse((self.root / "runs/m" / self.run_id / "export").exists())  # 预检在评估与导出之前
        nvinfer.write_text(NVINFER.replace("num-detected-classes=3", "num-detected-classes=4"))
        before = self.snapshot()
        code, _, err = self.train("publish", "m", self.run_id, "--accept-metrics")
        self.assertIn("fail the model check, nothing was changed", err)
        self.assertIn("num-detected-classes=4", err)
        self.assertEqual(self.snapshot(), before)


class PublishListTest(RunTestCase):
    def test_without_a_run_the_candidates_are_listed_and_nothing_is_published(self):
        code, out, _ = self.train("publish", "m")
        self.assertEqual(code, 0)
        self.assertIn("no completed runs of m", out)
        self.assertEqual(self.train("run", "m")[0], 0)
        full = self.only_run()
        self.backend.map50_95[full] = 0.61
        shutil.rmtree(self.root / "runs/m" / full / "eval")
        self.assertEqual(self.train("eval", "m", full)[0], 0)
        self.assertEqual(self.train("run", "m", "--option", "epochs=3")[0], 0)
        trial = next(p.name for p in (self.root / "runs/m").iterdir() if p.name != full)
        self.backend.fail = KeyboardInterrupt()
        self.train("run", "m", "--option", "epochs=4")
        stopped = next(p.name for p in (self.root / "runs/m").iterdir() if p.name not in (full, trial))
        (self.repo / "models/m/release.json").write_text(json.dumps({"release": "r1", "run": {"id": full}}))

        code, out, err = self.train("publish", "m")
        self.assertEqual(code, 0, err)
        lines = out.splitlines()
        self.assertIn(f"completed runs of m (the test split of dataset {self.dataset_id}); nothing was published:",
                      lines[0])
        [trial_line] = [line for line in lines if trial in line]
        [full_line] = [line for line in lines if full in line]
        self.assertRegex(trial_line, r"^   +\S+ +0\.5000 +0\.7000 .* epochs=3$")
        self.assertRegex(full_line, r"^  \* \S+ +0\.6100 .* -$")              # 当前发布的 run 带标记
        self.assertLess(lines.index(trial_line), lines.index(full_line))       # 最近完成的在前
        self.assertIn("* the run of the published release r1", out)
        self.assertIn(f"not publishable: {stopped} (interrupted)", out)
        self.assertIn("publish one with: train publish m <run>", out)
        self.assertFalse((self.root / "runs/m" / full / "export").exists())
        self.assertIn("name the run to publish", self.train("publish", "m", "--accept-metrics")[2])

        image(self.tmp / "more/x/x.jpg", "white")
        annotation(self.tmp / "more/x/x.json", LABELED, checked=True)
        self.import_(self.tmp / "more")
        newer = dataset.prepare(self.root, self.spec()).dataset_id            # 新数据集上还没有评估
        out = self.train("publish", "m")[1]
        self.assertIn(f"dataset {newer}", out)
        self.assertIn("not evaluated on this dataset yet: train eval m <run>", out)


@needs_numpy
class CompareTest(unittest.TestCase):
    def test_rows_are_compared_where_they_matter(self):
        native = numpy.array([[[10, 10, 20, 20, 0.9, 1], [0, 0, 5, 5, 0.1, 2], [0, 0, 1, 1, 0.01, 0]]],
                             dtype=numpy.float32)
        same = detect.compare("xyxy-confidence-class_id", native, native.copy())
        self.assertEqual((same["passed"], same["candidates"], same["confident"]), (True, 2, 1))
        exported = native.copy()
        exported[0, 2, :4] += 50                                              # 低于 compare_conf 的候选不比较
        exported[0, 1, 5] = 0                                                 # 低于 class_conf 的类别不比较
        self.assertTrue(detect.compare("xyxy-confidence-class_id", native, exported)["passed"])
        exported[0, 0, 5] = 2
        self.assertEqual(detect.compare("xyxy-confidence-class_id", native, exported)["class_mismatches"], 1)
        exported = native.copy()
        exported[0, 0, 0] += 0.6
        self.assertFalse(detect.compare("xyxy-confidence-class_id", native, exported)["passed"])
        exported = native.copy()
        exported[0, 1, 0] = numpy.nan
        self.assertFalse(detect.compare("xyxy-confidence-class_id", native, exported)["passed"])
        self.assertIn("shapes differ", detect.compare("xyxy-confidence-class_id", native, native[:, :2])["reason"])

    def test_letterbox_keeps_the_aspect_ratio(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = image(Path(temporary) / "a.jpg", "white", size=(200, 100))
            array = media.letterbox(path, 64, 64)
        self.assertEqual((array.shape, array.dtype), ((3, 64, 64), numpy.float32))
        self.assertAlmostEqual(float(array[0, 32, 32]), 1.0, places=2)       # 图像在中间
        self.assertAlmostEqual(float(array[0, 2, 32]), 114 / 255, places=2)   # 上下填充


class ExportSpecTest(TrainingTestCase):
    def test_export_settings_are_checked(self):
        path = spec.path(self.repo, "m")
        path.write_text(path.read_text().replace("  batch: 1\n", "  batch: 0\n  half: true\n")
                        .replace("dynamic: false", "dynamic: no-thanks"))
        with self.assertRaisesRegex(TrainError, "export.batch must be an integer >= 1(.|\n)*export.dynamic must be "
                                                "true or false(.|\n)*unknown keys half"):
            self.spec()


class RepositoryModelTest(unittest.TestCase):
    def test_repository_models_are_valid_training_specs(self):
        # 构建镜像的上下文不含 models/：这里在开发机上覆盖仓库中的真实规格。
        for directory in models.model_dirs(ROOT):
            spec.load(ROOT, directory.name)


if __name__ == "__main__":
    unittest.main()
