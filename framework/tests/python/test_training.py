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
from dsf_deploy import classes as class_diff, models
from dsf_train import (attributes, backends, cli, dataset, lock, media, paths, publish, runs, sources, spec, split, update,
                       xanylabeling)
from dsf_train.backends import ultralytics
from dsf_train.tasks import classify, detect
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
        # 训练机上设置的 TRAIN_DEVICE 不影响用例；需要它的用例自己设置。
        environment = mock.patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        os.environ.pop(runs.DEVICE_ENV, None)
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
        path.write_text(path.read_text() + "cascade: {parent: x, classes: [a]}\n")
        with self.assertRaisesRegex(TrainError, "cascade is only for second-stage classify models"):
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
        return dataset.prepare(self.repo, self.root, self.spec())

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

    def test_images_the_backend_rewrites_are_copied_into_the_view(self):
        # 结尾不是 FF D9 的 JPEG 能解码，但 Ultralytics 会原地“修复”它：经硬链接会改动来源。
        with (self.raw / "a/f_1.jpg").open("ab") as stream:
            stream.write(b"\x00\x00")
        self.import_(self.raw)
        result = self.prepare()
        sample = next(s for s in result.manifest["samples"] if s["image"] == "f_1.jpg")
        copied = result.directory / paths.VIEW / "ultralytics/images" / sample["split"] / "batch-01__f_1.jpg"
        self.assertNotEqual(copied.stat().st_ino, (self.source() / "f_1.jpg").stat().st_ino)
        self.assertEqual(copied.read_bytes(), (self.source() / "f_1.jpg").read_bytes())
        copied.write_bytes(b"rewritten")                                          # 重写视图不影响来源
        self.assertEqual(paths.sha256(self.source() / "f_1.jpg"), sample["sha256"])

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
        self.assertEqual(result.ignored_shapes, {"polygon shapes (not used by task detect)": 1})  # 忽略的类别与其他任务的形状不是错误
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


class ClassDiffTest(unittest.TestCase):
    def test_differences_by_name_and_position(self):
        self.assertEqual(class_diff.diff(["a", "b"], ["a", "b"]), [])
        self.assertEqual(class_diff.diff(["a", "b"], ["a", "b", "c"]), ["added c (class 2)"])
        self.assertEqual(class_diff.diff(["a", "b", "c"], ["a", "x", "c"]), ["class 1: b -> x"])
        self.assertEqual(class_diff.diff(["a", "b", "c"], ["b", "c"]),
                         ["removed a (was class 0)", "renumbered b 1->0, c 2->1"])


class ClassDependencyTest(TrainingTestCase):
    """来源 → 模型的类别依赖：sources.lock.json（docs/training.md §15.1）。"""
    def setUp(self):
        super().setUp()
        image(self.raw / "a/f_0.jpg", "red")
        image(self.raw / "a/f_1.jpg", "blue")
        annotation(self.raw / "a/f_0.json", LABELED, checked=True)
        annotation(self.raw / "a/f_1.json", NEGATIVE, checked=True)
        self.import_(self.raw)

    def prepare(self):
        return dataset.prepare(self.repo, self.root, self.spec())

    def set_source_classes(self, names, source="batch-01"):
        record = sources.read_record(self.source(source))
        record["classes"] = names
        paths.write_json(self.source(source) / paths.SOURCE_RECORD, record)

    def recorded(self):
        return json.loads(lock.path(self.repo, "m").read_text(encoding="utf-8"))

    def test_prepare_records_the_sources_and_refuses_changes_until_confirmed(self):
        self.assertEqual(sources.read_record(self.source())["classes"], CLASSES)
        result = self.prepare()
        self.assertIn("  note: recorded the classes of source batch-01 in "
                      f"{lock.path(self.repo, 'm')} (commit it with the model)", dataset.report(result, self.spec()))
        self.assertEqual(self.recorded(), {"schema": 1, "sources": {"batch-01": CLASSES}})
        self.assertEqual(self.prepare().notes, [])  # 记录不变时不再写入
        self.set_source_classes(CLASSES + ["新类别"])  # 另一个模型给共用的来源新增了类别
        with self.assertRaisesRegex(TrainError, "changed since m confirmed them(.|\n)*source batch-01:\n"
                                                "    added 新类别 \\(class 3\\)(.|\n)*train update m --confirm"):
            self.prepare()
        code, _, err = self.train("run", "m")
        self.assertEqual(code, 1)
        self.assertIn("added 新类别 (class 3)", err)
        code, out, _ = self.train("update", "m")
        self.assertEqual(code, 0)
        self.assertIn("  source batch-01: changed since confirmed:\n    added 新类别 (class 3)", out)
        code, out, _ = self.train("update", "m", "--confirm")
        self.assertEqual(code, 0)
        self.assertIn("  source batch-01:\n    added 新类别 (class 3)", out)
        self.assertEqual(self.recorded()["sources"]["batch-01"], CLASSES + ["新类别"])
        self.prepare()
        self.assertIn("  source batch-01: confirmed", self.train("update", "m")[1])

    def test_classes_of_the_source_the_model_does_not_use_are_ignored(self):
        self.set_source_classes(CLASSES + ["护栏"])
        annotation(self.source() / "f_0.json", LABELED, checked=True,
                   shapes=[{"label": "护栏", "points": [[1, 1], [9, 9]], "shape_type": "rectangle"}])
        sample = next(s for s in self.prepare().manifest["samples"] if s["image"] == "f_0.jpg")
        self.assertEqual(sample["objects"], [])
        annotation(self.source() / "f_0.json", LABELED, checked=True,
                   shapes=[{"label": "护拦", "points": [[1, 1], [9, 9]], "shape_type": "rectangle"}])
        with self.assertRaisesRegex(TrainError, "batch-01/f_0.jpg: unknown class '护拦'"):
            self.prepare()  # 两边都没有的标签仍是错误：拼写错误不会悄悄变成新类别

    def test_every_class_of_the_model_must_be_in_every_source(self):
        path = spec.path(self.repo, "m")
        path.write_text(path.read_text().replace("  - 防撞桶\n", "  - 防撞桶\n  - 新类别\n"))  # 手改规格
        with self.assertRaisesRegex(TrainError, "source batch-01 has no class 新类别(.|\n)*train update m add"):
            self.prepare()
        self.assertFalse(lock.path(self.repo, "m").exists())
        code, _, err = self.train("update", "m", "--confirm")
        self.assertEqual(code, 1)
        self.assertIn("source batch-01 has no class 新类别", err)
        with self.assertRaisesRegex(TrainError, "source batch-01 has no class 新类别"):
            sources.Importer(self.root, "batch-01", CLASSES + ["新类别"])  # 导入也不能绕过

    def test_sources_without_recorded_classes_use_labels_txt(self):
        record = sources.read_record(self.source())
        del record["classes"]  # 实现类别依赖之前导入的来源
        (self.source() / paths.SOURCE_LABELS).write_text("\n".join(CLASSES + ["护栏"]) + "\n", encoding="utf-8")
        paths.write_json(self.source() / paths.SOURCE_RECORD, record)
        self.assertEqual(sources.class_list(self.source(), record), CLASSES + ["护栏"])
        self.prepare()
        self.assertEqual(self.recorded()["sources"]["batch-01"], CLASSES + ["护栏"])
        image(self.tmp / "more/x.jpg", "white")
        sources.Importer(self.root, "batch-01", CLASSES[:2]).run([self.tmp / "more"])  # 另一个模型继续导入
        self.assertEqual(sources.read_record(self.source())["classes"], CLASSES + ["护栏"])
        self.assertEqual((self.source() / paths.SOURCE_LABELS).read_text(encoding="utf-8").splitlines(),
                         CLASSES + ["护栏"])

    def test_sources_added_to_or_removed_from_the_model(self):
        self.prepare()
        image(self.tmp / "more/x.jpg", "white")
        annotation(self.tmp / "more/x.json", NEGATIVE, checked=True)
        self.import_(self.tmp / "more", source="batch-02")
        self.set_data(sources=["batch-01", "batch-02"])
        code, _, err = self.train("run", "m")
        self.assertIn("does not record the classes of batch-02: run train prepare m first", err)
        self.assertEqual(self.prepare().notes, [f"recorded the classes of source batch-02 in "
                                                f"{lock.path(self.repo, 'm')} (commit it with the model)"])
        self.set_data(sources=["batch-02"])
        self.assertEqual(self.prepare().notes,
                         [f"dropped source batch-01 from {lock.path(self.repo, 'm')}: it is no longer in data.sources"])
        self.assertEqual(sorted(self.recorded()["sources"]), ["batch-02"])


class WriteClassesTest(TrainingTestCase):
    def test_classes_are_rewritten_in_place(self):
        path = spec.path(self.repo, "m")
        path.write_text(path.read_text().replace("  - 龙门架A\n", "  - 龙门架A  # 门架\n"))
        spec.write_classes(self.repo, "m", ["标志牌(圆牌)", "门架", "防撞桶"])
        text = path.read_text()
        self.assertIn("  - 门架  # 门架\n", text)
        self.assertIn("# 顺序即类别号", text)  # 注释与其他字段不变
        spec.write_classes(self.repo, "m", ["标志牌(圆牌)", "防撞桶"])
        self.assertEqual(self.spec()["classes"], ["标志牌(圆牌)", "防撞桶"])
        self.assertNotIn("门架", path.read_text())  # 删除的项连同行尾注释删去
        spec.write_classes(self.repo, "m", ["标志牌(圆牌)", "防撞桶", "a: b"])
        self.assertEqual(self.spec()["classes"], ["标志牌(圆牌)", "防撞桶", "a: b"])
        text = path.read_text()
        start = text.index("classes:\n")
        end = text.index("\n\n", start)
        path.write_text(text[:start] + "classes: [x, y]" + text[end:])  # 流式列表
        spec.write_classes(self.repo, "m", ["x"])
        self.assertEqual(self.spec()["classes"], ["x"])


class UpdateTest(TrainingTestCase):
    """train update rename / retire（docs/training.md §15.3）。m 与 m2 共用 batch-01，m2 只用其中两类。"""
    def setUp(self):
        super().setUp()
        image(self.raw / "a/f_0.jpg", "red")
        image(self.raw / "a/f_1.jpg", "blue")
        annotation(self.raw / "a/f_0.json", LABELED, checked=True)  # 一个 标志牌(圆牌) 框
        annotation(self.raw / "a/f_1.json", LABELED, checked=False,
                   shapes=[{"label": "龙门架A", "points": [[1, 1], [9, 9]], "shape_type": "rectangle"}])
        image(self.raw / "a/f_2.jpg", "green")
        annotation(self.raw / "a/f_2.json", NEGATIVE, checked=True)
        self.import_(self.raw)
        (self.repo / "models/m2").mkdir()
        text = spec.skeleton("m2", "detect", ["龙门架A", "防撞桶"]).replace("  sources: []", "  sources: [batch-01]")
        spec.path(self.repo, "m2").write_text(text, encoding="utf-8")
        for model_id in ("m", "m2"):
            dataset.prepare(self.repo, self.root, spec.load(self.repo, model_id))

    def labels(self, name):
        return [s.label for s in xanylabeling.read(self.source() / name).shapes]

    def source_classes(self):
        return sources.read_record(self.source())["classes"]

    def locked(self, model_id):
        return lock.read(self.repo, model_id)["batch-01"]

    def test_rename_rewrites_the_sources_and_follows_the_models_sharing_them(self):
        code, out, err = self.train("update", "m", "rename", "标志牌(圆牌)", "圆牌")
        self.assertEqual(code, 0, err)
        self.assertIn("  source batch-01: 1 annotations rewritten", out)
        self.assertEqual(self.labels("f_0.json"), ["圆牌"])
        self.assertTrue(xanylabeling.read(self.source() / "f_0.json").checked)  # checked 不变
        renamed = ["圆牌", "龙门架A", "防撞桶"]
        self.assertEqual(self.source_classes(), renamed)
        self.assertEqual((self.source() / paths.SOURCE_LABELS).read_text(encoding="utf-8").splitlines(), renamed)
        self.assertEqual(self.spec()["classes"], renamed)
        self.assertEqual(spec.load(self.repo, "m2")["classes"], ["龙门架A", "防撞桶"])  # 不用它的模型不变
        self.assertEqual((self.locked("m"), self.locked("m2")), (renamed, renamed))  # 两边的记录都自动更新
        for model_id in ("m", "m2"):
            dataset.prepare(self.repo, self.root, spec.load(self.repo, model_id))
        # 共用的类别：m2 一起改名
        code, out, err = self.train("update", "m", "rename", "龙门架A", "门架")
        self.assertEqual(code, 0, err)
        self.assertEqual(spec.load(self.repo, "m2")["classes"], ["门架", "防撞桶"])
        self.assertIn("  model m2: class 0 renamed", out)
        self.assertEqual(self.labels("f_1.json"), ["门架"])
        dataset.prepare(self.repo, self.root, spec.load(self.repo, "m2"))

    def test_rename_conflicts_are_found_before_anything_changes(self):
        code, _, err = self.train("update", "m", "rename", "龙门架A", "防撞桶")
        self.assertIn("m already has a class 防撞桶", err)
        record = sources.read_record(self.source())
        record["classes"] = CLASSES + ["圆牌"]
        paths.write_json(self.source() / paths.SOURCE_RECORD, record)
        code, _, err = self.train("update", "m", "rename", "标志牌(圆牌)", "圆牌")
        self.assertIn("not confirmed for m", err)  # 起点必须是确认过的状态
        self.train("update", "m", "--confirm")
        code, _, err = self.train("update", "m", "rename", "标志牌(圆牌)", "圆牌")
        self.assertIn("source batch-01 already has a class 圆牌", err)
        self.assertEqual(self.labels("f_0.json"), ["标志牌(圆牌)"])

    def test_retire_removes_the_class_from_the_model_only(self):
        code, out, err = self.train("update", "m", "retire", "标志牌(圆牌)")
        self.assertEqual(code, 0, err)
        self.assertIn("龙门架A is now class 0", out)
        self.assertEqual(self.spec()["classes"], ["龙门架A", "防撞桶"])
        self.assertEqual(self.source_classes(), CLASSES)  # 来源不变
        self.assertEqual(self.labels("f_0.json"), ["标志牌(圆牌)"])
        sample = next(s for s in dataset.prepare(self.repo, self.root, self.spec()).manifest["samples"]
                      if s["image"] == "f_0.jpg")
        self.assertEqual(sample["objects"], [])  # 读取时忽略

    def test_retire_into_merges_in_the_sources(self):
        code, _, err = self.train("update", "m", "retire", "龙门架A", "--into", "防撞桶")
        self.assertEqual(code, 1)
        self.assertIn("model m2 has 龙门架A as a class of its own", err)
        code, out, err = self.train("update", "m", "retire", "标志牌(圆牌)", "--into", "防撞桶")
        self.assertEqual(code, 0, err)
        self.assertEqual(self.labels("f_0.json"), ["防撞桶"])
        self.assertEqual(self.source_classes(), ["龙门架A", "防撞桶"])
        self.assertEqual(self.spec()["classes"], ["龙门架A", "防撞桶"])
        self.assertEqual(self.locked("m"), ["龙门架A", "防撞桶"])
        self.assertIn("m2 must review the changed sources: train update m2 --confirm", out)
        dataset.prepare(self.repo, self.root, self.spec())
        with self.assertRaisesRegex(TrainError, "removed 标志牌\\(圆牌\\) \\(was class 0\\)"):
            dataset.prepare(self.repo, self.root, spec.load(self.repo, "m2"))

    def checked(self, name):
        return xanylabeling.read(self.source() / f"{name}.json").checked

    def reviews(self):
        return sources.read_record(self.source())["reviews"]

    def test_add_unchecks_the_checked_images_for_review(self):
        code, out, err = self.train("update", "m", "add", "防撞墩")
        self.assertEqual(code, 0, err)
        self.assertIn("added 防撞墩 to m as class 3", out)
        self.assertIn("  source batch-01: class added, 2 checked images unchecked for review", out)
        self.assertIn("m2 must review the changed sources: train update m2 --confirm", out)
        self.assertEqual((self.checked("f_0"), self.checked("f_2")), (False, False))
        self.assertEqual(self.labels("f_0.json"), ["标志牌(圆牌)"])  # 只改 checked
        [review] = self.reviews()
        self.assertEqual((review["class"], review["from"], review["files"], review["revoked"]),
                         ("防撞墩", None, ["f_0.jpg", "f_2.jpg"], True))
        self.assertEqual(self.source_classes(), CLASSES + ["防撞墩"])
        self.assertEqual(self.spec()["classes"], CLASSES + ["防撞墩"])
        self.assertEqual(self.locked("m"), CLASSES + ["防撞墩"])
        with self.assertRaisesRegex(TrainError, "no checked images"):
            dataset.prepare(self.repo, self.root, self.spec())
        xanylabeling.update(self.source() / "f_0.json", checked=True)  # 人补框后重新勾选
        result = dataset.prepare(self.repo, self.root, self.spec())
        self.assertEqual(result.skipped[("batch-01", "waiting for review of 防撞墩")], 1)
        self.assertEqual(result.skipped[("batch-01", "not checked")], 1)
        self.assertIn("    review of 防撞墩 since ", self.train("update", "m")[1])
        self.assertIn(": 1 of 2 checked again", self.train("update", "m")[1])
        # 名单中的图片不能批量确认，除非给出理由
        code, _, err = self.train("accept", "batch-01", "f_2")
        self.assertIn("1 of these images wait for review of 防撞墩 (f_2.jpg)", err)
        code, _, err = self.train("accept", "batch-01", "--all", "--reason", "夜间画面，没有防撞墩")
        self.assertEqual(code, 0, err)
        [accepted] = self.reviews()[0]["accepted"]
        self.assertEqual((accepted["reason"], accepted["images"]), ("夜间画面，没有防撞墩", ["f_2.jpg"]))

    def test_add_from_a_class_unchecks_only_the_images_with_it(self):
        code, out, err = self.train("update", "m", "add", "标志牌(方牌)", "--from", "标志牌(圆牌)")
        self.assertEqual(code, 0, err)
        self.assertIn("1 checked images with 标志牌(圆牌) boxes unchecked", out)
        self.assertEqual((self.checked("f_0"), self.checked("f_2")), (False, True))
        self.assertEqual(self.reviews()[0]["from"], "标志牌(圆牌)")

    def test_add_with_a_skipped_source_unchecks_nothing(self):
        code, _, err = self.train("update", "m", "add", "防撞墩", "--skip-source", "batch-01")
        self.assertIn("--skip-source and --reason go together", err)
        code, out, err = self.train("update", "m", "add", "防撞墩", "--skip-source", "batch-01",
                                    "--reason", "隧道内素材")
        self.assertEqual(code, 0, err)
        self.assertIn("class added, nothing unchecked (隧道内素材)", out)
        self.assertEqual((self.checked("f_0"), self.checked("f_2")), (True, True))
        self.assertEqual(self.reviews()[0]["skipped"], "隧道内素材")
        self.assertEqual(self.source_classes(), CLASSES + ["防撞墩"])

    def test_a_source_that_has_the_class_already_is_not_reviewed_again(self):
        self.assertEqual(self.train("update", "m2", "add", "防撞墩")[0], 0)
        xanylabeling.update(self.source() / "f_0.json", checked=True)
        # 来源的变化正是 m 要做的这次修改，不必先 --confirm
        code, out, err = self.train("update", "m", "add", "防撞墩")
        self.assertEqual(code, 0, err)
        self.assertIn("  source batch-01: already has 防撞墩, unchanged", out)
        self.assertTrue(self.checked("f_0"))
        self.assertEqual(len(self.reviews()), 1)

    def test_an_interrupted_add_continues_with_its_list(self):
        original = xanylabeling.update

        def once(path, **fields):
            original(path, **fields)
            raise KeyboardInterrupt
        with mock.patch.object(xanylabeling, "update", once):
            with self.assertRaises(KeyboardInterrupt):
                update.add(self.repo, self.root, self.spec(), "防撞墩")
        self.assertEqual((self.checked("f_0"), self.checked("f_2")), (False, True))
        self.assertFalse(self.reviews()[0]["revoked"])
        with self.assertRaisesRegex(TrainError, "changed since m2 confirmed"):
            dataset.prepare(self.repo, self.root, spec.load(self.repo, "m2"))  # 其他模型不会用撤销了一半的来源
        update.add(self.repo, self.root, self.spec(), "防撞墩")
        self.assertEqual((self.checked("f_0"), self.checked("f_2")), (False, False))
        [review] = self.reviews()
        self.assertEqual((review["files"], review["revoked"]), (["f_0.jpg", "f_2.jpg"], True))
        self.assertEqual(self.spec()["classes"], CLASSES + ["防撞墩"])

    def test_an_interrupted_update_completes_when_run_again(self):
        original = spec.write_classes

        def interrupted(*args):
            raise KeyboardInterrupt
        with mock.patch.object(spec, "write_classes", interrupted):
            with self.assertRaises(KeyboardInterrupt):
                update.rename(self.repo, self.root, self.spec(), "龙门架A", "门架")
        self.assertEqual(self.source_classes(), ["标志牌(圆牌)", "门架", "防撞桶"])  # 来源已改，规格未改
        with self.assertRaisesRegex(TrainError, "unknown class '门架'|has no class 龙门架A"):
            dataset.prepare(self.repo, self.root, self.spec())
        self.assertIs(spec.write_classes, original)
        update.rename(self.repo, self.root, self.spec(), "龙门架A", "门架")
        self.assertEqual(self.spec()["classes"], ["标志牌(圆牌)", "门架", "防撞桶"])
        self.assertEqual(spec.load(self.repo, "m2")["classes"], ["门架", "防撞桶"])
        for model_id in ("m", "m2"):
            dataset.prepare(self.repo, self.root, spec.load(self.repo, model_id))


class FakeBackend(types.ModuleType):
    """记录调用的假后端；fail 设为异常时训练在写出 last 之后抛出它。"""
    NAME = "fake"
    TASKS = ("detect", "classify")
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
    data_path = staticmethod(ultralytics.data_path)
    crops = staticmethod(ultralytics.crops)
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
        if task == "classify":
            return classify.metrics([[8, 2], [1, 3]])
        return {"metrics": {"map50_95": value, "map50": 0.7, "precision": 0.8, "recall": 0.6},
                "classes": {0: {"map50_95": value, "map50": 0.7, "precision": 0.8, "recall": 0.6}}}

    def export(self, task, weights, layout, size, settings, target, workdir):
        self.calls.append(("export", weights, layout, size, dict(settings)))
        target.write_bytes(b"onnx of " + weights.read_bytes())

    def reference(self, task, weights, layout, batch):
        """检测：每张图 4 个候选，坐标来自图像均值，两个高分、两个低分。分类：概率来自图像均值。"""
        if task == "classify":
            means = numpy.array([float(item.mean()) for item in batch])
            return numpy.stack([0.25 + 0.5 * means, 0.75 - 0.5 * means], axis=1).astype(numpy.float32)
        rows = []
        for item in batch:
            mean = float(item.mean()) * 100
            rows.append([[mean, mean, mean + 10, mean + 10, 0.9, 1.0], [1, 2, 3, 4, 0.3, 2.0],
                         [5, 6, 7, 8, 0.02, 0.0], [0, 0, 1, 1, 0.01, 1.0]])
        return numpy.array(rows, dtype=numpy.float32)


    def predict(self, task, weights, images, options, conf, workdir, *, agnostic):
        self.calls.append(("predict", weights, list(images), dict(options), conf, agnostic))
        if task == "classify":  # 总是预测 0 号类别：标成其他类别的裁剪都是错分
            for path in images:
                yield [{"class": 0, "score": 0.9, "probs": [0.9, 0.1]}]
            return
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
        self.dataset_id = dataset.prepare(self.repo, self.root, self.spec()).dataset_id

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
        newer = dataset.prepare(self.repo, self.root, self.spec()).dataset_id
        code, out, err = self.train("eval", "m", run)
        self.assertEqual(code, 0, err)
        self.assertEqual(sorted(self.record(run)["evaluations"]), sorted([self.dataset_id, newer]))
        self.assertIn(f"on the test split of dataset {newer}", out)

    def test_the_device_belongs_to_the_machine_not_to_the_run(self):
        os.environ[runs.DEVICE_ENV] = "1"
        self.assertEqual(self.train("run", "m")[0], 0)
        run = self.only_run()
        self.assertEqual(self.record(run)["options"]["device"], "1")             # model.yaml 没有 device
        self.assertEqual(self.backend.calls[-1][-1], "1")
        os.environ[runs.DEVICE_ENV] = "0,1"
        self.assertEqual(self.train("run", "m", "--option", "device=0")[0], 0)   # --option 优先
        second = next(p.name for p in (self.root / "runs/m").iterdir() if p.name != run)
        self.assertEqual(self.record(second)["options"]["device"], 0)
        # 评估不沿用训练时的设备：--device，其次 TRAIN_DEVICE，都没有时才用 run 的设备。
        self.assertEqual(self.train("eval", "m", run)[0], 0)
        self.assertEqual(self.backend.calls[-1][-1], "0,1")
        self.assertEqual(self.train("eval", "m", run, "--device", "cpu")[0], 0)
        self.assertEqual(self.backend.calls[-1][-1], "cpu")
        del os.environ[runs.DEVICE_ENV]
        self.assertEqual(self.train("eval", "m", run)[0], 0)
        self.assertEqual(self.backend.calls[-1][-1], "1")
        self.assertIn("--split goes with --errors", self.train("eval", "m", run, "--split", "val")[2])

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
        [(_, weights, images, options, conf, agnostic)] = [c for c in self.backend.calls if c[0] == "predict"]
        self.assertEqual(weights, self.root / "runs/m" / self.run_id / "train" / ultralytics.BEST)
        self.assertEqual([p.name for p in images], ["n_0.jpg", "n_1.jpg"])
        self.assertEqual((options["device"], conf, agnostic), ("1", 0.3, True))  # 一个目标只留一个框
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
        names = lambda: {s["image"] for s in dataset.prepare(self.repo, self.root, self.spec()).manifest["samples"]
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
        original = path.read_text()
        path.write_text(original.replace("  - 防撞桶\n", ""))
        code, _, err = self.train("prelabel", "m", "--source", "batch-02", "--run", self.run_id)
        self.assertIn(f"run {self.run_id} cannot prelabel: it has classes that m/model.yaml does not (防撞桶)", err)
        # 规格新增了类别：训练于旧类别的模型仍可预标注旧类别，人只补新类别（§15.4）
        path.write_text(original.replace("  - 防撞桶\n", "  - 防撞桶\n  - 新类别\n"))
        code, out, err = self.train("prelabel", "m", "--source", "batch-02", "--run", self.run_id)
        self.assertEqual(code, 0, err)
        labels = [s.label for s in xanylabeling.read(self.directory / "n_0.json").shapes]
        self.assertEqual(labels, ["标志牌(圆牌)", "龙门架A"])


class ClassPrelabelTest(RunTestCase):
    """按类别预标注（docs/training.md §15.4）：新增 防撞墩 后，用认识它的 v0 只给复核名单中的图片追加候选框。"""
    def setUp(self):
        super().setUp()
        self.assertEqual(self.train("run", "m")[0], 0)
        self.run_id = self.only_run()
        self.assertEqual(self.train("update", "m", "add", "防撞墩")[0], 0)
        self.listed = sources.read_record(self.source())["reviews"][0]["files"]

    def v0(self, classes):
        """把 run 当作训练于这些类别的 v0（假后端预测类别 0 与 1）。"""
        record = self.record(self.run_id)
        record["classes"] = classes
        paths.write_json(self.root / "runs/m" / self.run_id / runs.RECORD, record)

    def prelabel(self):
        return self.train("prelabel", "m", "--source", "batch-01", "--class", "防撞墩", "--run", self.run_id)

    def labels(self, name):
        return [s.label for s in xanylabeling.read(self.source() / name).shapes]

    def test_candidates_are_added_to_the_unchecked_images_of_the_review_list(self):
        code, _, err = self.prelabel()
        self.assertIn(f"run {self.run_id} cannot prelabel: it does not know 防撞墩", err)
        self.v0(["防撞墩", "龙门架A"])  # 只需认识这个类别，类别列表不必与规格相同
        first = xanylabeling.annotation_path(self.source() / self.listed[0])
        self.backend.during_predict = lambda path: path.name == self.listed[0] and \
            xanylabeling.update(first, flags={"夜间": True})  # 推理期间人保存了第一张
        code, out, err = self.prelabel()
        self.assertEqual(code, 0, err)
        count = len(self.listed)
        self.assertIn(f"prelabeled {count - 1} images of batch-01 with run {self.run_id}: {count - 1} 防撞墩 boxes "
                      "added", out)
        self.assertIn("skipped 1 images: saved meanwhile", out)
        [(_, _, images, _, _, agnostic)] = [c for c in self.backend.calls if c[0] == "predict"]
        self.assertEqual(([p.name for p in images], agnostic), (self.listed, False))
        second = self.source() / self.listed[1].replace(".jpg", ".json")
        self.assertEqual(self.labels(second.name), ["标志牌(圆牌)", "防撞墩"])  # 已有的框不动，只追加
        self.assertFalse(xanylabeling.read(second).checked)
        self.assertEqual(self.labels(first.name), ["标志牌(圆牌)"])
        self.backend.during_predict = None
        code, out, _ = self.prelabel()  # 重复运行不叠加，跳过的那张这次补上
        self.assertIn(f"prelabeled 1 images of batch-01 with run {self.run_id}: 1 防撞墩 boxes added", out)
        self.assertIn(f"skipped {count - 1} images: no new candidates", out)
        xanylabeling.update(second, checked=True)
        code, out, err = self.train("prepare", "m")
        self.assertEqual(code, 0, err)
        self.assertIn(f"prelabel of batch-01 by run {self.run_id} for 防撞墩 at ", out)
        self.assertIn(f": 1 of {count - 1} images checked; 1 of 1 predicted boxes kept (100%), 0 removed, 0 added",
                      out)
        code, out, _ = self.prelabel()
        self.assertIn("skipped 1 images: already checked", out)

    def test_only_classes_with_a_review_list(self):
        self.v0(CLASSES)
        code, _, err = self.train("prelabel", "m", "--source", "batch-01", "--class", "龙门架A", "--run", self.run_id)
        self.assertIn("source batch-01 has no review list for 龙门架A", err)
        code, _, err = self.train("prelabel", "m", "--source", "batch-01", "--class", "护栏", "--run", self.run_id)
        self.assertIn("护栏 is not a class of m", err)


class ChangedClassesGateTest(RunTestCase):
    """类别改变之后的发布（docs/training.md §15.6）：在共有类别上比较，release.json 记录类别变化与按来源统计。
    假后端只报告类别 0 的每类指标，值按 run 区分。"""
    def setUp(self):
        super().setUp()
        self.assertEqual(self.train("run", "m")[0], 0)
        self.first = self.only_run()
        self.current = {"schema": 1, "model": "m", "release": "r1", "classes": CLASSES,
                        "run": {"id": self.first, "weights_sha256": self.record(self.first)["weights"]["best_sha256"]}}

    def retrain(self, *update):
        """train update，重新勾选被撤销的图片，prepare 并训练；返回 (新 run, 它在新数据集上的评估)。"""
        code, _, err = self.train("update", "m", *update)
        self.assertEqual(code, 0, err)
        for review in sources.read_record(self.source()).get("reviews") or []:
            for name in review["files"]:
                xanylabeling.update(self.source() / name.replace(".jpg", ".json"), checked=True)
        self.dataset_id = dataset.prepare(self.repo, self.root, self.spec()).dataset_id
        self.assertEqual(self.train("run", "m")[0], 0)
        second = max(p.name for p in (self.root / "runs/m").iterdir() if p.name != self.first)
        return second, publish._metrics(self.root, self.spec(), second, self.dataset_id, None)

    def gate(self, evaluation, accept=False):
        return publish.check_gate(self.repo, self.root, self.spec(), evaluation, self.current, accept)

    def test_a_new_class_is_compared_on_the_classes_both_models_know(self):
        patcher = mock.patch.object(detect, "macro_resolution", return_value=(0.0, None))  # 只检查比较本身
        patcher.start()
        self.addCleanup(patcher.stop)
        second, evaluation = self.retrain("add", "防撞墩")
        self.backend.map50_95 = {self.first: 0.6, second: 0.5}
        with self.assertRaisesRegex(TrainError, "map50_95 \\(mean over the 1 shared classes\\) 0.5000 is 0.1000 "
                                                "below the published release r1"):
            self.gate(evaluation)
        evaluated = [c for c in self.backend.calls if c[0] == "evaluate" and c[1].parents[2].name == self.first]
        self.assertIn(f"/eval/{self.dataset_id}-as-", str(evaluated[-1][2]))  # 旧模型在换算过类别号的视图上评估
        self.backend.map50_95 = {self.first: 0.6, second: 0.65}
        shutil.rmtree(self.root / "runs/m" / second / "eval")  # 去掉缓存的评估，按新的假指标重新评估
        evaluation = publish._metrics(self.root, self.spec(), second, self.dataset_id, None)
        regression = self.gate(evaluation)["regression"]
        self.assertEqual((regression["classes"], regression["published"], regression["passed"]),
                         (["标志牌(圆牌)"], 0.6, True))
        self.assertEqual(publish.class_changes(self.repo, self.spec(), self.current),
                         {"renamed": {}, "removed": [], "added": ["防撞墩"]})

    def test_renamed_classes_are_matched_by_the_recorded_rename(self):
        second, evaluation = self.retrain("rename", "标志牌(圆牌)", "圆牌")
        self.assertEqual(lock.renamed(self.repo, "m"), {"标志牌(圆牌)": "圆牌"})
        regression = self.gate(evaluation)["regression"]
        self.assertEqual((regression["classes"], regression["renamed"]), (["圆牌"], {"标志牌(圆牌)": "圆牌"}))
        self.assertEqual(publish.class_changes(self.repo, self.spec(), self.current),
                         {"renamed": {"标志牌(圆牌)": "圆牌"}, "removed": [], "added": []})
        self.assertTrue(lock.clear_renamed(self.repo / "models/m"))  # 发布后清空
        self.assertEqual(lock.renamed(self.repo, "m"), {})

    def test_the_class_a_new_class_was_split_from_is_not_compared(self):
        second, evaluation = self.retrain("add", "标志牌(方牌)", "--from", "标志牌(圆牌)")
        with self.assertRaisesRegex(TrainError, "cannot compare with the published release r1: no class it shares "
                                                "with model.yaml has test instances.*--accept-metrics"):
            self.gate(evaluation)
        self.assertTrue(self.gate(evaluation, accept=True)["accepted_without_comparison"])

    def test_training_data_is_counted_per_source(self):
        code, _, err = self.train("update", "m", "add", "防撞墩")
        self.assertEqual(code, 0, err)
        [review] = sources.read_record(self.source())["reviews"]
        for name in review["files"][1:]:  # 第一张还没复核
            xanylabeling.update(self.source() / name.replace(".jpg", ".json"), checked=True)
        dataset_id = dataset.prepare(self.repo, self.root, self.spec()).dataset_id
        count = len(review["files"])
        entry = publish.training_data(self.root, self.spec(), dataset_id)["batch-01"]
        self.assertEqual((entry["images"], entry["negatives"]), (count - 1, 0))
        self.assertEqual(entry["instances"], {"标志牌(圆牌)": count - 1, "龙门架A": 0, "防撞桶": 0, "防撞墩": 0})
        self.assertEqual(entry["skipped"], {"waiting for review of 防撞墩": 1})
        self.assertEqual([(r["class"], r["checked"], r["total"]) for r in entry["reviews"]],
                         [("防撞墩", count - 1, count)])


class RenameRecordTest(TrainingTestCase):
    def test_renames_are_composed(self):
        lock.record_rename(self.repo, "m", "A", "B")
        lock.record_rename(self.repo, "m", "B", "C")
        self.assertEqual(lock.renamed(self.repo, "m"), {"A": "C"})
        lock.write(self.repo, "m", {"batch-01": ["x"]})  # 写来源记录时保留改名记录
        self.assertEqual(lock.renamed(self.repo, "m"), {"A": "C"})
        lock.record_rename(self.repo, "m", "C", "A")  # 改回原名
        self.assertEqual(lock.renamed(self.repo, "m"), {})
        self.assertEqual(lock.read(self.repo, "m"), {"batch-01": ["x"]})


class UltralyticsPredictTest(unittest.TestCase):
    """适配器把 agnostic 交给 Ultralytics 的 agnostic_nms（它默认按类别分别做 NMS）。Ultralytics 用替身。"""

    def test_agnostic_reaches_agnostic_nms_and_results_keep_image_order(self):
        calls = []

        class Boxes:
            def __init__(self, rows):
                self.xyxy = types.SimpleNamespace(tolist=lambda: [r[:4] for r in rows])
                self.cls = types.SimpleNamespace(tolist=lambda: [r[4] for r in rows])
                self.conf = types.SimpleNamespace(tolist=lambda: [r[5] for r in rows])

        class YOLO:
            def __init__(self, weights, task):
                calls.append(("load", weights, task))

            def predict(self, sources, **arguments):
                calls.append(("predict", sources, arguments))
                return [types.SimpleNamespace(boxes=Boxes([[1, 2, 3, 4, 2.0, 0.8]])),
                        types.SimpleNamespace(boxes=Boxes([]))]

        stub = types.SimpleNamespace(YOLO=YOLO, settings=types.SimpleNamespace(update=lambda _: None))
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict("sys.modules", {"ultralytics": stub}), \
                mock.patch.object(ultralytics, "use_cjk_font"):
            for agnostic in (True, False):
                results = list(ultralytics.predict("detect", Path(tmp) / "best.pt", [Path("/a.jpg"), Path("/b.jpg")],
                                                   {"imgsz": 640, "device": "0,1", "epochs": 3}, 0.25, Path(tmp),
                                                   agnostic=agnostic))
                self.assertEqual(results, [[{"class": 2, "box": [1.0, 2.0, 3.0, 4.0], "score": 0.8}], []])
                _, sources_, arguments = calls[-1]
                self.assertEqual(sources_, ["/a.jpg", "/b.jpg"])
                self.assertEqual(arguments, {"conf": 0.25, "stream": True, "verbose": False,
                                             "agnostic_nms": agnostic, "imgsz": 640, "device": "0"})


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
        self.assertNotIn("class_changes", release)  # 首次发布没有可对照的类别
        self.assertEqual(release["training_data"]["batch-01"]["negatives"], 0)
        self.assertEqual(release["training_data"]["batch-01"]["skipped"], {})
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

    def test_publishing_after_a_rename_records_the_change_and_clears_the_rename(self):
        self.assertEqual(self.train("publish", "m", self.run_id, "--accept-metrics")[0], 0)
        self.assertEqual(self.train("update", "m", "rename", "标志牌(圆牌)", "圆牌")[0], 0)
        self.assertEqual(self.train("prepare", "m")[0], 0)
        self.assertEqual(self.train("run", "m")[0], 0)
        second = max(p.name for p in (self.root / "runs/m").iterdir() if p.name != self.run_id)
        code, out, err = self.train("publish", "m", second)
        self.assertEqual(code, 0, err)
        self.assertIn("compared on the shared classes: 圆牌", out)
        self.assertIn("sources.lock.json", out)  # 改名记录随发布清空
        release = json.loads((self.model_dir / "release.json").read_text(encoding="utf-8"))
        self.assertEqual(release["class_changes"], {"renamed": {"标志牌(圆牌)": "圆牌"}, "removed": [], "added": []})
        self.assertEqual(release["gate"]["regression"]["classes"], ["圆牌"])
        self.assertEqual(lock.renamed(self.repo, "m"), {})
        self.assertEqual(models.check_model(self.model_dir), [])

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
        # 这里检查比较本身：把一个测试实例造成的变化设为 0（实例数的影响见下一个测试）。
        patcher = mock.patch.object(detect, "macro_resolution", return_value=(0.0, None))
        patcher.start()
        self.addCleanup(patcher.stop)
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

    def test_with_few_test_instances_one_instance_sets_the_allowed_drop(self):
        self.assertEqual(self.train("publish", "m", self.run_id, "--accept-metrics")[0], 0)
        second = self.second_run()
        self.backend.map50_95[second] = 0.48                    # 下降 0.02：大于 max_drop，却小于一个测试实例的影响
        shutil.rmtree(self.root / "runs/m" / second / "eval")
        code, out, err = self.train("publish", "m", second)
        self.assertEqual(code, 0, err)
        regression = json.loads((self.model_dir / "release.json").read_text())["gate"]["regression"]
        test = [s for s in json.loads((self.root / "datasets/m" / self.dataset_id / "manifest.json").read_text())[
            "samples"] if s["split"] == "test"]
        count = sum(len(s["objects"]) for s in test)
        self.assertEqual(regression["resolution"], {"one_instance": 1 / count, "class": CLASSES[0], "instances": count})
        self.assertEqual((regression["allowed"], regression["passed"]), (1 / count, True))
        self.assertIn(f"allowed {1 / count:.4f}: one test instance of {CLASSES[0]} ({count} in the test split) moves "
                      f"map50_95 by {1 / count:.4f}, more than max_drop 0.01; add test instances for a tighter gate",
                      out)
        self.assertIn("(drop 0.0200)", out)

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
        (self.repo / "models/m/release.json").write_text(json.dumps({"release": "r0", "run": {"id": "elsewhere"}}))
        self.assertNotIn("* the run of", self.train("publish", "m")[1])           # 发布的 run 不在列表中时不标注

        image(self.tmp / "more/x/x.jpg", "white")
        annotation(self.tmp / "more/x/x.json", LABELED, checked=True)
        self.import_(self.tmp / "more")
        newer = dataset.prepare(self.repo, self.root, self.spec()).dataset_id            # 新数据集上还没有评估
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


def boxes(path: Path, items, checked=True, size=(640, 480)) -> Path:
    """写一份标注：items 是 (类别, [x1, y1, x2, y2], 属性) 的列表，形状格式取自真实样例；size 是图像的宽高。"""
    document = json.loads(LABELED.read_text(encoding="utf-8"))
    [template] = document["shapes"]
    shapes = []
    for label, (x1, y1, x2, y2), values in items:
        shapes.append({**template, "label": label, "attributes": values,
                       "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]})
    document.update({"imagePath": path.with_suffix(".jpg").name, "checked": checked, "shapes": shapes,
                     "imageWidth": size[0], "imageHeight": size[1]})
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def groups_in(split_name: str, count: int, source="batch-01") -> list[str]:
    """count 个按目录分组时落在 split_name 划分的采集组名（与 split.assign 的规则相同）。"""
    found, index = [], 0
    while len(found) < count:
        if split.assign(f"dir:{source}/g{index}", 42, 0.15, 0.15) == split_name:
            found.append(f"g{index}")
        index += 1
    return found


class SecondStageTestCase(TrainingTestCase):
    """父模型 m（检测，CLASSES）与二级模型 occ：作用于“标志牌(圆牌)”，类别是属性“遮挡”的值。"""
    PARENT_CLASS = CLASSES[0]

    def setUp(self):
        super().setUp()
        code, out, err = self.train("init", "occ", "--task", "classify", "--class", "否", "--class", "是",
                                    "--parent", "m", "--parent-class", self.PARENT_CLASS, "--attribute", "遮挡")
        self.assertEqual(code, 0, err)
        self.set_child(sources=["batch-01"], ignore_classes=["未知"])

    def set_child(self, **data):
        path = spec.path(self.repo, "occ")
        text = path.read_text(encoding="utf-8")
        for key, value in data.items():
            old = next(line for line in text.splitlines() if line.startswith(f"  {key}:"))
            text = text.replace(old, f"  {key}: {json.dumps(value, ensure_ascii=False)}")
        path.write_text(text, encoding="utf-8")

    def child(self):
        return spec.load(self.repo, "occ")

    def edit_child(self, old, new):
        path = spec.path(self.repo, "occ")
        text = path.read_text(encoding="utf-8")
        self.assertIn(old, text)
        path.write_text(text.replace(old, new), encoding="utf-8")

    def labeled_groups(self, per_split=(("train", 2), ("val", 1), ("test", 1))):
        """每个划分若干采集组，每组一张图：一个“遮挡=是”的大框与一个“遮挡=否”的框，导入来源。"""
        for split_name, count in per_split:
            for group in groups_in(split_name, count):
                shade = int(group[1:]) * 3 % 256    # 每张图不同：导入按内容去重
                image(self.raw / group / f"{group}.jpg", (shade, 80, 120), size=(640, 480))
                boxes(self.raw / group / f"{group}.json", [
                    (self.PARENT_CLASS, [10, 10, 110, 90], {"遮挡": "是"}),
                    (self.PARENT_CLASS, [200, 50, 260, 130], {"遮挡": "否"})])
        self.import_(self.raw)


class SecondStageSpecTest(SecondStageTestCase):
    def test_init_writes_a_valid_second_stage_spec(self):
        child = self.child()
        self.assertEqual((child["task"], child["classes"], child["cascade"], child["data"]["from_parent"]),
                         ("classify", ["否", "是"], {"parent": "m", "classes": [self.PARENT_CLASS]},
                          {"attribute": "遮挡", "min_size": 16}))
        self.assertEqual((child["output_layout"], child["input"]), ("softmax", {"width": 128, "height": 128}))
        self.assertEqual(models.check_spec("occ", child), [])

    def test_init_needs_the_parent_and_leaves_nothing_behind_on_errors(self):
        code, _, err = self.train("init", "x", "--task", "classify", "--class", "a")
        self.assertIn("give --parent <model>", err)
        code, _, err = self.train("init", "x", "--task", "classify", "--class", "a", "--parent", "m",
                                  "--parent-class", "没有", "--attribute", "遮挡")
        self.assertEqual(code, 1)
        self.assertIn("cascade.classes: m has no class 没有", err)
        self.assertFalse(spec.path(self.repo, "x").exists())
        code, _, err = self.train("init", "x", "--task", "detect", "--class", "a", "--parent", "m")
        self.assertIn("only for --task classify", err)

    def test_second_stage_fields_are_checked(self):
        cases = [("parent: m", "parent: nope", "cascade.parent: model nope does not exist"),
                 ("parent: m", "parent: occ", "cascade.parent must name another model"),
                 ("min_size: 16", "min_size: 16\n    padding: 0.1", "padding is not supported: the second-stage "
                                                                     "nvinfer crops the detected box itself"),
                 ("min_size: 16", "min_size: 0", "min_size must be an integer >= 1"),
                 ("attribute: 遮挡", "attribute: ''", "data.from_parent.attribute must name"),
                 ("width: 128", "width: 160", "input of a classify model must be square")]
        for old, new, message in cases:
            with self.subTest(new=new):
                self.edit_child(old, new)
                with self.assertRaisesRegex(TrainError, message):
                    self.child()
                self.edit_child(new, old)
        self.edit_child("cascade:\n", "cascade_off:\n")
        with self.assertRaisesRegex(TrainError, "cascade is required(.|\n)*whole-image classification is not "
                                                "supported yet"):
            self.child()

    def test_the_parent_must_detect(self):
        (self.repo / "models/occ2").mkdir()
        spec.path(self.repo, "occ2").write_text(spec.path(self.repo, "occ").read_text().replace(
            "id: occ", "id: occ2").replace("parent: m", "parent: occ"))
        with self.assertRaisesRegex(TrainError, "occ is a classify model; the parent must detect objects"):
            spec.load(self.repo, "occ2")

    def test_commands_that_treat_classes_as_labels_are_refused(self):
        code, _, err = self.train("import", "occ", str(self.raw), "--source", "batch-01")
        self.assertIn("occ is a second-stage model: its samples are boxes of m", err)
        code, _, err = self.train("update", "occ", "rename", "否", "无遮挡")
        self.assertIn("values of the box attribute 遮挡, not source labels", err)
        code, _, err = self.train("prelabel", "occ", "--source", "batch-01")
        self.assertIn("prelabel is not supported for task classify yet", err)


class SecondStagePrepareTest(SecondStageTestCase):
    def test_samples_are_parent_boxes_classified_by_the_attribute(self):
        [group] = groups_in("train", 1)
        image(self.raw / group / "a.jpg", "red", size=(640, 480))
        boxes(self.raw / group / "a.json", [
            (self.PARENT_CLASS, [10, 10, 110, 90], {"遮挡": "是"}),
            (self.PARENT_CLASS, [200, 50, 260, 130], {"遮挡": "否", "破损": "是"}),
            (self.PARENT_CLASS, [300, 50, 360, 130], {"遮挡": "未知"}),          # data.ignore_classes
            (self.PARENT_CLASS, [400, 50, 460, 130], {}),                         # 没有这个属性
            (self.PARENT_CLASS, [500, 50, 510, 60], {"遮挡": "是"}),             # 小于 min_size
            ("龙门架A", [0, 200, 300, 300], {"遮挡": "是"}),                      # 不是触发的父类别
        ])
        self.import_(self.raw)
        result = dataset.prepare(self.repo, self.root, self.child())
        [sample] = result.manifest["samples"]
        self.assertEqual([(o["class"], o["box"], o["parent"]) for o in sample["objects"]],
                         [(1, [10, 10, 110, 90], self.PARENT_CLASS), (0, [200, 50, 260, 130], self.PARENT_CLASS)])
        self.assertEqual(result.ignored_shapes, {"boxes with 遮挡=未知 (data.ignore_classes)": 1,
                                                 "boxes without attribute 遮挡": 1,
                                                 "boxes smaller than 16 px (data.from_parent.min_size)": 1})
        self.assertEqual((result.manifest["input"], result.manifest["from_parent"]),
                         ({"width": 128, "height": 128}, {"attribute": "遮挡", "min_size": 16}))
        lines = dataset.report(result, self.child())
        self.assertIn("  ignored 1 boxes without attribute 遮挡", lines)
        # 父模型的数据集清单没有额外字段：已有检测数据集的标识不变。
        self.assertNotIn("input", dataset.prepare(self.repo, self.root, self.spec()).manifest)

    def test_values_that_are_neither_classes_nor_ignored_are_errors(self):
        [group] = groups_in("train", 1)
        image(self.raw / group / "a.jpg", "red", size=(640, 480))
        boxes(self.raw / group / "a.json", [(self.PARENT_CLASS, [10, 10, 110, 90], {"遮挡": "也许"})])
        self.import_(self.raw)
        with self.assertRaisesRegex(TrainError, "遮挡='也许' is neither a class nor in data.ignore_classes"):
            dataset.prepare(self.repo, self.root, self.child())

    def test_the_view_holds_stretched_crops_in_class_number_folders(self):
        from PIL import Image
        self.labeled_groups()
        result = dataset.prepare(self.repo, self.root, self.child())
        view = result.directory / paths.VIEW / "ultralytics"
        self.assertEqual(ultralytics.data_path("classify", view), view)
        for split_name in split.SPLITS:  # 每个划分都有全部类别目录，类别号不会错位
            self.assertEqual(sorted(p.name for p in (view / split_name).iterdir()), ["000", "001"])
        crops = sorted((view / "train/001").iterdir())
        self.assertEqual(len(crops), 2)
        with Image.open(crops[0]) as crop:
            self.assertEqual(crop.size, (128, 128))   # 100x80 的框拉伸到网络输入，不保持宽高比
        self.assertEqual(len(list((view / "test/000").iterdir())), 1)

    def test_every_class_needs_training_samples(self):
        self.labeled_groups(per_split=(("val", 1),))
        with self.assertRaisesRegex(TrainError, "the train split has no samples of 否, 是"):
            dataset.prepare(self.repo, self.root, self.child())

    def test_the_sources_must_label_the_parent_classes(self):
        self.labeled_groups()
        self.edit_child(f"    - {self.PARENT_CLASS}\n", "    - 龙门架A\n")
        source_labels = lock.current(self.root, self.child())
        self.assertEqual(lock.missing_classes(self.child(), source_labels), [])
        record = sources.read_record(self.source())
        record["classes"] = ["防撞桶"]
        paths.write_json(self.source() / paths.SOURCE_RECORD, record)
        [problem] = lock.missing_classes(self.child(), lock.current(self.root, self.child()))
        self.assertIn("has no class 龙门架A (cascade.classes of occ)", problem)

    def test_renaming_a_parent_class_points_at_the_second_stage_model(self):
        self.labeled_groups()
        dataset.prepare(self.repo, self.root, self.spec())
        dataset.prepare(self.repo, self.root, self.child())
        result = update.rename(self.repo, self.root, self.spec(), self.PARENT_CLASS, "圆牌")
        self.assertTrue(any("second-stage model occ triggers on 标志牌(圆牌): rename it to 圆牌 in its "
                            "cascade.classes" in line for line in result.lines), result.lines)
        with self.assertRaisesRegex(TrainError, "cascade.classes: m has no class 标志牌\\(圆牌\\)"):
            self.child()


class AttributesTest(SecondStageTestCase):
    def setUp(self):
        super().setUp()
        [group] = groups_in("train", 1)
        image(self.raw / group / "a.jpg", "red", size=(640, 480))
        boxes(self.raw / group / "a.json", [
            (self.PARENT_CLASS, [10, 10, 110, 90], {"遮挡": "是", "破损": "否"}),
            ("防撞桶", [200, 50, 260, 130], {"破损": "是", "数量": "一个"}),
            ("龙门架A", [300, 50, 360, 130], {"遮挡": "也许"}),
        ], checked=False)  # 未检查的标注中的属性同样保留：它们也会在工具中被编辑
        self.import_(self.raw)

    def test_the_file_joins_the_model_attribute_with_the_attributes_in_the_source(self):
        code, out, err = self.train("prepare", "occ")   # 标注都没有勾选：prepare 失败
        self.assertEqual((code, "no checked images in batch-01" in err), (1, True))
        self.assertIn("warning: 1 boxes of batch-01 have 遮挡=也许, which occ does not accept", out)
        # 失败之前已经写好属性文件：改正标注所需的取值已经在里面。
        document = json.loads((self.source() / "attributes.json").read_text(encoding="utf-8"))
        # 模型声明的属性在触发类别上，取值为 classes + ignore_classes；同名属性的取值是各类别上的并集。
        self.assertEqual(document[self.PARENT_CLASS], {"破损": ["否", "是"], "遮挡": ["否", "是", "未知", "也许"]})
        self.assertEqual(document["防撞桶"], {"数量": ["一个"], "破损": ["否", "是"]})
        self.assertEqual(document["龙门架A"], {"遮挡": ["否", "是", "未知", "也许"]})
        self.assertEqual(list(document)[:3], [self.PARENT_CLASS, "龙门架A", "防撞桶"])   # labels.txt 的顺序
        self.assertEqual(document["__widget_types__"]["防撞桶"], {"数量": "radiobutton", "破损": "radiobutton"})
        self.assertIn("attributes: " + str(self.source() / "attributes.json"), out)

    def test_import_writes_the_file_and_update_rewrites_it(self):
        self.assertFalse((self.source() / "attributes.json").exists())  # setUp 直接导入，没有经过命令
        image(self.raw / "more/b.jpg", "blue", size=(640, 480))
        code, out, err = self.train("import", "m", str(self.raw / "more"), "--source", "batch-01")
        self.assertEqual(code, 0, err)
        self.assertTrue((self.source() / "attributes.json").is_file())
        # 父类别改名后二级模型的 cascade.classes 还是旧名：update 照常完成，属性文件暂不计入它并说明原因。
        lock.check(self.repo, self.root, self.spec(), record_new=True)  # rename 要求确认过的类别依赖
        code, out, err = self.train("update", "m", "rename", self.PARENT_CLASS, "圆牌")
        self.assertEqual(code, 0, err)
        self.assertIn("note: the attributes of occ are not in attributes.json until its model.yaml is valid again", out)
        document = json.loads((self.source() / "attributes.json").read_text(encoding="utf-8"))
        self.assertEqual(document["圆牌"], {"破损": ["否", "是"], "遮挡": ["也许", "是"]})  # 只剩标注中已有的

    def test_many_values_use_a_combobox(self):
        boxes(self.source() / "a.json", [("防撞桶", [1, 1, 50, 50], {"编号": str(n)}) for n in range(7)])
        result = attributes.generate(self.repo, self.root, "batch-01")
        self.assertEqual(result.document["__widget_types__"]["防撞桶"]["编号"], "combobox")

    def test_models_must_agree_on_the_values_of_an_attribute(self):
        (self.repo / "models/occ2").mkdir()
        spec.path(self.repo, "occ2").write_text(spec.path(self.repo, "occ").read_text(encoding="utf-8").replace(
            "id: occ", "id: occ2").replace("ignore_classes: [\"未知\"]", "ignore_classes: []"), encoding="utf-8")
        with self.assertRaisesRegex(TrainError, "models occ and occ2 give different values for attribute 遮挡"):
            attributes.generate(self.repo, self.root, "batch-01")


class ClassifyTaskTest(unittest.TestCase):
    def test_metrics_from_the_confusion_matrix(self):
        result = classify.metrics([[90, 5], [3, 2]])
        self.assertAlmostEqual(result["metrics"]["accuracy"], 0.92)
        self.assertAlmostEqual(result["metrics"]["recall"], (90 / 95 + 2 / 5) / 2)
        self.assertAlmostEqual(result["metrics"]["precision"], (90 / 93 + 2 / 7) / 2)
        self.assertEqual(result["classes"][1], {"accuracy": None, "recall": 2 / 5, "precision": 2 / 7})
        # 测试划分中没有样本的类别不列出，也不进入宏平均。
        self.assertEqual(sorted(classify.metrics([[4, 0], [0, 0]])["classes"]), [0])
        self.assertEqual(classify.metrics([[4, 0], [0, 0]])["metrics"]["recall"], 1.0)

    def test_one_test_instance_sets_the_resolution_of_a_metric(self):
        counts = {"否": 802, "是": 22, "从未出现": 0}
        self.assertEqual(classify.resolution("recall", counts), (1 / (2 * 22), "是"))   # 宏平均：稀有类别决定
        self.assertEqual(classify.resolution("accuracy", counts), (1 / 824, None))      # 按全部裁剪计
        self.assertEqual(detect.resolution("map50_95", {"a": 3, "b": 40}), (1 / 6, "a"))
        self.assertEqual(detect.resolution("map50_95", {}), (0.0, None))

    def test_labels_are_one_semicolon_separated_line(self):
        self.assertEqual(classify.labels_text(["否", "是"]), "否;是\n")

    @needs_numpy
    def test_softmax_outputs_are_compared(self):
        native = numpy.array([[0.9, 0.1], [0.5002, 0.4998], [0.2, 0.8]], dtype=numpy.float32)
        self.assertTrue(classify.compare("softmax", native, native.copy())["passed"])
        near_tie = native.copy()
        near_tie[1] = [0.4999, 0.5001]  # 几乎相等的两类可以在导出后交换名次
        result = classify.compare("softmax", native, near_tie)
        self.assertEqual((result["passed"], result["class_mismatches"]), (True, 0))
        flipped = native.copy()
        flipped[0] = [0.1, 0.9]
        result = classify.compare("softmax", native, flipped)
        self.assertEqual((result["passed"], result["class_mismatches"]), (False, 1))
        self.assertIn("output shapes differ", classify.compare("softmax", native, native[:, :1])["reason"])


CLASSIFIER_NVINFER = """[property]
onnx-file=artifacts/model.onnx
labelfile-path=artifacts/labels.txt
model-engine-file=/data/engines/occ/model_b1_gpu0_fp16.engine
batch-size=1
network-type=1
process-mode=2
gie-unique-id=2
operate-on-gie-id=1
operate-on-class-ids=0
"""


@needs_numpy
class SecondStagePublishTest(SecondStageTestCase):
    """假后端训练并发布二级模型；父模型 m 以最小的已发布目录代替（只供级联校验读取类别与 gie-unique-id）。"""
    def setUp(self):
        super().setUp()
        self.backend = FakeBackend()
        backends.BACKENDS["fake"] = self.backend
        self.addCleanup(backends.BACKENDS.pop, "fake")
        self.edit_child("backend: ultralytics", "backend: fake")
        self.labeled_groups()
        parent = self.repo / "models/m"
        (parent / "release.json").write_text(json.dumps({"classes": CLASSES}))
        (parent / "nvinfer.txt").write_text("[property]\ngie-unique-id=1\n")
        self.model_dir = self.repo / "models/occ"
        (self.model_dir / "nvinfer.txt").write_text(CLASSIFIER_NVINFER, encoding="utf-8")
        self.onnx = FakeOnnx(self.backend)
        self.onnx.signature = lambda path: {"inputs": [{"name": "input", "shape": [1, 3, 128, 128]}],
                                            "outputs": [{"name": "output", "shape": [1, 2]}]}
        self.onnx.run = lambda path, batch: self.backend.reference("classify", None, "softmax", batch)
        patcher = mock.patch.object(publish, "onnxio", self.onnx)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_a_second_stage_model_is_trained_evaluated_and_published(self):
        code, out, err = self.train("prepare", "occ")
        self.assertEqual(code, 0, err)
        code, out, err = self.train("run", "occ")
        self.assertEqual(code, 0, err)
        self.assertIn("accuracy 0.7857  recall 0.7750  precision 0.7444", out)
        [run] = [p.name for p in (self.root / "runs/occ").iterdir()]
        [train_call] = [c for c in self.backend.calls if c[0] == "train"]
        self.assertEqual(train_call[2]["imgsz"], 128)
        [evaluate_call] = [c for c in self.backend.calls if c[0] == "evaluate"]
        self.assertTrue(evaluate_call[2].is_dir())      # 分类的 data 是视图目录
        code, out, err = self.train("publish", "occ", run, "--accept-metrics")
        self.assertEqual(code, 0, err)
        release = json.loads((self.model_dir / "release.json").read_text(encoding="utf-8"))
        self.assertEqual((release["output_layout"], release["metrics"]["recall"]), ("softmax", 0.775))
        self.assertTrue(release["consistency"]["passed"])
        self.assertEqual(release["consistency"]["crops"], 2)   # 测试划分中一张图的两个裁剪
        self.assertEqual(release["training_data"]["batch-01"]["instances"], {"否": 4, "是": 4})
        self.assertEqual((self.model_dir / "artifacts/labels.txt").read_text(encoding="utf-8"), "否;是\n")
        self.assertIn("2 crops compared", out)
        [export] = [c for c in self.backend.calls if c[0] == "export"]
        self.assertEqual((export[2], export[3]), ("softmax", (128, 128)))
        self.assertEqual(models.check_model(self.model_dir), [])

    def test_misclassified_crops_are_listed_with_their_source_box(self):
        self.assertEqual(self.train("prepare", "occ")[0], 0)
        self.assertEqual(self.train("run", "occ")[0], 0)
        [run] = [p.name for p in (self.root / "runs/occ").iterdir()]
        code, out, err = self.train("eval", "occ", run, "--errors", "--split", "train")
        self.assertEqual(code, 0, err)
        self.assertIn("2 of 4 crops in the train split are misclassified", out)
        self.assertIn("labeled 是, predicted 否 0.900", out)
        self.assertIn("[10, 10, 110, 90] 标志牌(圆牌)", out)
        [dataset_id] = [p.name for p in (self.root / "runs/occ" / run / "eval").iterdir()]
        document = json.loads((self.root / "runs/occ" / run / "eval" / dataset_id / "errors-train.json").read_text())
        self.assertEqual({(e["label"], e["predicted"], e["source"]) for e in document["errors"]},
                         {("是", "否", "batch-01")})
        self.assertIn("--split goes with --errors", self.train("eval", "occ", run, "--split", "val")[2])
        self.assertIn("lists misclassified crops of classify models", self.train("eval", "m", run, "--errors")[2])

    def review_of_train_errors(self):
        self.assertEqual(self.train("prepare", "occ")[0], 0)
        self.assertEqual(self.train("run", "occ")[0], 0)
        [run] = [p.name for p in (self.root / "runs/occ").iterdir()]
        code, out, err = self.train("eval", "occ", run, "--errors", "--split", "train", "--split", "val")
        self.assertEqual(code, 0, err)
        name = f"occ-{run}-train-val"
        self.assertIn(f"train review apply {name}", out)
        return name, self.root / "reviews" / name

    def test_a_review_holds_only_the_images_to_check_and_applies_edits_back(self):
        name, folder = self.review_of_train_errors()
        [group, *_] = groups_in("train", 2)
        image_name = f"batch-01__{group}.jpg"
        self.assertEqual(sorted(p.name for p in folder.iterdir() if p.suffix == ".jpg"),
                         sorted(f"batch-01__{g}.jpg" for g in [*groups_in("train", 2), *groups_in("val", 1)]))
        self.assertEqual((folder / image_name).stat().st_ino, (self.source() / f"{group}.jpg").stat().st_ino)
        copy = folder / f"batch-01__{group}.json"
        self.assertEqual(json.loads(copy.read_text(encoding="utf-8"))["imagePath"], image_name)
        self.assertIn(f"[ ] {image_name}  [10, 10, 110, 90] 标志牌(圆牌)  labeled 是, model says 否 (0.90)  train",
                      (folder / "checklist.txt").read_text(encoding="utf-8"))
        self.assertIn(f"{name}: 3 images, 0 edited and not applied", self.train("review")[1])
        self.assertTrue((folder / "attributes.json").is_file())   # prepare 生成的属性文件随复核集带上
        # 人在复核集中改了一个框的属性（X-AnyLabeling 写临时文件再替换：副本换成新文件）。
        document = json.loads(copy.read_text(encoding="utf-8"))
        document["shapes"][0]["attributes"]["遮挡"] = "否"
        copy.unlink()
        copy.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
        code, out, _ = self.train("prepare", "occ")
        self.assertIn(f"warning: review {name} has 1 edited annotations that are not applied", out)
        code, out, err = self.train("review", "apply", name)
        self.assertEqual(code, 0, err)
        self.assertIn("wrote 1 annotations back, 2 unchanged", out)
        written = json.loads((self.source() / f"{group}.json").read_text(encoding="utf-8"))
        self.assertEqual((written["imagePath"], written["shapes"][0]["attributes"]["遮挡"]), (f"{group}.jpg", "否"))
        self.assertIn("wrote 0 annotations back, 3 unchanged", self.train("review", "apply", name)[1])
        self.assertNotIn("warning: review", self.train("prepare", "occ")[1])

    def test_an_annotation_changed_in_its_source_meanwhile_is_not_overwritten(self):
        name, folder = self.review_of_train_errors()
        [group, *_] = groups_in("train", 2)
        copy = folder / f"batch-01__{group}.json"
        copy.write_text(copy.read_text(encoding="utf-8").replace('"是"', '"否"'), encoding="utf-8")
        source = self.source() / f"{group}.json"
        source.write_text(source.read_text(encoding="utf-8").replace('"是"', '"未知"'), encoding="utf-8")
        before = source.read_bytes()
        code, out, _ = self.train("review", "apply", name)
        self.assertEqual(code, 1)
        self.assertIn(f"conflict: batch-01/{group}.jpg changed in its source since the review was made", out)
        self.assertEqual(source.read_bytes(), before)
        # 未应用的修改还在：重新建立同名复核集被拒绝，不会丢掉它们。
        [run] = [p.name for p in (self.root / "runs/occ").iterdir()]
        code, _, err = self.train("eval", "occ", run, "--errors", "--split", "train", "--split", "val")
        self.assertIn(f"review {name} has 1 edited annotations that are not applied yet", err)

    def test_a_config_that_acts_on_other_parent_classes_is_not_published(self):
        (self.model_dir / "nvinfer.txt").write_text(
            CLASSIFIER_NVINFER.replace("operate-on-class-ids=0", "operate-on-class-ids=2"), encoding="utf-8")
        self.assertEqual(self.train("prepare", "occ")[0], 0)
        self.assertEqual(self.train("run", "occ")[0], 0)
        [run] = [p.name for p in (self.root / "runs/occ").iterdir()]
        code, _, err = self.train("publish", "occ", run, "--accept-metrics")
        self.assertEqual(code, 1)
        self.assertIn("operate-on-class-ids=2 but cascade.classes are classes 0 of the published m", err)
        self.assertFalse((self.model_dir / "release.json").exists())


class RepositoryModelTest(unittest.TestCase):
    def test_repository_models_are_valid_training_specs(self):
        # 构建镜像的上下文不含 models/：这里在开发机上覆盖仓库中的真实规格。
        for directory in models.model_dirs(ROOT):
            spec.load(ROOT, directory.name)


if __name__ == "__main__":
    unittest.main()
