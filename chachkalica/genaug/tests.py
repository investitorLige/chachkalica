"""Tests for generative augmentation: validator, selection, builder, studio.

The backend is faked by a function that applies a classical edit chosen by a
keyword in the prompt ("dark" darkens, "shift" moves the whole frame, "boom"
fails), so everything around the model runs for real on a temp dataset.
"""

import json
import random
import tempfile
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse

from fleet.models import Dataset, FleetSettings
from genaug import jobs
from genaug.models import AugBuild, AugPreview, AugPrompt, AugSession, Status
from genaug.services import backend, builder, previews
from genaug.services.validation import GeometryThresholds, GeometryValidator

W, H = 320, 240


def textured(seed: int) -> np.ndarray:
    """A frame with real structure: smooth noise background + solid objects."""
    rng = np.random.default_rng(seed)
    small = rng.integers(40, 200, size=(H // 4, W // 4, 3), dtype=np.uint8)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_CUBIC)


def draw_object(image, box, color=(250, 250, 30)):
    x1, y1, x2, y2 = box
    image[y1:y2, x1:x2] = color
    # Inner structure, so the object is a shape rather than a flat patch.
    image[y1 + (y2 - y1) // 3:y1 + (y2 - y1) // 2, x1 + 4:x2 - 4] = (20, 20, 20)
    return image


BOX = (100, 60, 180, 190)


def to_yolo(box, class_id=0):
    x1, y1, x2, y2 = box
    return f"{class_id} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}\n"


def fake_edit(*, image_path, output_path, generation, editor_config, seed):
    """Stand-in for backend.edit: a classical edit keyed on the prompt."""
    image = cv2.imread(str(image_path)).astype(np.float32)
    text = generation["text"]
    if "boom" in text:
        raise backend.BackendError("CUDA out of memory (fake)")
    if "dark" in text:
        image = image * 0.4 + np.random.default_rng(seed).normal(0, 4, image.shape)
    if "shift" in text:
        image = np.roll(image, (int(0.06 * H), int(0.06 * W)), axis=(0, 1))
    if "vanish" in text:
        x1, y1, x2, y2 = BOX
        image[y1:y2, x1:x2] = image[y1:y2, x1 - 60:x1 - 60 + (x2 - x1)]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), np.clip(image, 0, 255).astype(np.uint8))
    return {"edit_ms": 1234}


class ValidatorTests(TestCase):
    def setUp(self):
        self.original = draw_object(textured(1), BOX)
        self.validator = GeometryValidator()

    def test_lighting_noise_and_colour_changes_pass(self):
        f = self.original.astype(np.float32)
        dark = np.clip(f * 0.35 + np.random.default_rng(0).normal(0, 6, f.shape), 0, 255)
        cast = np.clip(f * [0.8, 1.1, 0.9], 0, 255)
        for edited in (dark, cast):
            result = self.validator.validate(self.original, edited.astype(np.uint8), [BOX])
            self.assertTrue(result.valid, result.reason)
            self.assertEqual(result.boxes_checked, 1)

    def test_moved_object_is_rejected(self):
        moved = textured(1)
        x1, y1, x2, y2 = BOX
        dx = int(0.15 * (x2 - x1))
        draw_object(moved, (x1 + dx, y1, x2 + dx, y2))
        result = self.validator.validate(self.original, moved, [BOX])
        self.assertFalse(result.valid)
        self.assertIn("box 0", result.reason)

    def test_removed_object_is_rejected(self):
        result = self.validator.validate(self.original, textured(1), [BOX])
        self.assertFalse(result.valid)
        self.assertIn("changed shape or disappeared", result.reason)

    def test_whole_scene_shift_is_rejected(self):
        shifted = np.roll(self.original, (12, 20), axis=(0, 1))
        result = self.validator.validate(self.original, shifted, [BOX])
        self.assertFalse(result.valid)
        self.assertIn("whole scene shifted", result.reason)

    def test_blank_output_is_rejected(self):
        result = self.validator.validate(self.original, np.zeros_like(self.original), [BOX])
        self.assertFalse(result.valid)
        self.assertIn("blank", result.reason)

    def test_tiny_boxes_are_skipped_not_rejected(self):
        result = self.validator.validate(self.original, textured(99), [(5, 5, 11, 11)])
        self.assertTrue(result.valid)
        self.assertEqual((result.boxes_checked, result.boxes_skipped), (0, 1))

    def test_thresholds_are_honoured(self):
        lenient = GeometryValidator(GeometryThresholds(min_box_similarity=-1.0,
                                                       max_bbox_drift=10, max_global_shift=1))
        self.assertTrue(lenient.validate(self.original, textured(1), [BOX]).valid)

    def test_result_is_json_serializable(self):
        result = self.validator.validate(self.original, self.original, [BOX, (1, 1, 3, 3)])
        json.dumps(result.to_dict())


class SelectionTests(TestCase):
    def test_fraction_and_determinism(self):
        candidates = [(f"{i}.jpg", [{"class_id": 0, "bbox": (0.5, 0.5, 0.1, 0.1)}])
                      for i in range(100)]
        a = builder.select_images(candidates, ["helmet"], {}, 0.3, seed=1)
        self.assertEqual(len(a), 30)
        self.assertEqual(a, builder.select_images(candidates, ["helmet"], {}, 0.3, seed=1))
        self.assertNotEqual(a, builder.select_images(candidates, ["helmet"], {}, 0.3, seed=2))

    def test_class_weights_skew_and_zero_excludes(self):
        rare = [(f"r{i}.jpg", [{"class_id": 1, "bbox": (0.5, 0.5, 0.1, 0.1)}]) for i in range(50)]
        common = [(f"c{i}.jpg", [{"class_id": 0, "bbox": (0.5, 0.5, 0.1, 0.1)}]) for i in range(50)]
        background = [(f"b{i}.jpg", []) for i in range(50)]
        config = {"class_sampling": {"vest": 5.0, "helmet": 1.0, builder.BACKGROUND: 0}}
        picked = builder.select_images(rare + common + background, ["helmet", "vest"],
                                       config, 0.2, seed=0)
        self.assertEqual(len(picked), 30)
        self.assertFalse(any(name.startswith("b") for name in picked))
        self.assertGreater(sum(n.startswith("r") for n in picked),
                           sum(n.startswith("c") for n in picked))

    def test_assign_prompts_distinct_then_repeats_with_new_seed(self):
        prompts = [{"name": "a", "seed": 10}, {"name": "b", "seed": 20}]
        two = builder.assign_prompts(prompts, 2, random.Random(0))
        self.assertEqual(sorted(p["name"] for p, _ in two), ["a", "b"])
        three = builder.assign_prompts(prompts, 3, random.Random(0))
        self.assertEqual(len({(p["name"], s) for p, s in three}), 3)
        self.assertEqual(three[2][1], three[0][1] + 1)


class TempDatasetMixin:
    """A temp BASE_DIR laid out like production, with a small labelled dataset."""

    def make_dataset(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root = self.base / "data" / "source"
        override = override_settings(BASE_DIR=str(self.base))
        override.enable()
        self.addCleanup(override.disable)
        fs = FleetSettings.load()
        fs.source_dir = str(self.root)
        fs.save()

        ds = self.root / "ppe"
        (ds / "images").mkdir(parents=True)
        (ds / "labels").mkdir()
        (ds / "classes.txt").write_text("# tools: bbox\nhelmet\nvest\n")
        for i in range(6):
            cv2.imwrite(str(ds / "images" / f"f{i}.jpg"), draw_object(textured(i), BOX),
                        [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if i < 5:  # f5 has no label file: unlabeled, never augmented
                (ds / "labels" / f"f{i}.txt").write_text(to_yolo(BOX, i % 2))
        self.dataset = Dataset.objects.create(name="ppe", has_labels=True)
        return ds

    def make_session(self, **overrides):
        fields = dict(name="s", dataset=self.dataset, editor="mock", preview_count=2)
        fields.update(overrides)
        session = AugSession.objects.create(**fields)
        session.preview_images = previews.pick_preview_images(session)
        session.save()
        return session

    def make_build(self, prompts, **overrides):
        fields = dict(
            source_dataset=self.dataset, source_labels_dir=str(self.root / "ppe" / "labels"),
            output_name="ppe__genaug", prompts_snapshot=prompts, fraction=1.0,
            variants_per_image=1, editor="mock",
        )
        fields.update(overrides)
        return AugBuild.objects.create(**fields)


def prompt(name, text, seed=42):
    return {"name": name, "text": text, "negative_prompt": "", "seed": seed,
            "num_inference_steps": 40, "true_cfg_scale": 4.0}


@mock.patch("genaug.services.backend.edit", side_effect=fake_edit)
class BuilderTests(TempDatasetMixin, TestCase):
    def setUp(self):
        self.make_dataset()

    def test_accepted_variants_get_the_source_labels_and_a_manifest(self, edit):
        build = self.make_build([prompt("night", "make it dark")])
        jobs.run_build(build.pk)
        build.refresh_from_db()

        self.assertEqual(build.status, Status.OK)
        self.assertEqual((build.images_total, build.images_selected), (6, 5))
        self.assertEqual((build.variants_attempted, build.variants_accepted), (5, 5))
        out = self.root / "ppe__genaug"
        images = sorted(p.name for p in (out / "images").iterdir())
        self.assertIn("f0.jpg", images)  # originals included
        self.assertIn("f0__mock_night__seed42.jpg", images)
        self.assertNotIn("f5__mock_night__seed42.jpg", images)  # unlabeled: not augmented
        self.assertEqual((out / "labels" / "f0__mock_night__seed42.txt").read_text(),
                         (self.root / "ppe" / "labels" / "f0.txt").read_text())
        self.assertEqual((out / "classes.txt").read_text(),
                         (self.root / "ppe" / "classes.txt").read_text())
        # Originals are hard links, not copies.
        self.assertEqual((out / "images" / "f0.jpg").stat().st_ino,
                         (self.root / "ppe" / "images" / "f0.jpg").stat().st_ino)
        written = cv2.imread(str(out / "images" / "f0__mock_night__seed42.jpg"))
        self.assertEqual(written.shape[:2], (H, W))

        records = [json.loads(line) for line in
                   (out / builder.MANIFEST).read_text().splitlines()]
        self.assertEqual(len(records), 5)
        record = records[0]
        for key in ("source", "source_label", "generated", "augmentation", "prompt", "seed",
                    "editor", "params", "validation", "accepted", "timestamp", "build_id"):
            self.assertIn(key, record)
        self.assertEqual(record["split"], "train")
        self.assertTrue(json.loads((out / builder.BUILD_CONFIG).read_text())["prompts"])

        registered = Dataset.objects.get(name="ppe__genaug")
        self.assertEqual(build.output_dataset, registered)
        self.assertTrue(registered.has_labels)

    def test_rejected_variants_are_logged_but_not_written(self, edit):
        build = self.make_build([prompt("reframe", "shift it")])
        jobs.run_build(build.pk)
        build.refresh_from_db()
        self.assertEqual((build.variants_accepted, build.variants_rejected), (0, 5))
        out = self.root / "ppe__genaug"
        self.assertFalse(any("__mock_" in p.name for p in (out / "images").iterdir()))
        records = [json.loads(l) for l in (out / builder.MANIFEST).read_text().splitlines()]
        self.assertTrue(all(not r["accepted"] and r["validation"]["reason"] for r in records))

    def test_one_failing_generation_does_not_stop_the_build(self, edit):
        build = self.make_build([prompt("night", "make it dark"), prompt("bad", "boom")],
                                variants_per_image=2)
        jobs.run_build(build.pk)
        build.refresh_from_db()
        self.assertEqual(build.status, Status.OK)
        self.assertEqual((build.variants_attempted, build.variants_accepted,
                          build.variants_errored), (10, 5, 5))
        errors = [json.loads(l) for l in
                  (self.root / "ppe__genaug" / builder.MANIFEST).read_text().splitlines()
                  if "error" in json.loads(l)]
        self.assertTrue(all("out of memory" in r["error"] for r in errors))

    def test_second_build_reuses_the_cache_unless_forced(self, edit):
        jobs.run_build(self.make_build([prompt("night", "make it dark")]).pk)
        self.assertEqual(edit.call_count, 5)
        again = self.make_build([prompt("night", "make it dark")], output_name="ppe__genaug2")
        jobs.run_build(again.pk)
        again.refresh_from_db()
        self.assertEqual(edit.call_count, 5)
        self.assertEqual(again.variants_cached, 5)
        forced = self.make_build([prompt("night", "make it dark")], output_name="ppe__g3",
                                 force=True)
        jobs.run_build(forced.pk)
        self.assertEqual(edit.call_count, 10)

    def test_class_sampling_zero_excludes_a_class(self, edit):
        # f0,f2,f4 are helmet (class 0); f1,f3 are vest.
        build = self.make_build([prompt("night", "make it dark")],
                                class_sampling={"helmet": 0.0})
        jobs.run_build(build.pk)
        build.refresh_from_db()
        sources_used = {json.loads(l)["source"] for l in
                        (self.root / "ppe__genaug" / builder.MANIFEST).read_text().splitlines()}
        self.assertTrue(sources_used <= {"f1.jpg", "f3.jpg"})

    def test_gpu_busy_waits_then_continues(self, edit):
        calls = {"n": 0}

        def busy_once(**kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise backend.GpuBusy("only 2 GiB free")
            return fake_edit(**kwargs)

        edit.side_effect = busy_once
        build = self.make_build([prompt("night", "make it dark")])
        with mock.patch("genaug.services.generate.time.sleep") as sleep:
            jobs.run_build(build.pk)
        build.refresh_from_db()
        sleep.assert_called_once()
        self.assertEqual(build.variants_accepted, 5)
        self.assertEqual(build.waiting_reason, "")


@mock.patch("genaug.services.backend.edit", side_effect=fake_edit)
class PreviewTests(TempDatasetMixin, TestCase):
    def setUp(self):
        self.make_dataset()
        self.session = self.make_session()

    def test_preview_images_are_labelled_images_with_boxes(self, edit):
        self.assertEqual(len(self.session.preview_images), 2)
        self.assertNotIn("f5.jpg", self.session.preview_images)

    def test_run_prompt_fills_every_preview_and_revalidate_rejudges(self, edit):
        p = previews.create_prompt(self.session, name="vanish", text="vanish the object")
        jobs.run_prompt_previews(p.pk)
        p.refresh_from_db()
        self.assertEqual(p.status, Status.OK)
        rows = list(p.previews.all())
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r.status == Status.OK and r.output_relpath for r in rows))
        self.assertTrue(all(r.accepted is False for r in rows))

        self.session.min_box_similarity = -1.0
        self.session.max_bbox_drift = 10
        self.session.save()
        self.assertEqual(previews.revalidate(self.session), 2)
        self.assertTrue(all(r.accepted for r in AugPreview.objects.filter(prompt=p)))
        edit.assert_called()  # generation happened once …
        self.assertEqual(edit.call_count, 2)  # … and re-checking did not call it again

    def test_preview_errors_stay_per_image(self, edit):
        p = previews.create_prompt(self.session, name="bad", text="boom")
        jobs.run_prompt_previews(p.pk)
        p.refresh_from_db()
        self.assertEqual(p.status, Status.ERROR)
        self.assertIn("out of memory", p.last_error)


@mock.patch("genaug.services.backend.edit", side_effect=fake_edit)
class StudioAdminTests(TempDatasetMixin, TestCase):
    def setUp(self):
        self.make_dataset()
        self.session = self.make_session()
        user = get_user_model().objects.create_superuser("admin", "a@example.com", "pw")
        self.client.force_login(user)
        queue = mock.patch("genaug.admin._queue")
        self.queue = queue.start().return_value
        self.addCleanup(queue.stop)

    def studio(self):
        return reverse("admin:genaug_augsession_studio") + f"?session={self.session.pk}"

    def test_presets_are_seeded(self, edit):
        self.assertTrue(AugPrompt.objects.filter(name="night_cctv", builtin=True).exists())

    def test_studio_generate_state_and_image(self, edit):
        self.assertEqual(self.client.get(self.studio()).status_code, 200)
        resp = self.client.post(reverse("admin:genaug_augsession_generate"), {
            "session": self.session.pk, "name": "Night CCTV!", "text": "make it dark",
            "seed": "7", "num_inference_steps": "40", "true_cfg_scale": "4",
            "editor": "mock", "transformer": "q4_k_m",
        })
        self.assertEqual(resp.status_code, 302)
        p = self.session.prompts.get()
        self.assertEqual((p.name, p.seed, p.editor_config["editor"]), ("night_cctv", 7, "mock"))
        self.queue.enqueue.assert_called_once_with(
            jobs.run_prompt_previews, p.pk, job_timeout=jobs.PREVIEW_JOB_TIMEOUT)

        jobs.run_prompt_previews(p.pk)
        state = self.client.get(reverse("admin:genaug_augsession_state")
                                + f"?session={self.session.pk}").json()
        pv = state["prompts"][0]["previews"][0]
        self.assertTrue(pv["accepted"])
        image = self.client.get(pv["url"] + "&boxes=1")
        self.assertEqual(image.status_code, 200)
        self.assertEqual(image["Content-Type"], "image/jpeg")
        self.assertIn(b"night_cctv", self.client.get(self.studio()).content)

    def test_keep_reroll_save_and_build(self, edit):
        p = previews.create_prompt(self.session, name="night", text="make it dark",
                                   editor_config={"editor": "mock", "transformer": "q4_k_m",
                                                  "lightning": False})
        url = reverse("admin:genaug_augsession_prompt")
        self.client.post(url, {"prompt": p.pk, "action": "reroll"})
        self.assertEqual(sorted(self.session.prompts.values_list("seed", flat=True)), [42, 43])
        self.client.post(url, {"prompt": p.pk, "action": "save_preset", "preset_name": "my night"})
        self.assertTrue(AugPrompt.objects.filter(name="my_night", text="make it dark").exists())
        self.client.post(url, {"prompt": p.pk, "action": "drop"})
        p.refresh_from_db()
        self.assertFalse(p.keep)

        resp = self.client.post(reverse("admin:genaug_augsession_build"), {
            "session": self.session.pk, "output_name": "ppe_aug", "fraction": "0.5",
            "variants_per_image": "2", "seed": "1", "class__vest": "3",
            "include_originals": "on",
        })
        build = AugBuild.objects.get()
        self.assertRedirects(resp, reverse("admin:genaug_augbuild_report") + f"?build={build.pk}",
                             fetch_redirect_response=False)
        self.assertEqual(len(build.prompts_snapshot), 1)  # only the kept re-roll
        self.assertEqual(build.class_sampling, {"vest": 3.0})
        self.assertEqual((build.fraction, build.variants_per_image), (0.5, 2))
        self.queue.enqueue.assert_called_with(jobs.run_build, build.pk,
                                              job_timeout=jobs.BUILD_JOB_TIMEOUT)

        jobs.run_build(build.pk)
        report = self.client.get(reverse("admin:genaug_augbuild_report") + f"?build={build.pk}")
        self.assertEqual(report.status_code, 200)
        self.assertIn(b"Acceptance rate", report.content)

    def test_build_refuses_the_source_name_and_foreign_dirs(self, edit):
        previews.create_prompt(self.session, name="night", text="make it dark")
        url = reverse("admin:genaug_augsession_build")
        self.client.post(url, {"session": self.session.pk, "output_name": "ppe"})
        (self.root / "someone_elses").mkdir()
        self.client.post(url, {"session": self.session.pk, "output_name": "someone_elses"})
        self.client.post(url, {"session": self.session.pk, "output_name": "../escape"})
        self.assertFalse(AugBuild.objects.exists())


class LeakageGuardTests(TempDatasetMixin, TestCase):
    def test_training_on_a_build_while_validating_on_its_source_is_refused(self):
        from training.models import Experiment, ExperimentDataset, ExperimentModel
        from training.services import config_gen

        self.make_dataset()
        out = Dataset.objects.create(name="ppe__genaug")
        self.make_build([prompt("n", "dark")], output_dataset=out)
        exp = Experiment.objects.create(name="e")
        ExperimentModel.objects.create(experiment=exp, arch=ExperimentModel.YOLOX, num_classes=2)
        ExperimentDataset.objects.create(experiment=exp, dataset=out, role=ExperimentDataset.TRAIN)
        val = ExperimentDataset.objects.create(experiment=exp, dataset=self.dataset,
                                               role=ExperimentDataset.VAL)
        with mock.patch.object(config_gen, "dataset_entry", return_value={}):
            with self.assertRaisesMessage(ValueError, "evaluated on images it trained on"):
                config_gen.build_experiment_dict(exp, self.base / "out")
            val.delete()
            other = Dataset.objects.create(name="kenzo")
            ExperimentDataset.objects.create(experiment=exp, dataset=other,
                                             role=ExperimentDataset.VAL)
            config_gen.build_experiment_dict(exp, self.base / "out")  # unrelated val: fine
