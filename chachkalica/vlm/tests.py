"""Tests for the VLM section.

The backend container is stubbed everywhere — these cover the parts that must
work without a GPU: the tabs render, a video can be added all three ways, the
actions create runs and redirect to their screens, the polls page correctly on
``seq``, the frame loop samples at the requested rate, and — the part with the
most room to be quietly wrong — a dataset run's answers score against its labels
the way the report claims they do.
"""

import tempfile
from pathlib import Path
from unittest import mock

from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client, TestCase, override_settings
from django.urls import reverse

from fleet.models import Dataset, FleetSettings
from vlm import weights_catalog
from vlm.models import (
    VlmAlert, VlmDatasetResult, VlmDatasetRun, VlmModel, VlmRun, VlmVideo,
)
from vlm.services import dataset_runner, grading


def make_model(**overrides) -> VlmModel:
    defaults = {
        "name": "Test VLM",
        "backend": "transformers",
        "family": "smolvlm",
        "weights": "HuggingFaceTB/SmolVLM-Instruct",
        "prompt": "Describe anything unsafe.",
        "max_new_tokens": 32,
    }
    defaults.update(overrides)
    return VlmModel.objects.create(**defaults)


class WeightsCatalogTests(TestCase):
    def test_every_family_in_the_catalog_has_an_adapter_key(self):
        """The dropdown and the backend registry must agree on family keys.

        They live in different packages (one Django, one the backend image), so
        nothing but this test stops them drifting into a model row that can be
        created but never run.
        """
        catalog_keys = set(weights_catalog.VLM_WEIGHTS_CATALOG)
        choice_keys = {key for key, _label in weights_catalog.FAMILY_CHOICES}
        self.assertEqual(catalog_keys, choice_keys)

    def test_uncached_weights_are_labelled_not_hidden(self):
        with mock.patch.object(weights_catalog, "is_cached", return_value=False):
            options = weights_catalog.weights_options("qwen2_vl")
        labels = [label for _value, label in options]
        self.assertTrue(any("not in hf_cache" in label for label in labels))

    def test_cache_dir_name_matches_huggingface_layout(self):
        self.assertEqual(
            weights_catalog.cache_dir_name("Qwen/Qwen2.5-VL-3B-Instruct"),
            "models--Qwen--Qwen2.5-VL-3B-Instruct",
        )


class AdminSectionTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_superuser("admin", "a@b.c", "pw")
        self.client = Client()
        self.client.force_login(self.user)

    def test_both_tabs_appear_under_a_vlm_section(self):
        resp = self.client.get(reverse("admin:index"))
        self.assertContains(resp, "VLM")
        self.assertContains(resp, reverse("admin:vlm_vlmmodel_changelist"))
        self.assertContains(resp, reverse("admin:vlm_vlmvideo_changelist"))

    def test_model_changelist_renders(self):
        make_model()
        resp = self.client.get(reverse("admin:vlm_vlmmodel_changelist"))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "Test VLM")

    def test_adding_a_model_resolves_the_weights_dropdown(self):
        with mock.patch("vlm.forms.weights_catalog.is_cached", return_value=True):
            resp = self.client.post(reverse("admin:vlm_vlmmodel_add"), {
                "name": "Qwen watcher",
                "description": "",
                "backend": "transformers",
                "family": "qwen2_vl",
                "weights_choice": "Qwen/Qwen2.5-VL-3B-Instruct",
                "custom_weights": "",
                "variant": "",
                "quantization": "",
                "prompt": "Describe the scene.",
                "max_new_tokens": "64",
            })
        self.assertEqual(resp.status_code, 302, resp.content[:3000])
        model = VlmModel.objects.get(name="Qwen watcher")
        self.assertEqual(model.weights, "Qwen/Qwen2.5-VL-3B-Instruct")
        # The catalogue entry's variant is adopted, not asked for.
        self.assertEqual(model.variant, "3B")

    def test_adding_a_model_with_uncached_weights_is_refused(self):
        with mock.patch("vlm.forms.weights_catalog.is_cached", return_value=False):
            resp = self.client.post(reverse("admin:vlm_vlmmodel_add"), {
                "name": "Missing weights",
                "description": "",
                "backend": "transformers",
                "family": "qwen2_vl",
                "weights_choice": "Qwen/Qwen2.5-VL-7B-Instruct",
                "custom_weights": "",
                "variant": "",
                "quantization": "",
                "prompt": "Describe the scene.",
                "max_new_tokens": "64",
            })
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "not in data/hf_cache")
        self.assertFalse(VlmModel.objects.filter(name="Missing weights").exists())

    def test_video_add_form_offers_all_three_sources(self):
        resp = self.client.get(reverse("admin:vlm_vlmvideo_add"))
        self.assertEqual(resp.status_code, 200)
        for field in ("upload", "import_file", "source_url"):
            self.assertContains(resp, f'name="{field}"')


class VideoUploadTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_superuser("admin", "a@b.c", "pw")
        self.client = Client()
        self.client.force_login(self.user)

    def test_uploading_an_mp4_writes_it_under_the_vlm_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch("vlm.services.videos.vlm_videos_root", return_value=root), \
                 mock.patch("vlm.models.vlm_videos_root", return_value=root), \
                 mock.patch("vlm.services.videos.probe", return_value={
                     "duration_seconds": 3.0, "source_fps": 25.0,
                     "width": 640, "height": 480,
                 }):
                upload = SimpleUploadedFile(
                    "clip.mp4", b"\x00\x00\x00\x18ftypmp42fake", content_type="video/mp4",
                )
                resp = self.client.post(reverse("admin:vlm_vlmvideo_add"), {
                    "upload": upload, "import_file": "", "source_url": "",
                    "quality": "", "name": "",
                })
                self.assertEqual(resp.status_code, 302, resp.content[:2000])

                video = VlmVideo.objects.get()
                self.assertEqual(video.status, VlmVideo.READY)
                self.assertEqual(video.filename, "clip.mp4")
                self.assertTrue((root / "clip.mp4").is_file())
                # The probe result is what turns "1 fps" into a frame stride.
                self.assertEqual(video.source_fps, 25.0)

    def test_two_sources_at_once_is_rejected(self):
        resp = self.client.post(reverse("admin:vlm_vlmvideo_add"), {
            "upload": "", "import_file": "", "source_url": "http://example.com/v.mp4",
            "quality": "", "name": "x",
        })
        # One source is fine; adding a second must not be.
        self.assertEqual(resp.status_code, 302)
        VlmVideo.objects.all().delete()

        upload = SimpleUploadedFile("clip.mp4", b"fake", content_type="video/mp4")
        resp = self.client.post(reverse("admin:vlm_vlmvideo_add"), {
            "upload": upload, "import_file": "",
            "source_url": "http://example.com/v.mp4", "quality": "", "name": "y",
        })
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "exactly one of")


class RunActionTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_superuser("admin", "a@b.c", "pw")
        self.client = Client()
        self.client.force_login(self.user)
        self.model = make_model()
        self.video = VlmVideo.objects.create(
            name="clip", filename="clip.mp4", status=VlmVideo.READY, source_fps=25.0,
        )

    def _post_action(self, data):
        return self.client.post(
            reverse("admin:vlm_vlmvideo_changelist"),
            {"action": "run_vlm_inference", "_selected_action": [str(self.video.pk)],
             **data},
        )

    def test_action_creates_a_run_and_redirects_to_the_live_screen(self):
        with mock.patch.object(VlmVideo, "exists", return_value=True), \
             mock.patch("vlm.weights_catalog.is_cached", return_value=True), \
             mock.patch("vlm.admin.is_cached", return_value=True), \
             mock.patch("vlm.admin._queue") as queue:
            resp = self._post_action({"apply": "1", "fps": "1.0",
                                      "vlm_model": str(self.model.pk)})

        run = VlmRun.objects.get()
        self.assertRedirects(
            resp,
            reverse("admin:vlm_vlmvideo_run_live") + f"?run={run.pk}",
            fetch_redirect_response=False,
        )
        self.assertEqual(run.fps, 1.0)
        self.assertEqual(run.status, VlmRun.QUEUED)
        queue.return_value.enqueue.assert_called_once()

    def test_the_run_snapshots_the_prompt_it_started_with(self):
        """Editing the model afterwards must not rewrite history."""
        with mock.patch.object(VlmVideo, "exists", return_value=True), \
             mock.patch("vlm.admin.is_cached", return_value=True), \
             mock.patch("vlm.admin._queue"):
            self._post_action({"apply": "1", "fps": "1.0",
                               "vlm_model": str(self.model.pk)})

        self.model.prompt = "something else entirely"
        self.model.save()

        run = VlmRun.objects.get()
        self.assertEqual(run.prompt_snapshot, "Describe anything unsafe.")
        self.assertEqual(run.model_config_snapshot["weights"],
                         "HuggingFaceTB/SmolVLM-Instruct")

    def test_uncached_weights_block_the_run(self):
        with mock.patch.object(VlmVideo, "exists", return_value=True), \
             mock.patch("vlm.admin.is_cached", return_value=False), \
             mock.patch("vlm.admin._queue"):
            resp = self._post_action({"apply": "1", "fps": "1.0",
                                      "vlm_model": str(self.model.pk)})
        self.assertEqual(VlmRun.objects.count(), 0)
        self.assertEqual(resp.status_code, 200)

    def test_sampling_faster_than_the_video_is_rejected(self):
        with mock.patch.object(VlmVideo, "exists", return_value=True), \
             mock.patch("vlm.admin.is_cached", return_value=True), \
             mock.patch("vlm.admin._queue"):
            self._post_action({"apply": "1", "fps": "60",
                               "vlm_model": str(self.model.pk)})
        self.assertEqual(VlmRun.objects.count(), 0)


class AlertPollTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_superuser("admin", "a@b.c", "pw")
        self.client = Client()
        self.client.force_login(self.user)
        video = VlmVideo.objects.create(name="clip", filename="clip.mp4")
        self.run = VlmRun.objects.create(
            video=video, fps=1.0, status=VlmRun.RUNNING, model_label_snapshot="m",
        )
        for seq in range(1, 4):
            VlmAlert.objects.create(
                run=self.run, seq=seq, frame_index=seq * 25,
                video_timestamp_seconds=float(seq), text=f"alert {seq}",
            )

    def test_poll_returns_only_alerts_after_the_cursor(self):
        with mock.patch("vlm.admin.heartbeat.mark_viewed") as marked:
            resp = self.client.get(
                reverse("admin:vlm_vlmvideo_run_alerts"),
                {"run": self.run.pk, "after": 1},
            )
        data = resp.json()
        self.assertEqual([a["seq"] for a in data["alerts"]], [2, 3])
        self.assertEqual(data["run"]["status"], "running")
        # The poll doubles as the viewer heartbeat; without it the worker pauses.
        marked.assert_called_once_with(self.run.pk)

    def test_poll_requires_a_logged_in_staff_user(self):
        anon = Client()
        resp = anon.get(reverse("admin:vlm_vlmvideo_run_alerts"), {"run": self.run.pk})
        self.assertEqual(resp.status_code, 302)
        self.assertIn("login", resp["Location"])

    def test_live_screen_renders_existing_alerts_for_a_finished_run(self):
        self.run.status = VlmRun.OK
        self.run.save()
        resp = self.client.get(reverse("admin:vlm_vlmvideo_run_live"),
                               {"run": self.run.pk})
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "alert 3")

    def test_cancel_marks_the_run_and_rejects_get(self):
        url = reverse("admin:vlm_vlmvideo_run_cancel") + f"?run={self.run.pk}"
        self.assertEqual(self.client.get(url).status_code, 404)

        resp = self.client.post(url)
        self.assertEqual(resp.status_code, 200)
        self.run.refresh_from_db()
        self.assertEqual(self.run.status, VlmRun.CANCEL_REQUESTED)


class RunnerTests(TestCase):
    """The frame loop, with cv2 and the backend both faked."""

    def setUp(self):
        video = VlmVideo.objects.create(
            name="clip", filename="clip.mp4", status=VlmVideo.READY, source_fps=10.0,
        )
        self.run = VlmRun.objects.create(
            video=video, fps=2.0, status=VlmRun.RUNNING,
            model_config_snapshot={"backend": "transformers", "family": "smolvlm",
                                   "weights": "w", "max_new_tokens": 8},
            prompt_snapshot="what do you see?",
            model_label_snapshot="m",
        )

    def _run_with_fake_capture(self, frames: int, pos_msec_broken: bool = False,
                               viewed: bool = True):
        """Drive the loop over ``frames`` fake frames."""
        import sys
        import types

        state = {"index": 0}

        class FakeCapture:
            def isOpened(self):
                return True

            def get(self, prop):
                if prop == "FPS":
                    return 10.0
                if prop == "FRAME_COUNT":
                    return float(frames)
                if prop == "POS_MSEC":
                    # The position of the frame read() will return next, which is
                    # how cv2 actually behaves — hence 100ms per frame at 10 fps.
                    # Returning 0.0 instead reproduces the codecs that report
                    # nothing, where the runner must fall back to the frame index
                    # rather than stamping every alert at t=0.
                    return 0.0 if pos_msec_broken else state["index"] * 100.0
                return 0.0

            def read(self):
                if state["index"] >= frames:
                    return False, None
                state["index"] += 1
                return True, object()

            def release(self):
                pass

        fake_cv2 = types.SimpleNamespace(
            CAP_PROP_FPS="FPS", CAP_PROP_FRAME_COUNT="FRAME_COUNT",
            CAP_PROP_POS_MSEC="POS_MSEC",
            VideoCapture=lambda _path: FakeCapture(),
            imwrite=lambda _path, _frame: True,
        )

        with mock.patch.dict(sys.modules, {"cv2": fake_cv2}), \
             mock.patch.object(VlmVideo, "exists", return_value=True), \
             mock.patch("vlm.services.runner.backend.infer",
                        return_value={"text": "a person", "latency_ms": 12}) as infer, \
             mock.patch("vlm.services.runner.heartbeat.recently_viewed",
                        return_value=viewed):
            from vlm.services import runner

            result = runner.run_vlm_on_video(self.run)
        return result, infer

    def test_samples_at_the_requested_rate(self):
        # 10 fps source, 2 fps requested → every 5th frame → 4 of 20.
        result, infer = self._run_with_fake_capture(frames=20)
        self.run.refresh_from_db()
        self.assertEqual(self.run.frame_stride, 5)
        self.assertEqual(infer.call_count, 4)
        self.assertEqual(result["alerts"], 4)
        self.assertEqual(VlmAlert.objects.filter(run=self.run).count(), 4)
        self.assertEqual(
            list(VlmAlert.objects.filter(run=self.run).values_list("seq", flat=True)),
            [1, 2, 3, 4],
        )

    def test_timestamps_come_from_the_container_when_it_reports_them(self):
        """The stamp is the sampled frame's own position, not the next frame's."""
        self._run_with_fake_capture(frames=20)
        stamps = list(VlmAlert.objects.filter(run=self.run)
                      .values_list("video_timestamp_seconds", flat=True))
        self.assertEqual(stamps, [0.0, 0.5, 1.0, 1.5])

    def test_timestamps_fall_back_when_the_container_reports_none(self):
        self._run_with_fake_capture(frames=20, pos_msec_broken=True)
        stamps = list(VlmAlert.objects.filter(run=self.run)
                      .values_list("video_timestamp_seconds", flat=True))
        # frame_index / source_fps for frames 0, 5, 10, 15 at 10 fps.
        self.assertEqual(stamps, [0.0, 0.5, 1.0, 1.5])

    def test_an_unwatched_run_stops_instead_of_holding_a_worker(self):
        """Nobody watching for long enough ends the run, keeping what it produced."""
        from vlm.services import runner

        with mock.patch.object(runner, "VIEWER_GRACE_SECONDS", 0.0), \
             mock.patch.object(runner, "ABANDONED_AFTER_SECONDS", 0.0), \
             mock.patch.object(runner, "IDLE_POLL_SECONDS", 0.0):
            result, infer = self._run_with_fake_capture(frames=20, viewed=False)

        # It stopped after the first frame, and that frame's alert survived.
        self.assertTrue(result["cancelled"])
        self.assertEqual(infer.call_count, 1)
        self.assertEqual(VlmAlert.objects.filter(run=self.run).count(), 1)

    def test_cancellation_stops_after_the_current_frame(self):
        import sys
        import types

        state = {"index": 0}

        class FakeCapture:
            def isOpened(self):
                return True

            def get(self, prop):
                return {"FPS": 10.0, "FRAME_COUNT": 100.0}.get(prop, 0.0)

            def read(self):
                state["index"] += 1
                return True, object()  # never ends on its own

            def release(self):
                pass

        fake_cv2 = types.SimpleNamespace(
            CAP_PROP_FPS="FPS", CAP_PROP_FRAME_COUNT="FRAME_COUNT",
            CAP_PROP_POS_MSEC="POS_MSEC",
            VideoCapture=lambda _path: FakeCapture(),
            imwrite=lambda _path, _frame: True,
        )

        run = self.run

        def cancel_after_first(*_args, **_kwargs):
            VlmRun.objects.filter(pk=run.pk).update(status=VlmRun.CANCEL_REQUESTED)
            return {"text": "seen", "latency_ms": 1}

        with mock.patch.dict(sys.modules, {"cv2": fake_cv2}), \
             mock.patch.object(VlmVideo, "exists", return_value=True), \
             mock.patch("vlm.services.runner.backend.infer",
                        side_effect=cancel_after_first), \
             mock.patch("vlm.services.runner.heartbeat.recently_viewed",
                        return_value=True):
            from vlm.services import runner

            result = runner.run_vlm_on_video(run)

        self.assertTrue(result["cancelled"])
        self.assertEqual(result["alerts"], 1)


# ---------------------------------------------------------------------------
# Dataset runs
# ---------------------------------------------------------------------------

def write_dataset(root: Path, name: str = "ppe", *, classes=("helmet", "vest"),
                  images=("a.jpg", "b.jpg", "c.jpg"), labels=None) -> Path:
    """Build a minimal labeled dataset on disk and return its directory.

    Images are placeholder bytes: nothing in the graded path decodes them (the
    backend is always stubbed here), so their content is irrelevant and their
    names are not.
    """
    dataset_dir = root / name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    for image in images:
        (dataset_dir / image).write_bytes(b"not-really-a-jpeg")
    if labels:
        labels_dir = dataset_dir / "labels"
        labels_dir.mkdir(exist_ok=True)
        for image, text in labels.items():
            (labels_dir / f"{Path(image).stem}.txt").write_text(text, encoding="utf-8")
    return dataset_dir


class Scored:
    """A stand-in for a stored result row, for the metric aggregation tests."""

    def __init__(self, gt, predicted, has_label=True):
        self.gt_classes = gt
        self.predicted_classes = predicted
        self.has_label = has_label


class GradingTests(TestCase):
    """The pure half: text in, class names out, then the aggregation."""

    CLASSES = ["helmet", "vest"]

    def _predict(self, text, alias_map=None, negation=True):
        matchers = grading.compile_matchers(self.CLASSES, alias_map)
        return grading.predict_classes(text, matchers, negation=negation)

    def test_a_class_name_in_the_answer_counts_as_claimed(self):
        self.assertEqual(self._predict("A worker in a helmet."), ["helmet"])

    def test_separators_and_plurals_are_flexible(self):
        matchers = grading.compile_matchers(["hard_hat"], None)
        for text in ("wearing a hard hat", "two hard-hats", "a hardhat"):
            self.assertEqual(
                grading.predict_classes(text, matchers), ["hard_hat"], text,
            )

    def test_a_substring_of_a_longer_word_is_not_a_match(self):
        # "vested" is not a vest, and this is why the patterns are bounded.
        self.assertEqual(self._predict("the interest is vested elsewhere"), [])

    def test_aliases_carry_the_words_the_model_actually_uses(self):
        self.assertEqual(self._predict("wearing a hi-vis jacket"), [])
        self.assertEqual(
            self._predict("wearing a hi-vis jacket", {"vest": ["hi-vis jacket"]}),
            ["vest"],
        )

    def test_a_negated_mention_is_not_a_claim(self):
        """The failure this exists to prevent: "no helmet" scoring as a helmet."""
        self.assertEqual(self._predict("No, the worker is not wearing a helmet."), [])
        # ...and the toggle really does turn the heuristic off.
        self.assertEqual(
            self._predict("No, the worker is not wearing a helmet.", negation=False),
            ["helmet"],
        )

    def test_negation_does_not_reach_across_a_clause(self):
        self.assertEqual(
            self._predict("There is no vest, but he does wear a helmet."), ["helmet"],
        )

    def test_one_unnegated_mention_is_enough(self):
        self.assertEqual(
            self._predict("No helmet at first; later he puts a helmet on."), ["helmet"],
        )

    def test_ground_truth_reduces_a_yolo_file_to_its_class_names(self):
        text = "1 0.5 0.5 0.2 0.2\n0 0.1 0.1 0.1 0.1\n1 0.9 0.9 0.05 0.05\n"
        self.assertEqual(
            grading.ground_truth_classes(text, self.CLASSES), ["vest", "helmet"],
        )

    def test_an_out_of_range_class_id_is_dropped_not_raised(self):
        self.assertEqual(
            grading.ground_truth_classes("7 0.5 0.5 0.1 0.1\n", self.CLASSES), [],
        )

    def test_an_empty_label_file_means_nothing_is_present(self):
        self.assertEqual(grading.ground_truth_classes("", self.CLASSES), [])

    def test_metrics_score_each_class_and_the_exact_match(self):
        results = [
            Scored(["helmet"], ["helmet"]),                    # exact
            Scored(["helmet", "vest"], ["helmet"]),            # missed the vest
            Scored([], ["vest"]),                              # claimed one that isn't
            Scored(["vest"], ["vest"], has_label=False),       # unlabeled: not scored
        ]
        metrics = grading.metrics(results, self.CLASSES)

        self.assertEqual(metrics["graded_images"], 3)
        self.assertEqual(metrics["unlabeled_images"], 1)
        self.assertEqual(metrics["exact_matches"], 1)
        self.assertAlmostEqual(metrics["exact_match_accuracy"], 1 / 3, places=3)

        by_name = {row["name"]: row for row in metrics["per_class"]}
        self.assertEqual(
            (by_name["helmet"]["tp"], by_name["helmet"]["fp"], by_name["helmet"]["fn"]),
            (2, 0, 0),
        )
        self.assertEqual(
            (by_name["vest"]["tp"], by_name["vest"]["fp"], by_name["vest"]["fn"]),
            (0, 1, 1),
        )

    def test_a_class_nobody_mentions_is_left_out_of_the_macro_average(self):
        """Otherwise an unused class in classes.txt would score every run zero."""
        metrics = grading.metrics([Scored(["helmet"], ["helmet"])], self.CLASSES)
        by_name = {row["name"]: row for row in metrics["per_class"]}
        self.assertIsNone(by_name["vest"]["f1"])
        self.assertEqual(metrics["macro"]["f1"], 1.0)

    def test_the_confusion_matrix_covers_single_class_images_only(self):
        results = [
            Scored(["helmet"], ["helmet"]),
            Scored(["helmet"], ["vest"]),
            Scored(["helmet"], []),
            Scored(["helmet"], ["helmet", "vest"]),
            Scored(["helmet", "vest"], ["helmet"]),  # two labeled classes: excluded
        ]
        confusion = grading.metrics(results, self.CLASSES)["confusion"]
        self.assertEqual(confusion["total"], 4)
        self.assertEqual(confusion["columns"], ["helmet", "vest", "(none)", "(multiple)"])
        row = next(r for r in confusion["rows"] if r["actual"] == "helmet")
        self.assertEqual(row["counts"], [1, 1, 1, 1])


class DatasetImageSelectionTests(TestCase):
    def test_a_capped_run_spans_the_dataset(self):
        """A cap must not silently turn into "the first N files"."""
        with tempfile.TemporaryDirectory() as tmp:
            names = [f"img{i:03d}.jpg" for i in range(10)]
            dataset_dir = write_dataset(Path(tmp), images=names)
            picked = dataset_runner.select_images(dataset_dir, 5)
            self.assertEqual([p.name for p in picked],
                             ["img000.jpg", "img002.jpg", "img004.jpg",
                              "img006.jpg", "img008.jpg"])

    def test_no_cap_means_every_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = write_dataset(Path(tmp))
            self.assertEqual(len(dataset_runner.select_images(dataset_dir, 0)), 3)


class DatasetRunnerTests(TestCase):
    """The image loop, with the backend faked and a dataset on a temp disk."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        # Lay the temp tree out the way production is — the dataset under
        # BASE_DIR/data — so the default path through the loop is the real one:
        # images handed to the backend by path, on the mount it shares.
        self.base = Path(self.tmp.name)
        self.root = self.base / "data" / "source"
        self.root.mkdir(parents=True)
        override = override_settings(BASE_DIR=str(self.base))
        override.enable()
        self.addCleanup(override.disable)

        settings_row = FleetSettings.load()
        settings_row.source_dir = str(self.root)
        settings_row.save()

        self.dataset_dir = write_dataset(
            self.root, labels={
                "a.jpg": "0 0.5 0.5 0.2 0.2\n",          # helmet
                "b.jpg": "1 0.5 0.5 0.2 0.2\n",          # vest
                # c.jpg deliberately has no label file: unlabeled, not empty.
            },
        )
        self.dataset = Dataset.objects.create(name="ppe")
        self.model = make_model()

    def _make_run(self, **overrides):
        defaults = {
            "dataset": self.dataset,
            "dataset_name_snapshot": "ppe",
            "vlm_model": self.model,
            "model_config_snapshot": self.model.config(),
            "prompt_snapshot": "What PPE is worn?",
            "model_label_snapshot": self.model.name,
            "grade": True,
            "classes_snapshot": ["helmet", "vest"],
            "alias_map": {"helmet": [], "vest": []},
            "status": VlmDatasetRun.RUNNING,
        }
        defaults.update(overrides)
        return VlmDatasetRun.objects.create(**defaults)

    def _answers(self, *texts):
        """Stub the backend with one answer per image, in order."""
        return mock.patch(
            "vlm.services.dataset_runner.backend.infer",
            side_effect=[{"text": t, "latency_ms": 5} for t in texts],
        )

    def test_every_image_gets_a_row_scored_against_its_label_file(self):
        run = self._make_run()
        with self._answers("A helmet.", "No vest here.", "A helmet and a vest."):
            result = dataset_runner.run_vlm_on_dataset(run)

        self.assertFalse(result["cancelled"])
        rows = list(run.results.order_by("seq"))
        self.assertEqual([r.image_filename for r in rows],
                         ["a.jpg", "b.jpg", "c.jpg"])
        # a.jpg: labeled helmet, answered helmet → success.
        self.assertTrue(rows[0].matched)
        # b.jpg: labeled vest, answered "no vest" → the negation pass makes this
        # a miss rather than a spurious success.
        self.assertEqual(rows[1].predicted_classes, [])
        self.assertFalse(rows[1].matched)
        # c.jpg: no label file, so it is answered but never scored.
        self.assertTrue(rows[2].has_label is False)
        self.assertIsNone(rows[2].matched)

        run.refresh_from_db()
        self.assertEqual(run.metrics["graded_images"], 2)
        self.assertEqual(run.metrics["unlabeled_images"], 1)
        self.assertEqual(run.metrics["exact_matches"], 1)
        self.assertEqual(run.metrics["exact_match_accuracy"], 0.5)

    def test_a_capped_run_only_asks_about_that_many_images(self):
        run = self._make_run(image_limit=2)
        with self._answers("A helmet.", "A vest.") as infer:
            dataset_runner.run_vlm_on_dataset(run)
        self.assertEqual(infer.call_count, 2)
        run.refresh_from_db()
        self.assertEqual(run.images_total, 2)
        self.assertEqual(run.images_processed, 2)

    def test_one_failed_image_is_recorded_and_the_run_carries_on(self):
        run = self._make_run()
        with mock.patch(
            "vlm.services.dataset_runner.backend.infer",
            side_effect=[
                {"text": "A helmet.", "latency_ms": 5},
                RuntimeError("backend hiccup"),
                {"text": "A vest.", "latency_ms": 5},
            ],
        ):
            dataset_runner.run_vlm_on_dataset(run)

        run.refresh_from_db()
        self.assertEqual(run.images_processed, 3)
        self.assertEqual(run.errors_count, 1)
        failed = run.results.get(seq=2)
        self.assertIn("backend hiccup", failed.error)
        # A failed call is a gap in the measurement, not a wrong answer: it must
        # not be scored as "the model named no classes".
        self.assertIsNone(failed.matched)
        self.assertEqual(run.metrics["graded_images"], 1)

    def test_a_wedged_backend_aborts_instead_of_writing_thousands_of_failures(self):
        run = self._make_run()
        with mock.patch.object(dataset_runner, "MAX_CONSECUTIVE_ERRORS", 2), \
             mock.patch("vlm.services.dataset_runner.backend.infer",
                        side_effect=RuntimeError("connection refused")):
            with self.assertRaises(RuntimeError):
                dataset_runner.run_vlm_on_dataset(run)
        self.assertEqual(run.results.count(), 2)

    def test_cancellation_stops_after_the_current_image(self):
        run = self._make_run()

        def cancel_after_first(*_args, **_kwargs):
            VlmDatasetRun.objects.filter(pk=run.pk).update(
                status=VlmDatasetRun.CANCEL_REQUESTED)
            return {"text": "A helmet.", "latency_ms": 1}

        with mock.patch("vlm.services.dataset_runner.backend.infer",
                        side_effect=cancel_after_first):
            result = dataset_runner.run_vlm_on_dataset(run)

        self.assertTrue(result["cancelled"])
        self.assertEqual(run.results.count(), 1)
        # What it did produce is still scored.
        run.refresh_from_db()
        self.assertEqual(run.metrics["graded_images"], 1)

    def test_a_run_failing_on_every_image_is_still_stoppable(self):
        """The cancel check has to sit on the error path too, not just the happy one."""
        run = self._make_run()

        def fail_then_cancel(*_args, **_kwargs):
            VlmDatasetRun.objects.filter(pk=run.pk).update(
                status=VlmDatasetRun.CANCEL_REQUESTED)
            raise RuntimeError("backend down")

        with mock.patch("vlm.services.dataset_runner.backend.infer",
                        side_effect=fail_then_cancel):
            result = dataset_runner.run_vlm_on_dataset(run)

        self.assertTrue(result["cancelled"])
        self.assertEqual(run.results.count(), 1)

    def test_an_ungraded_run_stores_answers_and_no_verdicts(self):
        run = self._make_run(grade=False, classes_snapshot=[], alias_map={})
        with self._answers("one", "two", "three"):
            dataset_runner.run_vlm_on_dataset(run)
        run.refresh_from_db()
        self.assertEqual(run.metrics, {})
        self.assertEqual([r.matched for r in run.results.all()], [None, None, None])
        # The ground truth is still recorded, so the run can be graded later.
        self.assertEqual(run.results.get(seq=1).gt_classes, [])

    def test_regrading_rescores_stored_answers_without_calling_the_backend(self):
        """The point of storing raw text: better aliases cost no GPU."""
        run = self._make_run()
        with self._answers("A hard hat.", "A hi-vis jacket.", "nothing"):
            dataset_runner.run_vlm_on_dataset(run)
        run.refresh_from_db()
        self.assertEqual(run.metrics["exact_matches"], 0)

        with mock.patch("vlm.services.dataset_runner.backend.infer") as infer:
            metrics = dataset_runner.regrade(
                run,
                alias_map={"helmet": ["hard hat"], "vest": ["hi-vis jacket"]},
                negation=True,
            )
        infer.assert_not_called()
        self.assertEqual(metrics["exact_matches"], 2)
        self.assertEqual(metrics["exact_match_accuracy"], 1.0)

    def test_an_image_on_the_shared_mount_is_handed_over_by_path(self):
        """No copy in the normal case — the backend opens the very same file."""
        run = self._make_run(image_limit=1)
        with self._answers("A helmet.") as infer:
            dataset_runner.run_vlm_on_dataset(run)
        handed_over = Path(infer.call_args[0][0])
        self.assertEqual(handed_over, self.dataset_dir / "a.jpg")

    def test_a_dataset_outside_the_shared_mount_is_staged_instead(self):
        """vlm-backend can only open paths on the shared data/ mount.

        ``source_dir`` may be an absolute path anywhere, and a path the backend
        cannot see would otherwise fail on every single image.
        """
        outside = Path(self.tmp.name) / "elsewhere"
        write_dataset(outside, name="ppe")
        settings_row = FleetSettings.load()
        settings_row.source_dir = str(outside)
        settings_row.save()

        run = self._make_run(image_limit=1)
        with self._answers("A helmet.") as infer:
            dataset_runner.run_vlm_on_dataset(run)

        handed_over = Path(infer.call_args[0][0])
        self.assertEqual(handed_over.parent, dataset_runner.scratch_dir())
        # ...and the scratch copy does not outlive the run.
        self.assertFalse(handed_over.exists())


class DatasetRunActionTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_superuser("admin", "a@b.c", "pw")
        self.client = Client()
        self.client.force_login(self.user)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        settings_row = FleetSettings.load()
        settings_row.source_dir = str(self.root)
        settings_row.save()
        self.model = make_model()

    def _post(self, data):
        return self.client.post(
            reverse("admin:vlm_vlmmodel_changelist"),
            {"action": "run_on_dataset", "_selected_action": [str(self.model.pk)],
             **data},
        )

    def test_the_action_creates_a_run_and_redirects_to_its_report(self):
        write_dataset(self.root, labels={"a.jpg": "0 0.5 0.5 0.2 0.2\n"})
        dataset = Dataset.objects.create(name="ppe")

        with mock.patch("vlm.admin.is_cached", return_value=True), \
             mock.patch("vlm.admin._queue") as queue:
            resp = self._post({"apply": "1", "dataset": str(dataset.pk),
                               "image_limit": "0", "grade": "on",
                               "negation_aware": "on"})

        run = VlmDatasetRun.objects.get()
        self.assertRedirects(
            resp, reverse("admin:vlm_vlmdatasetrun_report") + f"?run={run.pk}",
            fetch_redirect_response=False,
        )
        self.assertTrue(run.grade)
        self.assertEqual(run.status, VlmDatasetRun.QUEUED)
        # The classes.txt is snapshotted, so editing it later cannot rewrite
        # what this run was scored against.
        self.assertEqual(run.classes_snapshot, ["helmet", "vest"])
        self.assertEqual(run.alias_map, {"helmet": [], "vest": []})
        self.assertEqual(run.prompt_snapshot, self.model.prompt)
        queue.return_value.enqueue.assert_called_once()

    def test_grading_an_unlabeled_dataset_is_refused(self):
        write_dataset(self.root, name="raw")
        dataset = Dataset.objects.create(name="raw")

        with mock.patch("vlm.admin.is_cached", return_value=True), \
             mock.patch("vlm.admin._queue"):
            resp = self._post({"apply": "1", "dataset": str(dataset.pk),
                               "image_limit": "0", "grade": "on"})

        self.assertEqual(VlmDatasetRun.objects.count(), 0)
        self.assertContains(resp, "nothing to grade against")

    def test_the_same_dataset_runs_fine_ungraded(self):
        write_dataset(self.root, name="raw")
        dataset = Dataset.objects.create(name="raw")

        with mock.patch("vlm.admin.is_cached", return_value=True), \
             mock.patch("vlm.admin._queue"):
            self._post({"apply": "1", "dataset": str(dataset.pk),
                        "image_limit": "0"})

        run = VlmDatasetRun.objects.get()
        self.assertFalse(run.grade)

    def test_uncached_weights_block_the_run(self):
        write_dataset(self.root)
        dataset = Dataset.objects.create(name="ppe")
        with mock.patch("vlm.admin.is_cached", return_value=False), \
             mock.patch("vlm.admin._queue"):
            resp = self._post({"apply": "1", "dataset": str(dataset.pk),
                               "image_limit": "0"})
        self.assertEqual(VlmDatasetRun.objects.count(), 0)
        self.assertContains(resp, "not in data/hf_cache")

    def test_a_dataset_with_no_images_is_refused(self):
        (self.root / "empty").mkdir()
        dataset = Dataset.objects.create(name="empty")
        with mock.patch("vlm.admin.is_cached", return_value=True), \
             mock.patch("vlm.admin._queue"):
            resp = self._post({"apply": "1", "dataset": str(dataset.pk),
                               "image_limit": "0"})
        self.assertEqual(VlmDatasetRun.objects.count(), 0)
        self.assertContains(resp, "no images under")


class DatasetReportTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_superuser("admin", "a@b.c", "pw")
        self.client = Client()
        self.client.force_login(self.user)
        self.run = VlmDatasetRun.objects.create(
            dataset_name_snapshot="ppe",
            model_label_snapshot="Test VLM",
            prompt_snapshot="What PPE is worn?",
            grade=True,
            classes_snapshot=["helmet", "vest"],
            alias_map={"helmet": [], "vest": []},
            status=VlmDatasetRun.RUNNING,
            images_total=2,
            images_processed=2,
        )
        VlmDatasetResult.objects.create(
            run=self.run, seq=1, image_filename="a.jpg", text="A hard hat.",
            has_label=True, gt_classes=["helmet"], predicted_classes=[], matched=False,
        )
        VlmDatasetResult.objects.create(
            run=self.run, seq=2, image_filename="b.jpg", text="A vest.",
            has_label=True, gt_classes=["vest"], predicted_classes=["vest"], matched=True,
        )

    def test_the_dataset_runs_tab_is_registered(self):
        resp = self.client.get(reverse("admin:vlm_vlmdatasetrun_changelist"))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "ppe")

    def test_the_report_renders_answers_and_scores(self):
        self.run.metrics = grading.metrics(
            list(self.run.results.all()), self.run.classes_snapshot,
        )
        self.run.save()
        resp = self.client.get(reverse("admin:vlm_vlmdatasetrun_report"),
                               {"run": self.run.pk})
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "A hard hat.")
        self.assertContains(resp, "0.500")  # exact-match accuracy tile

    def test_the_mismatch_filter_shows_only_the_failures(self):
        resp = self.client.get(reverse("admin:vlm_vlmdatasetrun_report"),
                               {"run": self.run.pk, "only": "mismatches"})
        self.assertContains(resp, "A hard hat.")
        self.assertNotContains(resp, "A vest.")

    def test_the_progress_poll_pages_on_the_seq_cursor(self):
        resp = self.client.get(reverse("admin:vlm_vlmdatasetrun_progress"),
                               {"run": self.run.pk, "after": 1})
        data = resp.json()
        self.assertEqual([r["seq"] for r in data["results"]], [2])
        self.assertEqual(data["run"]["status"], "running")

    def test_the_progress_poll_requires_a_logged_in_staff_user(self):
        anon = Client()
        resp = anon.get(reverse("admin:vlm_vlmdatasetrun_progress"),
                        {"run": self.run.pk})
        self.assertEqual(resp.status_code, 302)
        self.assertIn("login", resp["Location"])

    def test_regrading_from_the_report_rewrites_the_verdicts(self):
        url = reverse("admin:vlm_vlmdatasetrun_regrade") + f"?run={self.run.pk}"
        self.assertEqual(self.client.get(url).status_code, 404)

        resp = self.client.post(url, {"alias__helmet": "hard hat",
                                      "alias__vest": "", "negation_aware": "on"})
        self.assertEqual(resp.status_code, 302)

        self.run.refresh_from_db()
        self.assertEqual(self.run.alias_map["helmet"], ["hard hat"])
        self.assertEqual(self.run.metrics["exact_match_accuracy"], 1.0)
        self.assertTrue(self.run.results.get(seq=1).matched)

    def test_cancel_is_post_only_and_marks_the_run(self):
        url = reverse("admin:vlm_vlmdatasetrun_cancel") + f"?run={self.run.pk}"
        self.assertEqual(self.client.get(url).status_code, 404)

        resp = self.client.post(url)
        self.assertEqual(resp.status_code, 302)
        self.run.refresh_from_db()
        self.assertEqual(self.run.status, VlmDatasetRun.CANCEL_REQUESTED)

    def test_a_thumbnail_cannot_escape_its_dataset(self):
        """The filename is ours, but the containment check is what makes it safe."""
        result = self.run.results.get(seq=1)
        result.image_filename = "../../../etc/passwd"
        result.save()
        resp = self.client.get(reverse("admin:vlm_vlmdatasetrun_image"),
                               {"result": result.pk})
        self.assertEqual(resp.status_code, 404)


class LabelledImageTests(TestCase):
    """Clicking a row's image opens it with the label file's own boxes on it."""

    def setUp(self):
        self.user = User.objects.create_superuser("admin", "a@b.c", "pw")
        self.client = Client()
        self.client.force_login(self.user)

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        settings_row = FleetSettings.load()
        settings_row.source_dir = str(self.root)
        settings_row.save()

        # A real decodable image, unlike write_dataset's placeholder bytes —
        # the renderer actually opens this one.
        self.dataset_dir = self.root / "ppe"
        self.dataset_dir.mkdir()
        (self.dataset_dir / "classes.txt").write_text("helmet\nvest\n", encoding="utf-8")
        self._write_image(self.dataset_dir / "a.jpg")
        self._write_image(self.dataset_dir / "b.jpg")
        labels_dir = self.dataset_dir / "labels"
        labels_dir.mkdir()
        (labels_dir / "a.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
        # b.jpg has no label file at all: unlabeled, nothing to draw.

        self.run = VlmDatasetRun.objects.create(
            dataset_name_snapshot="ppe", model_label_snapshot="m",
            grade=True, classes_snapshot=["helmet", "vest"],
            status=VlmDatasetRun.OK,
        )
        self.labeled = VlmDatasetResult.objects.create(
            run=self.run, seq=1, image_filename="a.jpg", text="A helmet.",
            has_label=True, gt_classes=["helmet"],
            predicted_classes=["helmet"], matched=True,
        )
        self.unlabeled = VlmDatasetResult.objects.create(
            run=self.run, seq=2, image_filename="b.jpg", text="Nothing.",
            has_label=False,
        )

    @staticmethod
    def _write_image(path: Path) -> None:
        import cv2
        import numpy as np

        cv2.imwrite(str(path), np.full((80, 120, 3), 255, dtype=np.uint8))

    def _get_image(self, result, **params):
        return self.client.get(reverse("admin:vlm_vlmdatasetrun_image"),
                               {"result": result.pk, **params})

    def test_without_labels_the_raw_file_is_served(self):
        resp = self._get_image(self.labeled)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(b"".join(resp.streaming_content),
                         (self.dataset_dir / "a.jpg").read_bytes())

    def test_with_labels_the_boxes_are_drawn_on(self):
        import cv2
        import numpy as np

        resp = self._get_image(self.labeled, labels="1")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp["Content-Type"], "image/jpeg")

        rendered = cv2.imdecode(
            np.frombuffer(resp.content, dtype=np.uint8), cv2.IMREAD_COLOR)
        # Same geometry as the source, but no longer a blank white frame.
        self.assertEqual(rendered.shape[:2], (80, 120))
        self.assertTrue((rendered < 200).any(), "nothing was drawn on the image")

    def test_an_unlabeled_image_falls_back_to_the_plain_file(self):
        """Nothing annotated is not an error — it is the honest answer."""
        resp = self._get_image(self.unlabeled, labels="1")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(b"".join(resp.streaming_content),
                         (self.dataset_dir / "b.jpg").read_bytes())

    def test_a_file_that_is_not_an_image_is_a_404_not_a_500(self):
        (self.dataset_dir / "a.jpg").write_bytes(b"definitely not a jpeg")
        resp = self._get_image(self.labeled, labels="1")
        self.assertEqual(resp.status_code, 404)

    def test_a_polygon_label_is_drawn_too(self):
        (self.dataset_dir / "labels" / "a.txt").write_text(
            "1 0.1 0.1 0.9 0.2 0.5 0.8\n", encoding="utf-8")
        resp = self._get_image(self.labeled, labels="1")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp["Content-Type"], "image/jpeg")

    def test_the_report_links_its_thumbnails_at_the_annotated_copy(self):
        resp = self.client.get(reverse("admin:vlm_vlmdatasetrun_report"),
                               {"run": self.run.pk})
        image_url = reverse("admin:vlm_vlmdatasetrun_image")
        self.assertContains(resp, f"{image_url}?result={self.labeled.pk}&amp;labels=1")
