"""Tests for the Marketing Studio section.

Two things are being proven here. First, that the section is genuinely its own:
its own tables, its own video root, its own bundle root, its own presets, and
in particular its own output directory — the render loop is shared code, and
two apps writing into one folder would collide on both output names and the
per-job scratch frame. Second, that the wizard still behaves the way the
marketing action in ``videos`` does, minus the model-source step.

``VideosRegressionTests`` at the bottom lives here rather than in
``videos/tests.py`` so that suite stays untouched: it pins the defaults of the
``root=`` parameter this app added to ``videos.services.inference``.
"""

import json
import subprocess
import tempfile
from pathlib import Path
from unittest import mock

from django.contrib.admin.helpers import ACTION_CHECKBOX_NAME
from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from fleet.models import FleetSettings
from marketing_studio import jobs
from marketing_studio.models import Render, RenderPreset, Video
from marketing_studio.services import paths
from marketing_studio.services.library import list_video_files
from training.models import TrainingSettings
from training.services import bundles
from videos.services import inference

User = get_user_model()


class StudioBase(TestCase):
    """A studio whose two roots are throwaway directories, and a logged-in staff
    user. Mirrors ``videos.tests.VideosBase``, over this app's settings."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.videos_root = self.root / "videos"
        self.bundles_root = self.root / "bundles"
        self.videos_root.mkdir()
        self.bundles_root.mkdir()

        fs = FleetSettings.load()
        # Absolute -> used verbatim by the path helpers.
        fs.marketing_videos_dir = str(self.videos_root)
        fs.marketing_bundles_dir = str(self.bundles_root)
        fs.save()

        User.objects.create_superuser("admin", "a@b.co", "pw")
        self.client.login(username="admin", password="pw")

    def tearDown(self):
        self.tmp.cleanup()

    def _touch(self, name: str) -> Path:
        p = self.videos_root / name
        p.write_bytes(b"fake-mp4")
        return p

    def _video(self, name="clip", filename="clip.mp4") -> Video:
        self._touch(filename)
        return Video.objects.create(name=name, filename=filename, status=Video.READY)

    def _bundle(self, name="ppe-bundle", root: Path | None = None) -> Path:
        """A complete bundle under ``root`` (the studio's bundle root by default)."""
        from training.tests_bundles import MANIFEST

        bundle = (root or self.bundles_root) / name
        (bundle / "models").mkdir(parents=True)
        (bundle / "pipeline.json").write_text(json.dumps(MANIFEST))
        (bundle / "models/model.engine").write_bytes(b"x")
        (bundle / "models/detector.engine").write_bytes(b"x")
        (bundle / "runtime").mkdir()
        (bundle / "infer.py").write_text("")
        return bundle


# ---------------------------------------------------------------- isolation


class IsolationTests(StudioBase):
    """The whole reason this app has its own tables rather than proxies."""

    def test_video_path_is_under_the_studio_root(self):
        video = self._video()
        self.assertEqual(video.path(), self.videos_root / "clip.mp4")
        self.assertTrue(video.exists())

    def test_scanner_reads_the_studio_root_only(self):
        self._touch("a.mp4")
        (FleetSettings.load(), None)  # no-op, keeps the intent explicit below
        from videos.services.videos import list_video_files as videos_scanner

        self.assertEqual(list_video_files(), ["a.mp4"])
        # The detection Videos tab's root is a different (unset, default) dir.
        self.assertNotIn("a.mp4", videos_scanner())

    def test_the_two_apps_hold_independent_rows(self):
        from videos.models import Video as DetectionVideo

        self._video(name="clip", filename="clip.mp4")
        DetectionVideo.objects.create(
            name="clip", filename="clip.mp4", status=DetectionVideo.READY)
        # Same name and filename, different tables: neither unique constraint
        # sees the other.
        self.assertEqual(Video.objects.count(), 1)
        self.assertEqual(DetectionVideo.objects.count(), 1)

    def test_output_dirs_do_not_collide(self):
        """The load-bearing separation: same filename, different absolute path.

        ``unique_output_filename`` only collision-checks the directory it is
        given, and ``run_inference_on_video``'s scratch frame is named after the
        job pk — which two tables share. One shared folder would corrupt both.
        """
        studio = inference.output_dir(paths.output_root())
        detection = inference.output_dir()
        self.assertNotEqual(studio, detection)
        self.assertEqual(studio, self.videos_root / "inferred")

        name_a = inference.unique_output_filename("clip", root=paths.output_root())
        name_b = inference.unique_output_filename("clip")
        self.assertEqual(name_a, name_b)  # same name...
        self.assertNotEqual(studio / name_a, detection / name_b)  # ...different file

    def test_unique_output_filename_only_avoids_its_own_directory(self):
        studio = inference.output_dir(paths.output_root())
        (studio / "clip_inferred.mp4").write_bytes(b"x")
        self.assertEqual(
            inference.unique_output_filename("clip", root=paths.output_root()),
            "clip_inferred_2.mp4")

    def test_presets_are_this_app_s_own(self):
        from videos.models import RenderPreset as DetectionPreset

        RenderPreset.objects.create(name="house", style={"palette": "neon"})
        self.assertFalse(DetectionPreset.objects.exists())
        # The same name is legal on both sides.
        DetectionPreset.objects.create(name="house", style={"palette": "vivid"})
        self.assertEqual(RenderPreset.objects.get().style["palette"], "neon")


class BundleSettingsShimTests(StudioBase):
    """``paths.bundle_settings()`` retargets bundle lookups without disturbing
    the training app's own view of the setting."""

    def test_bundles_root_is_the_studio_s(self):
        self.assertEqual(paths.bundles_root(), self.bundles_root)
        self.assertNotEqual(paths.bundles_root(), bundles.bundles_root())

    def test_it_delegates_every_other_attribute(self):
        ts = TrainingSettings.load()
        shim = paths.bundle_settings()
        self.assertEqual(shim.bundles_root, str(self.bundles_root))
        # The trainer is the same trainer — anything but the root comes from the
        # real singleton, which is what keeps validate(load_test=True) working.
        self.assertEqual(shim.service_base_url, ts.service_base_url)

    def test_it_does_not_write_back_to_training_settings(self):
        paths.bundle_settings()
        TrainingSettings.load().refresh_from_db()
        self.assertNotEqual(TrainingSettings.load().bundles_root,
                            str(self.bundles_root))

    def test_listing_sees_only_the_studio_s_bundles(self):
        self._bundle("mine")
        training_root = self.root / "training-bundles"
        training_root.mkdir()
        self._bundle("theirs", root=training_root)
        ts = TrainingSettings.load()
        ts.bundles_root = str(training_root)
        ts.save()

        studio = [b["relpath"] for b in bundles.list_bundles(paths.bundle_settings())]
        training = [b["relpath"] for b in bundles.list_bundles()]
        self.assertEqual(studio, ["mine"])
        self.assertEqual(training, ["theirs"])

    def test_out_of_root_paths_are_still_rejected(self):
        self._bundle()
        with self.assertRaises(bundles.BundleError):
            bundles.resolve("../../etc", ts=paths.bundle_settings())


# ------------------------------------------------------------------- wizard


class RenderActionTests(StudioBase):
    def setUp(self):
        super().setUp()
        self.video = self._video()
        self.url = reverse("admin:marketing_studio_video_changelist")

    def _post(self, **extra):
        data = {
            "action": "run_marketing_render",
            ACTION_CHECKBOX_NAME: [str(self.video.pk)],
            **extra,
        }
        return self.client.post(self.url, data, follow=True)

    def test_the_action_is_offered(self):
        resp = self.client.get(self.url)
        self.assertContains(resp, "Run model inference for marketing")

    def test_it_opens_straight_on_the_real_form(self):
        """No model-source step: the bundle select and the Look section are on
        the very first page the action renders."""
        self._bundle()
        resp = self._post()
        self.assertContains(resp, 'name="bundle_path"')
        self.assertContains(resp, "Sync bundle")
        self.assertContains(resp, "ppe-bundle")
        self.assertContains(resp, 'name="style_palette"')
        self.assertNotContains(resp, 'name="model_source"')
        # Nothing preselected: the geometry arrives via an explicit sync.
        self.assertContains(resp, 'value="">———')

    def test_the_bundle_select_points_at_this_app_s_sync_endpoint(self):
        self._bundle()
        resp = self._post()
        self.assertContains(
            resp, reverse("admin:marketing_studio_video_bundle_sync"))

    def test_a_training_only_bundle_is_not_offered(self):
        training_root = self.root / "training-bundles"
        training_root.mkdir()
        self._bundle("theirs", root=training_root)
        ts = TrainingSettings.load()
        ts.bundles_root = str(training_root)
        ts.save()
        resp = self._post()
        self.assertNotContains(resp, "theirs")

    @mock.patch("marketing_studio.admin._queue")
    def test_apply_takes_its_geometry_from_the_manifest(self, queue):
        bundle = self._bundle()
        self._post(
            apply="1", bundle_path="ppe-bundle",
            # What a stale page or a hand-edited POST might send for the locked
            # fields — the bundle's own values must win.
            pipeline="raw", detector_expand_ratio="9", merge_nms_iou="0.99",
            detector_checkpoint="/wrong/person.pt",
            score_threshold="0.5", frame_stride="2",
        )
        render = Render.objects.get()
        self.assertEqual(render.bundle_path, "ppe-bundle")
        self.assertEqual(render.pipeline, "people_detect_first")
        self.assertEqual(render.detector_expand_ratio, 0.1)
        self.assertEqual(render.detector_min_box_size, 224.0)
        self.assertEqual(render.merge_nms_iou, 0.3)
        self.assertEqual(render.detector_checkpoint,
                         str(bundle / "models/detector.engine"))
        # The knobs the operator keeps.
        self.assertEqual(render.score_threshold, 0.5)
        self.assertEqual(render.frame_stride, 2)
        self.assertEqual(render.model_checkpoint(), str(bundle / "models/model.engine"))
        queue.return_value.enqueue.assert_called_once()

    @mock.patch("marketing_studio.admin._queue")
    def test_apply_always_stores_a_look(self, queue):
        self._bundle()
        self._post(apply="1", bundle_path="ppe-bundle", score_threshold="0.5",
                   frame_stride="1", style_palette="neon", style_box_style="corners")
        render = Render.objects.get()
        self.assertTrue(render.render_style)
        self.assertEqual(render.render_style["palette"], "neon")
        self.assertEqual(render.render_style["box_style"], "corners")

    @mock.patch("marketing_studio.admin._queue")
    def test_apply_writes_into_the_studio_output_dir(self, queue):
        self._bundle()
        self._post(apply="1", bundle_path="ppe-bundle", score_threshold="0.5",
                   frame_stride="1")
        render = Render.objects.get()
        self.assertEqual(render.output_filename, "clip_inferred.mp4")
        self.assertEqual(render.output_path().parent, self.videos_root / "inferred")

    @mock.patch("marketing_studio.admin._queue")
    def test_apply_without_a_bundle_asks_for_one(self, queue):
        self._bundle()
        resp = self._post(apply="1", bundle_path="", score_threshold="0.5",
                          frame_stride="1")
        self.assertFalse(Render.objects.exists())
        queue.return_value.enqueue.assert_not_called()
        self.assertContains(resp, "Choose a bundle")

    @mock.patch("marketing_studio.admin._queue")
    def test_apply_rejects_a_bundle_that_lives_only_in_the_training_root(self, queue):
        """The regression guard on the shim: a relpath valid over there must not
        resolve here."""
        training_root = self.root / "training-bundles"
        training_root.mkdir()
        self._bundle("theirs", root=training_root)
        ts = TrainingSettings.load()
        ts.bundles_root = str(training_root)
        ts.save()
        resp = self._post(apply="1", bundle_path="theirs", score_threshold="0.5",
                          frame_stride="1")
        self.assertFalse(Render.objects.exists())
        queue.return_value.enqueue.assert_not_called()
        self.assertContains(resp, "not usable")

    @mock.patch("marketing_studio.admin._queue")
    def test_apply_rejects_a_bundle_missing_its_model(self, queue):
        bundle = self._bundle()
        (bundle / "models/model.engine").unlink()
        resp = self._post(apply="1", bundle_path="ppe-bundle", score_threshold="0.5",
                          frame_stride="1")
        self.assertFalse(Render.objects.exists())
        self.assertContains(resp, "is not usable")

    @mock.patch("marketing_studio.admin._queue")
    def test_a_bad_style_field_is_reported_and_nothing_is_queued(self, queue):
        self._bundle()
        resp = self._post(apply="1", bundle_path="ppe-bundle", score_threshold="0.5",
                          frame_stride="1", style_box_opacity="5")
        self.assertFalse(Render.objects.exists())
        queue.return_value.enqueue.assert_not_called()
        self.assertContains(resp, "box_opacity")

    @mock.patch("marketing_studio.admin._queue")
    def test_a_rejected_submit_re_renders_what_was_typed(self, queue):
        self._bundle()
        resp = self._post(apply="1", bundle_path="", score_threshold="0.5",
                          frame_stride="1", style_palette="sunset")
        self.assertEqual(resp.context["style_values"]["style_palette"], "sunset")

    @mock.patch("marketing_studio.admin._queue")
    def test_the_form_arrives_carrying_the_last_look_used_here(self, queue):
        self._bundle()
        Render.objects.create(
            video=self.video, bundle_path="ppe-bundle",
            render_style={"palette": "ocean", "box_style": "corners"})
        resp = self._post()
        self.assertEqual(resp.context["style_values"]["style_palette"], "ocean")

    @mock.patch("marketing_studio.admin._queue")
    def test_the_last_look_does_not_leak_in_from_the_videos_app(self, queue):
        from videos.models import InferenceJob
        from videos.models import Video as DetectionVideo

        other = DetectionVideo.objects.create(
            name="other", filename="other.mp4", status=DetectionVideo.READY)
        InferenceJob.objects.create(
            video=other, model_source=InferenceJob.BUNDLE, bundle_path="x",
            render_style={"palette": "sunset"})
        self._bundle()
        resp = self._post()
        self.assertNotEqual(resp.context["style_values"]["style_palette"], "sunset")

    def test_the_relabel_dropdown_offers_the_bundle_s_classes(self):
        self._bundle()
        resp = self._post(bundle_path="ppe-bundle")
        self.assertContains(resp, "helmet")
        self.assertContains(resp, "vest")


class BundleSyncViewTests(StudioBase):
    def setUp(self):
        super().setUp()
        self.url = reverse("admin:marketing_studio_video_bundle_sync")

    def test_get_is_rejected(self):
        self.assertEqual(self.client.get(self.url).status_code, 405)

    def test_anonymous_is_redirected_to_login(self):
        self.client.logout()
        resp = self.client.post(self.url, {"bundle": "ppe-bundle"})
        self.assertEqual(resp.status_code, 302)
        self.assertIn("/login", resp["Location"])

    def test_no_bundle_is_a_400(self):
        resp = self.client.post(self.url, {"bundle": ""})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("No bundle selected", resp.json()["error"])

    def test_it_validates_against_the_studio_root(self):
        self._bundle()
        resp = self.client.post(self.url, {"bundle": "ppe-bundle"})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["defaults"]["pipeline"], "people_detect_first")

    def test_a_training_only_bundle_does_not_validate_here(self):
        training_root = self.root / "training-bundles"
        training_root.mkdir()
        self._bundle("theirs", root=training_root)
        ts = TrainingSettings.load()
        ts.bundles_root = str(training_root)
        ts.save()
        body = self.client.post(self.url, {"bundle": "theirs"}).json()
        self.assertFalse(body["ok"])


class PreviewViewTests(StudioBase):
    def setUp(self):
        super().setUp()
        self.video = self._video()
        self.url = reverse("admin:marketing_studio_video_preview")

    def test_get_is_rejected(self):
        self.assertEqual(self.client.get(self.url).status_code, 405)

    def test_anonymous_is_redirected_to_login(self):
        self.client.logout()
        resp = self.client.post(self.url, {"video": self.video.pk})
        self.assertEqual(resp.status_code, 302)
        self.assertIn("/login", resp["Location"])

    def test_unknown_video_is_a_404_json(self):
        resp = self.client.post(self.url, {"video": 999999})
        self.assertEqual(resp.status_code, 404)
        self.assertIn("error", resp.json())

    def test_a_bad_style_is_a_json_error_not_a_500(self):
        self._bundle()
        resp = self.client.post(self.url, {
            "video": self.video.pk, "bundle_path": "ppe-bundle",
            "score_threshold": "0.5", "frame_stride": "1",
            "style_box_opacity": "5",
        })
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Box opacity", resp.json()["error"])

    @mock.patch("videos.services.inference.preview_frame")
    def test_a_rendered_frame_comes_back_as_a_data_uri(self, preview):
        preview.return_value = {
            "jpeg": b"\xff\xd8jpeg", "width": 1920, "height": 1080,
            "boxes": [], "detections": 0, "frame_index": 12,
            "frames_total": 100, "cached": False,
        }
        self._bundle()
        resp = self.client.post(self.url, {
            "video": self.video.pk, "bundle_path": "ppe-bundle",
            "score_threshold": "0.5", "frame_stride": "1", "position": "0.25",
        })
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["image"].startswith("data:image/jpeg;base64,"))
        self.assertEqual(resp.json()["frame_index"], 12)

    @mock.patch("videos.services.inference.preview_frame")
    def test_it_previews_against_the_studio_output_root(self, preview):
        """The preview cache lives under the output dir — it must be this app's,
        not the Videos tab's."""
        preview.return_value = {
            "jpeg": b"\xff\xd8", "width": 8, "height": 8, "boxes": [],
            "detections": 0, "frame_index": 0, "frames_total": 1, "cached": True,
        }
        self._bundle()
        self.client.post(self.url, {
            "video": self.video.pk, "bundle_path": "ppe-bundle",
            "score_threshold": "0.5", "frame_stride": "1",
        })
        self.assertEqual(preview.call_args.kwargs["root"], paths.output_root())


class RenderPresetViewTests(StudioBase):
    def setUp(self):
        super().setUp()
        self.url = reverse("admin:marketing_studio_video_render_preset_save")

    def test_get_is_rejected(self):
        self.assertEqual(self.client.get(self.url).status_code, 405)

    def test_anonymous_is_redirected_to_login(self):
        self.client.logout()
        resp = self.client.post(self.url, {"preset_name": "x"})
        self.assertEqual(resp.status_code, 302)

    def test_a_nameless_preset_is_refused(self):
        resp = self.client.post(self.url, {"preset_name": "  "})
        self.assertEqual(resp.status_code, 400)
        self.assertFalse(RenderPreset.objects.exists())

    def test_saving_stores_the_whole_look(self):
        resp = self.client.post(self.url, {
            "preset_name": "house", "style_palette": "neon",
            "style_box_style": "corners",
        })
        self.assertEqual(resp.status_code, 200)
        preset = RenderPreset.objects.get()
        self.assertEqual(preset.name, "house")
        self.assertEqual(preset.style["palette"], "neon")
        self.assertEqual(resp.json()["values"]["style_palette"], "neon")

    def test_the_same_name_upserts(self):
        self.client.post(self.url, {"preset_name": "house", "style_palette": "neon"})
        self.client.post(self.url, {"preset_name": "house", "style_palette": "ocean"})
        self.assertEqual(RenderPreset.objects.count(), 1)
        self.assertEqual(RenderPreset.objects.get().style["palette"], "ocean")

    def test_a_bad_style_is_refused(self):
        resp = self.client.post(self.url, {"preset_name": "x", "style_box_opacity": "5"})
        self.assertEqual(resp.status_code, 400)
        self.assertFalse(RenderPreset.objects.exists())

    def test_a_preset_saved_before_a_knob_existed_still_loads(self):
        RenderPreset.objects.create(name="old", style={"palette": "neon"})
        self.assertEqual(RenderPreset.objects.get().summary(), "rounded · neon")


# ------------------------------------------------------------------- admin


class AdminRenderTests(StudioBase):
    def test_the_section_appears_on_the_index_with_its_three_tabs(self):
        resp = self.client.get(reverse("admin:index"))
        self.assertContains(resp, "Marketing Studio")
        for url in ["admin:marketing_studio_video_changelist",
                    "admin:marketing_studio_render_changelist",
                    "admin:marketing_studio_renderpreset_changelist"]:
            self.assertContains(resp, reverse(url))

    def test_every_changelist_renders(self):
        video = self._video()
        Render.objects.create(video=video, bundle_path="b",
                              render_style={"palette": "neon"})
        RenderPreset.objects.create(name="house", style={"palette": "neon"})
        for url in ["admin:marketing_studio_video_changelist",
                    "admin:marketing_studio_render_changelist",
                    "admin:marketing_studio_renderpreset_changelist"]:
            self.assertEqual(self.client.get(reverse(url)).status_code, 200)

    def test_the_render_log_takes_no_new_rows_by_hand(self):
        resp = self.client.get(reverse("admin:marketing_studio_render_add"))
        self.assertEqual(resp.status_code, 403)

    def test_the_player_page_embeds_the_stream(self):
        video = self._video()
        url = reverse("admin:marketing_studio_video_play") + f"?video={video.pk}"
        resp = self.client.get(url)
        self.assertContains(resp, "<video")
        self.assertContains(resp, reverse("admin:marketing_studio_video_stream"))

    def test_the_url_names_are_this_app_s_own(self):
        """The _url_name discipline: the studio's routes must not collide with
        the videos app's identically-shaped ones."""
        self.assertNotEqual(reverse("admin:marketing_studio_video_play"),
                            reverse("admin:videos_video_play"))
        self.assertNotEqual(reverse("admin:marketing_studio_render_stream"),
                            reverse("admin:videos_inferencejob_stream"))


class CleanupSignalTests(StudioBase):
    def test_deleting_a_video_removes_its_file(self):
        video = self._video()
        path = video.path()
        self.assertTrue(path.is_file())
        video.delete()
        self.assertFalse(path.exists())

    def test_deleting_a_render_removes_its_output(self):
        video = self._video()
        out = inference.output_dir(paths.output_root()) / "clip_inferred.mp4"
        out.write_bytes(b"x")
        render = Render.objects.create(
            video=video, bundle_path="b", output_filename="clip_inferred.mp4")
        render.delete()
        self.assertFalse(out.exists())

    def test_a_cascade_from_the_video_also_cleans_the_render(self):
        video = self._video()
        out = inference.output_dir(paths.output_root()) / "clip_inferred.mp4"
        out.write_bytes(b"x")
        Render.objects.create(
            video=video, bundle_path="b", output_filename="clip_inferred.mp4")
        video.delete()
        self.assertFalse(out.exists())
        self.assertFalse(Render.objects.exists())

    def test_a_missing_file_is_not_an_error(self):
        video = self._video()
        video.path().unlink()
        video.delete()  # must not raise
        render = Render.objects.create(
            video=Video.objects.create(name="b", filename=""), bundle_path="b",
            output_filename="")
        render.delete()  # blank output_filename must not raise either


class RenderEndToEndTests(StudioBase):
    """The whole loop on a real video file: decode, style, encode — with only
    the model mocked (the trainer is a separate container and holds the GPU).

    The point here beyond ``videos``' own copy of this: proving the output
    lands under the *studio's* root, at the studio's own scratch-frame path.
    """

    BOXES = [{"cx": 0.5, "cy": 0.5, "w": 0.3, "h": 0.4, "confidence": 0.9,
              "class_id": 0, "class_name": "helmet"}]

    def setUp(self):
        super().setUp()
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
             "-i", "testsrc=size=640x360:rate=10:duration=1",
             "-pix_fmt", "yuv420p", str(self.videos_root / "clip.mp4")],
            check=True, capture_output=True,
        )
        self.video = Video.objects.create(
            name="clip", filename="clip.mp4", status=Video.READY)
        self.bundle = self._bundle()

    def _render(self, **style):
        return Render(
            video=self.video, bundle_path="ppe-bundle", pipeline="raw",
            render_style=style,
            output_filename=inference.unique_output_filename(
                "clip", root=paths.output_root()),
        )

    @mock.patch("training.services.runner.predict_image")
    def test_it_renders_into_the_studio_root_at_the_delivery_height(self, predict):
        import cv2

        predict.return_value = {"boxes": self.BOXES}
        render = self._render(output_height=180, box_style="corners",
                              counter="top_left", crf=30)
        result = inference.run_inference_on_video(render, root=paths.output_root())

        self.assertEqual(result["frames_total"], 10)
        output = self.videos_root / "inferred" / render.output_filename
        self.assertTrue(output.is_file())
        # And nowhere near the detection app's folder.
        self.assertFalse(
            (inference.output_dir() / render.output_filename).exists())

        capture = cv2.VideoCapture(str(output))
        try:
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), 180)
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 320)
        finally:
            capture.release()

    @mock.patch("training.services.runner.predict_image")
    def test_the_job_wrapper_records_success_on_the_row(self, predict):
        predict.return_value = {"boxes": self.BOXES}
        render = self._render(box_style="corners")
        render.save()
        jobs.run_render(render.pk)
        render.refresh_from_db()
        self.assertEqual(render.status, Render.OK)
        self.assertEqual(render.frames_total, 10)
        self.assertTrue(render.output_exists())

    @mock.patch("training.services.runner.predict_image")
    def test_a_failing_model_is_recorded_on_the_row_and_re_raised(self, predict):
        predict.side_effect = RuntimeError("trainer said no")
        render = self._render(box_style="corners")
        render.save()
        with self.assertRaises(RuntimeError):
            jobs.run_render(render.pk)
        render.refresh_from_db()
        self.assertEqual(render.status, Render.ERROR)
        self.assertIn("trainer said no", render.last_error)

    @mock.patch("training.services.runner.predict_image")
    def test_the_preview_renders_one_frame_and_caches_it(self, predict):
        predict.return_value = {"boxes": self.BOXES}
        render = self._render(box_style="corners")
        first = inference.preview_frame(render, 0.5, root=paths.output_root())
        self.assertFalse(first["cached"])
        self.assertTrue(first["jpeg"].startswith(b"\xff\xd8"))
        second = inference.preview_frame(render, 0.5, root=paths.output_root())
        self.assertTrue(second["cached"])
        self.assertEqual(predict.call_count, 1)
        # The cache is the studio's own, under its output dir.
        self.assertTrue(
            (self.videos_root / "inferred" / ".preview_cache").is_dir())


# ------------------------------------------------------- shared-module guard


class VideosRegressionTests(TestCase):
    """Pins the defaults of the ``root=`` parameter this app added to
    ``videos.services.inference``. Lives here so ``videos/tests.py`` needs no
    edit — that suite passing unchanged is the other half of the proof."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        fs = FleetSettings.load()
        fs.videos_dir = self.tmp.name
        fs.save()

    def tearDown(self):
        self.tmp.cleanup()

    def test_output_dir_still_defaults_to_the_videos_root(self):
        from fleet.services.paths import videos_root

        self.assertEqual(inference.output_dir(), videos_root() / "inferred")

    def test_unique_output_filename_still_defaults_there(self):
        (Path(self.tmp.name) / "inferred").mkdir(exist_ok=True)
        (Path(self.tmp.name) / "inferred" / "clip_inferred.mp4").write_bytes(b"x")
        self.assertEqual(inference.unique_output_filename("clip"),
                         "clip_inferred_2.mp4")
