"""Tests for :mod:`training.services.bundles` — bundles as a model source.

A bundle is the one model source that can arrive from *outside* this app (scp a
directory in and it is offered), so the tests lean on the two things that follow
from that: nothing about a manifest may be trusted without checking, and every
value the manifest carries must reach the inference row unchanged, since the
bundle — not the form — is the record of what runs.
"""

import json
import tempfile
from pathlib import Path
from unittest import mock

from django.test import TestCase

from training import pipelines
from training.models import TrainingSettings
from training.services import bundles

# A manifest shaped like the real thing (chachak.bundle_export.manifest builds
# these): flat pipeline name, nested tiling/detector, thresholds under `defaults`.
MANIFEST = {
    "schema_version": 1,
    "bundle": {
        "name": "ppe",
        "created_utc": "2026-07-31T10:58:47Z",
        "default_format": "engine",
        "exporter": "chachak.bundle_export",
    },
    "pipeline": "people_detect_first",
    "artifacts": {
        "model": "models/model.engine",
        "detector": "models/detector.engine",
        "extra_models": [],
    },
    "defaults": {"conf": 0.4, "detector_conf": 0.5},
    "name": "ppe",
    "classes": {"0": "helmet", "1": "vest"},
    "merge_nms_iou": 0.3,
    "tiling": {"tile_size_px": None, "tile_width_pct": 50.0,
               "tile_height_pct": 50.0, "overlap": 0.2},
    "detector": {"expand_ratio": 0.1, "min_box_size": 224.0, "nms_iou": 0.5},
    "chain": [],
}


class BundleTestCase(TestCase):
    """Points ``bundles_root`` at a tmpdir and can write bundles into it."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

        ts = TrainingSettings.load()
        ts.bundles_root = str(self.root)
        ts.save()

    def write_bundle(self, name="ppe-bundle", manifest=None, *, model=True,
                     detector=True, runtime=True, meta=True) -> Path:
        """A bundle on disk, with each part independently omittable.

        Every ``False`` here corresponds to a real way a copied bundle arrives
        broken — an interrupted scp, a bundle assembled by hand, an engine-only
        copy that left ``models/`` behind.
        """
        bundle = self.root / name
        (bundle / "models").mkdir(parents=True)
        blob = manifest if manifest is not None else MANIFEST
        (bundle / bundles.MANIFEST_NAME).write_text(json.dumps(blob))
        if model:
            (bundle / "models/model.engine").write_bytes(b"engine")
            if meta:
                (bundle / "models/model.meta.json").write_text(json.dumps({
                    "arch": "rfdetr", "class_map": {"0": "helmet", "1": "vest"},
                    "input": {"size": 960},
                }))
        if detector:
            (bundle / "models/detector.engine").write_bytes(b"engine")
            if meta:
                (bundle / "models/detector.meta.json").write_text(
                    json.dumps({"arch": "yolox", "class_map": {"0": "person"}}))
        if runtime:
            (bundle / "runtime").mkdir()
            (bundle / "infer.py").write_text("")
        return bundle


class ListBundlesTests(BundleTestCase):
    def test_a_directory_with_a_manifest_is_a_bundle(self):
        self.write_bundle()

        found = bundles.list_bundles()

        self.assertEqual([b["relpath"] for b in found], ["ppe-bundle"])
        self.assertEqual(found[0]["pipeline"], "people_detect_first")
        self.assertEqual(found[0]["kind"], "TensorRT")
        self.assertEqual(found[0]["classes"], ["helmet", "vest"])
        self.assertTrue(found[0]["ok"])

    def test_a_directory_without_a_manifest_is_not(self):
        (self.root / "just-a-folder" / "models").mkdir(parents=True)
        (self.root / "just-a-folder/models/model.onnx").write_bytes(b"")

        self.assertEqual(bundles.list_bundles(), [])

    def test_nested_bundles_are_found(self):
        # Operators group them; a whole tree can also be copied in at once.
        self.write_bundle("site-a/ppe-bundle")

        self.assertEqual(
            [b["relpath"] for b in bundles.list_bundles()], ["site-a/ppe-bundle"])

    def test_a_bundle_missing_its_model_is_listed_but_flagged(self):
        # Hiding it would leave the operator with a directory that is visibly on
        # disk and inexplicably absent from the dropdown.
        self.write_bundle(model=False)

        found = bundles.list_bundles()

        self.assertEqual(len(found), 1)
        self.assertFalse(found[0]["ok"])

    def test_an_unparseable_manifest_is_listed_but_flagged(self):
        bundle = self.write_bundle()
        (bundle / bundles.MANIFEST_NAME).write_text("{ not json")

        found = bundles.list_bundles()

        self.assertEqual(len(found), 1)
        self.assertFalse(found[0]["ok"])

    def test_a_root_that_is_itself_a_bundle_is_selectable(self):
        # An operator can point the setting straight at one bundle; whatever
        # list_bundles offers has to round-trip through resolve().
        ts = TrainingSettings.load()
        ts.bundles_root = str(self.write_bundle())
        ts.save()

        found = bundles.list_bundles()

        self.assertEqual(len(found), 1)
        self.assertTrue(bundles.model_artifact(found[0]["relpath"]).is_file())

    def test_a_missing_root_is_empty_not_an_error(self):
        ts = TrainingSettings.load()
        ts.bundles_root = str(self.root / "does-not-exist")
        ts.save()

        self.assertEqual(bundles.list_bundles(), [])


class ResolveTests(BundleTestCase):
    def test_resolves_a_bundle_directory(self):
        bundle = self.write_bundle()

        self.assertEqual(bundles.resolve("ppe-bundle"), bundle)

    def test_rejects_a_path_escaping_the_root(self):
        with self.assertRaises(bundles.BundleError):
            bundles.resolve("../../etc")

    def test_rejects_a_blank_selection(self):
        with self.assertRaises(bundles.BundleError):
            bundles.resolve("")

    def test_rejects_a_directory_that_is_not_a_bundle(self):
        (self.root / "plain").mkdir()

        with self.assertRaises(bundles.BundleError):
            bundles.resolve("plain")

    def test_rejects_a_newer_manifest_schema(self):
        # Reading a newer bundle would mean silently dropping fields it carries.
        self.write_bundle(manifest={**MANIFEST, "schema_version": 99})

        with self.assertRaises(bundles.BundleError):
            bundles.read_manifest("ppe-bundle")

    def test_model_artifact_must_stay_inside_the_bundle(self):
        self.write_bundle(manifest={
            **MANIFEST, "artifacts": {**MANIFEST["artifacts"], "model": "../../evil.engine"}})

        with self.assertRaises(bundles.BundleError):
            bundles.model_artifact("ppe-bundle")

    def test_model_artifact_must_be_a_runnable_suffix(self):
        bundle = self.write_bundle(manifest={
            **MANIFEST, "artifacts": {**MANIFEST["artifacts"], "model": "models/model.pt"}})
        (bundle / "models/model.pt").write_bytes(b"")

        with self.assertRaises(bundles.BundleError):
            bundles.model_artifact("ppe-bundle")


class PipelineDefaultsTests(BundleTestCase):
    """The manifest's vocabulary translated into the one every consumer reads."""

    def test_geometry_comes_across_whole(self):
        self.write_bundle()

        defaults = bundles.pipeline_defaults("ppe-bundle")

        self.assertEqual(defaults["pipeline"], pipelines.PEOPLE_DETECT_FIRST)
        self.assertEqual(defaults["detector_expand_ratio"], 0.1)
        self.assertEqual(defaults["detector_min_box_size"], 224.0)
        self.assertEqual(defaults["tile_width_pct"], 50.0)
        self.assertEqual(defaults["tile_height_pct"], 50.0)
        self.assertEqual(defaults["overlap"], 0.2)
        self.assertEqual(defaults["merge_nms_iou"], 0.3)
        self.assertEqual(defaults["chain"], [])

    def test_the_detector_is_the_bundled_one_not_the_exporter_s(self):
        # The whole point of a bundle: its detector travels with it, so the path
        # must be the copy that is actually here.
        bundle = self.write_bundle(manifest={
            **MANIFEST,
            "provenance": {"models": {"detector": {"checkpoint": "/other/machine/best.pt"}}},
        })

        defaults = bundles.pipeline_defaults("ppe-bundle")

        self.assertEqual(defaults["detector_checkpoint"],
                         str(bundle / "models/detector.engine"))

    def test_shipped_conf_becomes_the_score_threshold(self):
        self.write_bundle()

        self.assertEqual(bundles.pipeline_defaults("ppe-bundle")["score_threshold"], 0.4)

    def test_an_older_manifest_without_defaults_falls_back(self):
        manifest = {k: v for k, v in MANIFEST.items() if k != "defaults"}
        self.write_bundle(manifest={**manifest, "score_threshold": 0.25})

        self.assertEqual(bundles.pipeline_defaults("ppe-bundle")["score_threshold"], 0.25)

    def test_an_unreadable_bundle_has_no_defaults(self):
        self.assertIsNone(bundles.pipeline_defaults("nope"))

    def test_geometry_types_match_the_model_fields(self):
        self.write_bundle(manifest={
            **MANIFEST, "tiling": {**MANIFEST["tiling"], "tile_size_px": 640.0}})

        geometry = bundles.geometry(bundles.pipeline_defaults("ppe-bundle"))

        # PositiveIntegerField, but chachak's manifest is float-tolerant.
        self.assertIsInstance(geometry["tile_size_px"], int)
        self.assertNotIn("score_threshold", geometry)


class ApplyDefaultsTests(BundleTestCase):
    def test_overwrites_whatever_the_row_carried(self):
        from videos.models import InferenceJob

        bundle = self.write_bundle()
        job = InferenceJob(
            model_source=InferenceJob.BUNDLE, bundle_path="ppe-bundle",
            # What a hand-edited POST (or a stale page) might have sent.
            pipeline=InferenceJob.RAW, detector_expand_ratio=9.0,
            merge_nms_iou=0.99, detector_checkpoint="/wrong/detector.pt",
        )

        job.sync_bundle()

        self.assertEqual(job.pipeline, pipelines.PEOPLE_DETECT_FIRST)
        self.assertEqual(job.detector_expand_ratio, 0.1)
        self.assertEqual(job.merge_nms_iou, 0.3)
        self.assertEqual(job.detector_checkpoint, str(bundle / "models/detector.engine"))

    def test_leaves_the_operator_s_threshold_alone(self):
        from videos.models import InferenceJob

        self.write_bundle()
        job = InferenceJob(model_source=InferenceJob.BUNDLE,
                           bundle_path="ppe-bundle", score_threshold=0.8)

        job.sync_bundle()

        self.assertEqual(job.score_threshold, 0.8)

    def test_does_nothing_for_the_other_sources(self):
        from videos.models import InferenceJob

        job = InferenceJob(model_source=InferenceJob.EXPORTED,
                           artifact_path="a.onnx", pipeline=InferenceJob.RAW)

        self.assertIsNone(job.sync_bundle())
        self.assertEqual(job.pipeline, InferenceJob.RAW)


class ValidateTests(BundleTestCase):
    def _statuses(self, result):
        return {c["label"]: c["status"] for c in result["checks"]}

    def test_a_complete_bundle_passes(self):
        self.write_bundle()

        result = bundles.validate("ppe-bundle")

        self.assertTrue(result["ok"], result["checks"])
        statuses = self._statuses(result)
        self.assertEqual(statuses["Manifest"], "ok")
        self.assertEqual(statuses["Model artifact"], "ok")
        self.assertEqual(statuses["Detector artifact"], "ok")
        # The .meta.json each artifact needs to be loadable at all, reported so
        # the operator sees the arch and input size the bundle will run at.
        meta = {c["label"]: c["detail"] for c in result["checks"]}["Model metadata"]
        self.assertIn("rfdetr", meta)
        self.assertIn("960", meta)
        self.assertEqual(statuses["Classes"], "ok")
        self.assertEqual(statuses["Standalone runtime"], "ok")
        # An engine is a portability caveat, not a failure — the load test is what
        # settles whether this one runs here.
        self.assertEqual(statuses["Format"], "info")
        self.assertEqual(statuses["Load test"], "info")

    def test_a_missing_model_fails(self):
        self.write_bundle(model=False)

        result = bundles.validate("ppe-bundle")

        self.assertFalse(result["ok"])
        self.assertEqual(self._statuses(result)["Model artifact"], "fail")

    def test_a_person_pipeline_without_its_detector_fails(self):
        self.write_bundle(detector=False)

        result = bundles.validate("ppe-bundle")

        self.assertFalse(result["ok"])
        self.assertEqual(self._statuses(result)["Detector artifact"], "fail")

    def test_a_raw_pipeline_needs_no_detector(self):
        self.write_bundle(manifest={
            **MANIFEST, "pipeline": "raw",
            "artifacts": {**MANIFEST["artifacts"], "detector": None},
        }, detector=False)

        result = bundles.validate("ppe-bundle")

        self.assertTrue(result["ok"], result["checks"])
        self.assertEqual(self._statuses(result)["Detector artifact"], "info")

    def test_an_unknown_pipeline_fails(self):
        self.write_bundle(manifest={**MANIFEST, "pipeline": "wormhole"})

        result = bundles.validate("ppe-bundle")

        self.assertFalse(result["ok"])
        self.assertEqual(self._statuses(result)["Pipeline"], "fail")

    def test_no_classes_fails(self):
        self.write_bundle(manifest={**MANIFEST, "classes": {}})

        result = bundles.validate("ppe-bundle")

        self.assertFalse(result["ok"])
        self.assertEqual(self._statuses(result)["Classes"], "fail")

    def test_a_declared_format_that_contradicts_the_model_fails(self):
        self.write_bundle(manifest={
            **MANIFEST, "bundle": {**MANIFEST["bundle"], "default_format": "onnx"}})

        result = bundles.validate("ppe-bundle")

        self.assertFalse(result["ok"])
        self.assertEqual(self._statuses(result)["Format"], "fail")

    def test_a_bundle_that_cannot_run_standalone_still_serves(self):
        self.write_bundle(runtime=False)

        result = bundles.validate("ppe-bundle")

        self.assertTrue(result["ok"], result["checks"])
        self.assertEqual(self._statuses(result)["Standalone runtime"], "info")

    def test_an_unreadable_manifest_reports_rather_than_raising(self):
        # This is rendered into a form the operator is mid-way through filling in.
        result = bundles.validate("nope")

        self.assertFalse(result["ok"])
        self.assertEqual(self._statuses(result), {"Manifest": "fail"})
        self.assertIsNone(result["defaults"])

    def test_defaults_come_back_for_the_form_to_apply(self):
        self.write_bundle()

        result = bundles.validate("ppe-bundle")

        self.assertEqual(result["defaults"]["pipeline"], pipelines.PEOPLE_DETECT_FIRST)
        self.assertEqual(result["name"], "ppe")


class LoadTestTests(BundleTestCase):
    """The half that proves the artifacts load, not just that the files exist."""

    def test_runs_the_bundle_through_the_trainer(self):
        bundle = self.write_bundle()

        with mock.patch("training.services.runner.predict_image",
                        return_value={"boxes": [], "classes": {"0": "helmet"}}) as predict:
            result = bundles.validate("ppe-bundle", load_test=True)

        self.assertTrue(result["ok"], result["checks"])
        payload = predict.call_args.args[0]
        self.assertEqual(payload["model_checkpoint"], str(bundle / "models/model.engine"))
        self.assertEqual(payload["detector_checkpoint"],
                         str(bundle / "models/detector.engine"))
        self.assertEqual(payload["pipeline"], pipelines.PEOPLE_DETECT_FIRST)
        # Built, not chosen: the frame exists only to make the trainer load things.
        self.assertTrue(Path(payload["image_path"]).is_file())

    def test_a_trainer_failure_is_the_answer_not_an_exception(self):
        # The failure a copied .engine actually hits: built for another GPU.
        self.write_bundle()

        with mock.patch("training.services.runner.predict_image",
                        side_effect=RuntimeError("deserialization failed")):
            result = bundles.validate("ppe-bundle", load_test=True)

        self.assertFalse(result["ok"])
        checks = {c["label"]: c for c in result["checks"]}
        self.assertEqual(checks["Load test"]["status"], "fail")
        self.assertIn("deserialization failed", checks["Load test"]["detail"])

    def test_skipped_when_there_is_no_model_to_load(self):
        self.write_bundle(model=False)

        with mock.patch("training.services.runner.predict_image") as predict:
            result = bundles.validate("ppe-bundle", load_test=True)

        self.assertFalse(predict.called)
        self.assertEqual(
            {c["label"]: c["status"] for c in result["checks"]}["Load test"], "info")

    def test_never_runs_unless_asked(self):
        # Saving an inference row validates too, and must not evict the trainer's
        # warm model or contend for the GPU.
        self.write_bundle()

        with mock.patch("training.services.runner.predict_image") as predict:
            bundles.validate("ppe-bundle")

        self.assertFalse(predict.called)
