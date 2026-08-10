"""All four ``gpu_infer`` pipelines against their ``chachak.pipeline`` equivalents, on CPU.

    python -m unittest gpu_infer.tests.test_pipelines_cpu

No GPU, no TensorRT, no engine. Both sides are driven by the *same* fake model — a pure function
of one preprocessed image — so anything that differs is a difference in **orchestration**: region
construction, chunk composition, remapping, merge order, or the masked tail. That is precisely
what needs checking, and it is checkable without hardware.

How the fake keeps the comparison honest:

* It is a pure function of a single preprocessed image, so its output cannot depend on how images
  were batched. Without that, a batching difference between the two paths would look like a
  numerics difference and hide the thing under test.
* The ``chachak`` side wraps it exactly as ``trt_infer.adapter.TrtAdapter.predict`` does —
  ``preprocess_torch`` -> raw -> ``to_friendy_torch`` -> ``.cpu()`` — using the real reference
  functions. So the reference path is the genuine one, not a reimplementation.
* It emits a fixed ``K`` with a ``num_detections`` count below it, so the padded tail's masking is
  exercised rather than bypassed.

Assertions are exact, **including row order**: both paths concatenate regions in region-major
order and both NMS implementations return survivors in descending score, so a row-order
difference would be a real defect rather than a presentation detail.
"""

import sys
import unittest
from dataclasses import replace
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402

from chachak.config import DetectorConfig, PipelineConfig, TilingConfig  # noqa: E402
from chachak.detector import Detector  # noqa: E402
from chachak.pipeline import (  # noqa: E402
    BatchDetectPipeline,
    BatchPeoplePipeline,
    ChainedPipeline,
    PeopleDetectFirstPipeline,
)
from onnx_infer.meta import InputSpec, ModelMeta  # noqa: E402
from trt_infer.postprocess_torch import to_friendy_torch  # noqa: E402
from trt_infer.preprocess_torch import preprocess_torch  # noqa: E402

from gpu_infer.config import GpuOptions  # noqa: E402
from gpu_infer.engine import EngineOutputs  # noqa: E402
from gpu_infer.pipeline import (  # noqa: E402
    BatchDetect,
    BatchPeople,
    Chained,
    MultiModel,
    PeopleDetectFirst,
)

MAX_DET = 12


def _meta(num_classes, class_map, *, size=640, layout="rgb", pad_value=0.0, scale="unit"):
    return ModelMeta(
        arch="rfdetr" if num_classes > 1 else "yolox",
        num_classes=num_classes,
        class_map=class_map,
        score_threshold=0.001,
        input=InputSpec(
            resize_mode="letterbox", size=size, multiple=0,
            pad_value=pad_value, input_scale=scale,
        ),
        normalize=None,
        layout=layout,
        box_coords="input_pixels",
        clip_boxes=True,
    )


MODEL_META = _meta(4, {0: "helmet", 1: "no_helmet", 2: "vest", 3: "no_vest"})
DETECTOR_META = _meta(1, {0: "person"}, layout="bgr", pad_value=114.0, scale="byte")


def _raw_for(prepared_chw, num_classes):
    """Deterministic ``(boxes, scores, labels, num_detections)`` for one preprocessed image.

    Seeded from the image's own content, so the same image always yields the same detections no
    matter which batch it travelled in. Scores descend so the NMS has a meaningful ranking, and
    ``num_detections`` is deliberately below ``MAX_DET`` so the padded tail's mask does real work.
    """
    key = int((prepared_chw.double().abs().sum() * 1000).round().item()) % (2**31 - 1)
    generator = torch.Generator().manual_seed(key)
    canvas = float(prepared_chw.shape[-1])

    x1 = torch.rand((MAX_DET,), generator=generator) * canvas * 0.7
    y1 = torch.rand((MAX_DET,), generator=generator) * canvas * 0.7
    boxes = torch.stack(
        [
            x1,
            y1,
            x1 + torch.rand((MAX_DET,), generator=generator) * canvas * 0.25 + 1.0,
            y1 + torch.rand((MAX_DET,), generator=generator) * canvas * 0.25 + 1.0,
        ],
        dim=-1,
    )
    scores = torch.sort(torch.rand((MAX_DET,), generator=generator), descending=True).values
    labels = torch.randint(0, num_classes, (MAX_DET,), generator=generator)
    count = 4 + (key % (MAX_DET - 4))  # in [4, MAX_DET)
    return boxes, scores, labels, int(count)


class FakeAdapter:
    """Stands in for ``trt_infer.adapter.TrtAdapter`` on the ``chachak`` side."""

    class _Model:
        def __init__(self, max_batch):
            self.max_batch = max_batch

    def __init__(self, meta, max_batch=1):
        self.meta = meta
        self.name = meta.arch
        self.num_classes = meta.num_classes
        self.score_threshold = meta.score_threshold
        self._model = self._Model(max_batch)

    def to(self, device):
        return self

    def eval(self):
        return self

    def predict(self, images, score_threshold=None):
        threshold = self.score_threshold if score_threshold is None else score_threshold
        results = []
        for image in images:
            prepared, transform = preprocess_torch(image, self.meta)
            boxes, scores, labels, count = _raw_for(prepared[0], self.meta.num_classes)
            friendy = to_friendy_torch(
                boxes[:count], scores[:count], labels[:count], transform, threshold,
                clip_boxes=self.meta.clip_boxes, box_coords=self.meta.box_coords,
            )
            results.append(friendy.detach().cpu())
        return results


class FakeEngine:
    """Stands in for ``gpu_infer.engine.AsyncEngine``, returning the padded raw outputs."""

    def __init__(self, meta, max_batch=1):
        self.meta = meta
        self.max_batch = max_batch
        self.efficientnms = True  # emits a num_detections count, like the plugin graphs
        self.submissions = 0

    def submit(self, batched):
        self.submissions += 1
        rows = [_raw_for(batched[i], self.meta.num_classes) for i in range(batched.shape[0])]
        return EngineOutputs(
            boxes=torch.stack([row[0] for row in rows]),
            scores=torch.stack([row[1] for row in rows]),
            labels=torch.stack([row[2] for row in rows]),
            num_detections=torch.tensor([row[3] for row in rows], dtype=torch.int32),
        )

    def submit_single(self, stacked, index):
        return self.submit(stacked[index : index + 1])


class FakeLoaded:
    """Stands in for ``gpu_infer.loader.LoadedEngine``."""

    def __init__(self, meta, max_batch=1):
        self.meta = meta
        self.engine = FakeEngine(meta, max_batch)
        self.max_input_hw = None
        self.class_map = {int(k): str(v) for k, v in meta.class_map.items()}

    @property
    def arch(self):
        return self.meta.arch

    @property
    def batchable(self):
        return self.engine.max_batch > 1


def _config(pipeline, **overrides):
    base = dict(
        name="cpu-parity",
        pipeline=pipeline,
        model_checkpoint=Path("/nonexistent/model.engine"),
        images=Path("/nonexistent/images"),
        labels=None,
        classes={0: "helmet", 1: "no_helmet", 2: "vest", 3: "no_vest"},
        output_dir=Path("/nonexistent/out"),
        device="cpu",
        infer_batch_size=4,
        score_threshold=0.05,
        map_score_threshold=None,
        merge_nms_iou=0.5,
        tiling=TilingConfig(
            tile_size_px=None, tile_width_pct=50.0, tile_height_pct=50.0,
            overlap=0.2, nms_iou=0.5,
        ),
        detector=DetectorConfig(
            checkpoint=Path("/nonexistent/detector.engine"),
            score_threshold=0.3, person_class_name="person", person_class_id=None,
            expand_ratio=0.15, nms_iou=0.5, min_box_size=0.0, batch_size=4,
        ),
        chain=[],
    )
    base.update(overrides)
    return PipelineConfig(**base)


def _frame(height=480, width=640, *, seed=3):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand((3, height, width), generator=generator)


_STRICT = GpuOptions.strict_parity(nms_rounds=32)


class _ParityCase(unittest.TestCase):
    """Shared harness: build both sides from one config and compare exactly."""

    def _reference(self, config, frame, cls, *, max_batch=1):
        adapter = FakeAdapter(MODEL_META, max_batch)
        detector = None
        if cls in (PeopleDetectFirstPipeline, BatchPeoplePipeline):
            detector = Detector(
                FakeAdapter(DETECTOR_META, max_batch),
                person_class_id=0,
                score_threshold=config.detector.score_threshold,
                batch_size=config.detector.batch_size,
            )
        pipeline = cls(adapter, torch.device("cpu"), config, detector=detector)
        return pipeline.process_batch([frame], [{"image_path": "frame.jpg"}])[0]

    def _actual(self, config, frame, cls, *, max_batch=1):
        model = FakeLoaded(MODEL_META, max_batch)
        detector = (
            FakeLoaded(DETECTOR_META, max_batch)
            if cls in (PeopleDetectFirst, BatchPeople)
            else None
        )
        pipeline = cls(config, model, detector=detector, options=_STRICT, device="cpu")
        return pipeline.process_frame(frame)

    def _assert_same(self, config, frame, reference_cls, actual_cls, *, max_batch=1):
        expected = self._reference(config, frame, reference_cls, max_batch=max_batch)
        actual = self._actual(config, frame, actual_cls, max_batch=max_batch)
        self.assertEqual(
            actual.shape[0], expected.shape[0],
            f"detection count differs: gpu={actual.shape[0]} chachak={expected.shape[0]}",
        )
        self.assertEqual(actual.tolist(), expected.tolist())
        return actual


class PeopleDetectFirstParityTest(_ParityCase):
    def test_matches_across_frames(self):
        config = _config("people_detect_first")
        for seed in range(6):
            with self.subTest(seed=seed):
                self._assert_same(
                    config, _frame(seed=seed), PeopleDetectFirstPipeline, PeopleDetectFirst
                )

    def test_matches_across_expand_ratios(self):
        for ratio in (0.0, 0.1, 0.5, 1.0):
            with self.subTest(expand_ratio=ratio):
                config = _config(
                    "people_detect_first",
                    detector=replace(_config("people_detect_first").detector, expand_ratio=ratio),
                )
                self._assert_same(
                    config, _frame(seed=11), PeopleDetectFirstPipeline, PeopleDetectFirst
                )

    def test_matches_with_a_min_box_size_floor(self):
        """Exercises grow-to-floor; the upscale branch stays dead unless the frame is smaller."""
        for min_box_size in (0.0, 64.0, 200.0):
            with self.subTest(min_box_size=min_box_size):
                base = _config("people_detect_first")
                config = _config(
                    "people_detect_first",
                    detector=replace(base.detector, min_box_size=min_box_size),
                )
                self._assert_same(
                    config, _frame(seed=13), PeopleDetectFirstPipeline, PeopleDetectFirst
                )

    def test_matches_when_the_frame_is_smaller_than_the_floor(self):
        """The upscale branch: no real pixels left, so the crop tensor is interpolated."""
        base = _config("people_detect_first")
        config = _config(
            "people_detect_first", detector=replace(base.detector, min_box_size=300.0)
        )
        self._assert_same(
            config, _frame(120, 160, seed=17), PeopleDetectFirstPipeline, PeopleDetectFirst
        )

    def test_matches_with_a_batchable_engine(self):
        """max_batch > 1 changes chunk composition on both sides; it must change it the same."""
        config = _config("people_detect_first", infer_batch_size=4)
        self._assert_same(
            config, _frame(seed=19), PeopleDetectFirstPipeline, PeopleDetectFirst, max_batch=8
        )

    def test_matches_across_merge_thresholds(self):
        for nms_iou in (0.3, 0.5, 0.9):
            with self.subTest(merge_nms_iou=nms_iou):
                config = _config("people_detect_first", merge_nms_iou=nms_iou)
                self._assert_same(
                    config, _frame(seed=23), PeopleDetectFirstPipeline, PeopleDetectFirst
                )

    def test_matches_when_the_detector_finds_nobody(self):
        """A threshold above every fake score: both paths must return an empty ``[0, 6]``."""
        base = _config("people_detect_first")
        config = _config(
            "people_detect_first",
            detector=replace(base.detector, score_threshold=1.01),
        )
        actual = self._assert_same(
            config, _frame(seed=29), PeopleDetectFirstPipeline, PeopleDetectFirst
        )
        self.assertEqual(list(actual.shape), [0, 6])


class BatchDetectParityTest(_ParityCase):
    def test_matches_with_fractional_tiles(self):
        config = _config("batch_detect")
        for seed in (3, 5, 7):
            with self.subTest(seed=seed):
                self._assert_same(config, _frame(seed=seed), BatchDetectPipeline, BatchDetect)

    def test_matches_with_fixed_pixel_tiles(self):
        config = _config(
            "batch_detect",
            tiling=TilingConfig(tile_size_px=256, overlap=0.2, nms_iou=0.5),
        )
        self._assert_same(config, _frame(seed=31), BatchDetectPipeline, BatchDetect)

    def test_matches_across_overlaps(self):
        for overlap in (0.0, 0.2, 0.5):
            with self.subTest(overlap=overlap):
                config = _config(
                    "batch_detect",
                    tiling=TilingConfig(
                        tile_size_px=None, tile_width_pct=50.0, tile_height_pct=50.0,
                        overlap=overlap, nms_iou=0.5,
                    ),
                )
                self._assert_same(config, _frame(seed=37), BatchDetectPipeline, BatchDetect)

    def test_matches_with_a_ragged_tile_grid(self):
        """A frame whose tiles do not divide evenly, so edge tiles are clamped."""
        config = _config("batch_detect")
        self._assert_same(config, _frame(481, 641, seed=41), BatchDetectPipeline, BatchDetect)

    def test_matches_with_a_batchable_engine(self):
        config = _config("batch_detect", infer_batch_size=4)
        self._assert_same(
            config, _frame(seed=43), BatchDetectPipeline, BatchDetect, max_batch=8
        )

    def test_uses_the_tiling_nms_threshold_not_the_merge_one(self):
        """``batch_detect`` merges with ``tiling.nms_iou``; getting this wrong is invisible
        unless the two thresholds differ."""
        config = _config(
            "batch_detect",
            merge_nms_iou=0.99,
            tiling=TilingConfig(
                tile_size_px=None, tile_width_pct=50.0, tile_height_pct=50.0,
                overlap=0.2, nms_iou=0.3,
            ),
        )
        self._assert_same(config, _frame(seed=47), BatchDetectPipeline, BatchDetect)


class BatchPeopleParityTest(_ParityCase):
    def test_matches_across_frames(self):
        config = _config("batch_people")
        for seed in (3, 5, 7):
            with self.subTest(seed=seed):
                self._assert_same(config, _frame(seed=seed), BatchPeoplePipeline, BatchPeople)

    def test_ignores_tile_size_px_like_the_original_does(self):
        """``batch_people`` always tiles fractionally, even when ``tile_size_px`` is set.

        That is an inconsistency in ``chachak.pipeline`` (``_person_boxes`` calls ``tile_frame``
        directly while ``_tile_infer`` checks ``tile_size_px``). Reproduced deliberately — this
        test is what pins it, so a later "fix" here fails loudly instead of quietly changing what
        an existing config detects.
        """
        config = _config(
            "batch_people",
            tiling=TilingConfig(
                tile_size_px=256, tile_width_pct=50.0, tile_height_pct=50.0,
                overlap=0.2, nms_iou=0.5,
            ),
        )
        self._assert_same(config, _frame(seed=53), BatchPeoplePipeline, BatchPeople)

    def test_matches_with_the_detector_nms_threshold(self):
        for nms_iou in (0.3, 0.7):
            with self.subTest(detector_nms_iou=nms_iou):
                base = _config("batch_people")
                config = _config(
                    "batch_people", detector=replace(base.detector, nms_iou=nms_iou)
                )
                self._assert_same(config, _frame(seed=59), BatchPeoplePipeline, BatchPeople)

    def test_matches_with_a_batchable_detector(self):
        config = _config("batch_people")
        self._assert_same(
            config, _frame(seed=61), BatchPeoplePipeline, BatchPeople, max_batch=8
        )


class ChainParityTest(_ParityCase):
    def _reference_chain(self, config, frame, children):
        adapter = FakeAdapter(MODEL_META)
        detector = Detector(
            FakeAdapter(DETECTOR_META),
            person_class_id=0,
            score_threshold=config.detector.score_threshold,
            batch_size=config.detector.batch_size,
        )
        lookup = {
            "people_detect_first": PeopleDetectFirstPipeline,
            "batch_people": BatchPeoplePipeline,
            "batch_detect": BatchDetectPipeline,
        }
        subs = [
            lookup[name](adapter, torch.device("cpu"), config, detector=detector)
            for name in children
        ]
        chained = ChainedPipeline(
            adapter, torch.device("cpu"), config, detector=detector, pipelines=subs
        )
        return chained.process_batch([frame], [{"image_path": "frame.jpg"}])[0]

    def _actual_chain(self, config, frame, children):
        from gpu_infer.pipeline import _PIPELINES

        model = FakeLoaded(MODEL_META)
        detector = FakeLoaded(DETECTOR_META)
        subs = [
            _PIPELINES[name](config, model, detector=detector, options=_STRICT, device="cpu")
            for name in children
        ]
        chained = Chained(
            config, model, pipelines=subs, detector=detector, options=_STRICT, device="cpu"
        )
        return chained.process_frame(frame)

    def test_matches_for_two_children(self):
        children = ["people_detect_first", "batch_detect"]
        config = _config("chain", chain=children)
        expected = self._reference_chain(config, _frame(seed=67), children)
        actual = self._actual_chain(config, _frame(seed=67), children)
        self.assertEqual(actual.shape[0], expected.shape[0])
        self.assertEqual(actual.tolist(), expected.tolist())

    def test_matches_for_three_children(self):
        children = ["people_detect_first", "batch_people", "batch_detect"]
        config = _config("chain", chain=children)
        expected = self._reference_chain(config, _frame(seed=71), children)
        actual = self._actual_chain(config, _frame(seed=71), children)
        self.assertEqual(actual.tolist(), expected.tolist())

    def test_names_itself_like_the_original(self):
        children = ["people_detect_first", "batch_detect"]
        config = _config("chain", chain=children)
        model = FakeLoaded(MODEL_META)
        from gpu_infer.pipeline import _PIPELINES

        subs = [
            _PIPELINES[name](config, model, options=_STRICT, device="cpu")
            for name in children
        ]
        chained = Chained(config, model, pipelines=subs, options=_STRICT, device="cpu")
        self.assertEqual(chained.name, "chain[people_detect_first+batch_detect]")


class MultiModelTest(_ParityCase):
    """The class-space remap and cross-model merge the bundle's ``run_batch`` performs."""

    def test_single_model_remaps_onto_the_eval_class_space(self):
        from chachak._friendy import remap_raw_predictions_to_eval_classes

        config = _config("people_detect_first")
        frame = _frame(seed=73)
        eval_classes = {0: "vest", 1: "helmet"}  # deliberately reordered and narrowed

        expected_raw = self._reference(config, frame, PeopleDetectFirstPipeline)
        expected = remap_raw_predictions_to_eval_classes(
            expected_raw, dict(MODEL_META.class_map), eval_classes
        )

        model = FakeLoaded(MODEL_META)
        detector = FakeLoaded(DETECTOR_META)
        pipeline = PeopleDetectFirst(
            config, model, detector=detector, options=_STRICT, device="cpu"
        )
        wrapped = MultiModel(
            config, [(pipeline, model)], options=_STRICT, eval_classes=eval_classes
        )
        actual = wrapped.process_frame(frame)
        self.assertEqual(actual.tolist(), expected.tolist())

    def test_no_eval_classes_means_no_remap(self):
        config = _config("people_detect_first")
        frame = _frame(seed=79)
        expected = self._reference(config, frame, PeopleDetectFirstPipeline)

        model = FakeLoaded(MODEL_META)
        detector = FakeLoaded(DETECTOR_META)
        pipeline = PeopleDetectFirst(
            config, model, detector=detector, options=_STRICT, device="cpu"
        )
        wrapped = MultiModel(config, [(pipeline, model)], options=_STRICT, eval_classes=None)
        self.assertEqual(wrapped.process_frame(frame).tolist(), expected.tolist())


class ProfileHelperTest(unittest.TestCase):
    """The two ``chachak.detector`` helpers reimplemented in ``gpu_infer.loader``."""

    def test_fit_to_input_profile_matches_the_original(self):
        from chachak.detector import _fit_to_input_profile

        from gpu_infer.loader import fit_to_input_profile

        for shape in ((3, 1080, 1920), (3, 480, 640), (3, 2160, 3840), (3, 100, 100)):
            for max_hw in (None, (640, 640), (1024, 1024), (2000, 2000)):
                with self.subTest(shape=shape, max_hw=max_hw):
                    image = torch.rand(shape, generator=torch.Generator().manual_seed(5))
                    expected = _fit_to_input_profile(image, max_hw)
                    actual = fit_to_input_profile(image, max_hw)
                    self.assertEqual(list(actual.shape), list(expected.shape))
                    self.assertTrue(torch.equal(actual, expected))

    def test_resolve_person_class_id_matches_the_original_rules(self):
        from gpu_infer.loader import resolve_person_class_id

        self.assertEqual(
            resolve_person_class_id({0: "car", 3: "person"}, person_class_name="person"), 3
        )
        self.assertEqual(
            resolve_person_class_id({0: "car"}, person_class_name="person", person_class_id=7), 7
        )
        # Absent name falls back to 0, matching load_detector's documented behaviour.
        self.assertEqual(resolve_person_class_id({0: "car"}, person_class_name="person"), 0)
        self.assertEqual(resolve_person_class_id({}, person_class_name=None), 0)

    def test_case_insensitive_person_lookup(self):
        from gpu_infer.loader import resolve_person_class_id

        self.assertEqual(
            resolve_person_class_id({2: "Person"}, person_class_name="person"), 2
        )


class UnsupportedPipelineTest(unittest.TestCase):
    def test_an_unknown_pipeline_is_refused_by_name(self):
        from gpu_infer.pipeline import build

        config = _config("people_detect_first")
        config = replace(config, pipeline="raw")
        with self.assertRaises(ValueError) as caught:
            build(config, FakeLoaded(MODEL_META), options=_STRICT)
        self.assertIn("raw", str(caught.exception))

    def test_a_chain_with_no_children_is_refused(self):
        from gpu_infer.pipeline import build

        config = _config("chain", chain=[])
        with self.assertRaises(ValueError):
            build(config, FakeLoaded(MODEL_META), options=_STRICT)

    def test_people_pipelines_require_a_detector(self):
        config = _config("people_detect_first")
        pipeline = PeopleDetectFirst(
            config, FakeLoaded(MODEL_META), detector=None, options=_STRICT, device="cpu"
        )
        with self.assertRaises(ValueError):
            pipeline.process_frame(_frame(seed=83))


if __name__ == "__main__":
    unittest.main()
