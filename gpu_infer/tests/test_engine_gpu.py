"""``gpu_infer.engine.AsyncEngine`` against ``trt_infer.session.TrtModel.run_torch``.

    python -m unittest gpu_infer.tests.test_engine_gpu

Needs a CUDA GPU, TensorRT, and at least one engine built by the *running* TensorRT version;
skips with a specific reason otherwise (see :mod:`gpu_infer.tests.engines`).

This module exists to prove **one** claim, on which the entire package rests:

    Removing ``stream.synchronize()`` from the submission path does not change a single output
    byte, because the output copy is itself stream-ordered.

Everything else in ``gpu_infer`` is device-agnostic torch and is tested on CPU. This is the part
that can only be established against a real engine on real hardware, so it is tested
adversarially: not just "one submit matches", but "N submits with no synchronization anywhere
between them, all outputs held live, still match".
"""

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gpu_infer.tests import engines  # noqa: E402

_SKIP = engines.skip_reason()


def _input_for(engine, batch=1):
    """A deterministic input matching this engine's optimum profile shape."""
    import torch

    shape = engine._warmup_shape()
    if shape is None:
        raise unittest.SkipTest("engine profile shape unreadable")
    shape = (batch,) + tuple(shape[1:])
    generator = torch.Generator(device="cpu").manual_seed(1234 + batch)
    return torch.rand(shape, generator=generator).to(
        device=engine.device, dtype=torch.float32
    )


def _triples_equal(case, actual, expected, label):
    """Every tensor in two ``[boxes, scores, labels]`` triples is byte-identical."""
    import torch

    for name, got, want in zip(("boxes", "scores", "labels"), actual, expected):
        case.assertEqual(
            tuple(got.shape), tuple(want.shape), f"{label} {name} shape"
        )
        case.assertTrue(torch.equal(got, want), f"{label} {name} differs")


@unittest.skipIf(_SKIP, _SKIP or "")
class AsyncMatchesSynchronousTest(unittest.TestCase):
    """The load-bearing differential."""

    @classmethod
    def setUpClass(cls):
        from onnx_infer.meta import ModelMeta

        from gpu_infer.engine import AsyncEngine

        cls.bundle = engines.find_bundles()[0]
        cls.path = cls.bundle / "models" / "model.engine"
        cls.meta = ModelMeta.load(cls.path.with_suffix(".meta.json"))
        cls.engine = AsyncEngine(cls.path, cls.meta)
        print(f"\n[test_engine_gpu] {cls.bundle.name} arch={cls.meta.arch} "
              f"efficientnms={cls.engine.efficientnms} max_batch={cls.engine.max_batch}")

    def _reference(self, batched):
        """The synchronous path, as a ``[boxes, scores, labels]`` list per image."""
        return self.engine.model.run_torch(batched)

    def _async(self, batched):
        """The un-synchronized path, reshaped to the same per-image triples."""
        outputs = self.engine.submit(batched)
        valid = outputs.valid_mask()
        triples = []
        for index in range(outputs.batch):
            keep = valid[index]
            triples.append(
                [
                    outputs.boxes[index][keep],
                    outputs.scores[index][keep],
                    outputs.labels[index][keep],
                ]
            )
        return triples

    def test_one_submission_matches(self):
        batched = _input_for(self.engine)
        actual = self._async(batched)
        expected = self._reference(batched)
        self.assertEqual(len(actual), len(expected))
        for index, (got, want) in enumerate(zip(actual, expected)):
            _triples_equal(self, got, want, f"image {index}")

    def test_many_chained_submissions_with_no_synchronization(self):
        """**The claim.** Submit repeatedly, holding every result, syncing nowhere.

        If the stream-ordered ``.clone()`` were not sufficient, one of two things would show up
        here: an earlier submission's held output would be overwritten by a later execution
        reusing the allocator's buffer, or a copy would read a buffer TensorRT had not finished
        writing. Both are order-of-magnitude-obvious failures on real detections, and both are
        invisible in a single-submission test.
        """
        import torch

        count = 8
        inputs = [_input_for(self.engine, batch=1) for _ in range(count)]
        # Distinct inputs, so a stale buffer cannot coincidentally match.
        inputs = [tensor * (0.2 + 0.1 * index) for index, tensor in enumerate(inputs)]

        held = [self._async(tensor) for tensor in inputs]  # no sync anywhere in here
        torch.cuda.current_stream().synchronize()

        for index, tensor in enumerate(inputs):
            expected = self._reference(tensor)
            for image, (got, want) in enumerate(zip(held[index], expected)):
                _triples_equal(self, got, want, f"submission {index} image {image}")

    def test_held_outputs_survive_later_submissions(self):
        """A result captured before further work must not change afterwards."""
        import torch

        first_input = _input_for(self.engine, batch=1)
        first = self._async(first_input)
        snapshot = [tensor.clone() for tensor in first[0]]

        for index in range(4):
            self._async(_input_for(self.engine, batch=1) * (0.5 + index * 0.1))
        torch.cuda.current_stream().synchronize()

        _triples_equal(self, first[0], snapshot, "held output")

    def test_repeated_submission_of_the_same_input_is_deterministic(self):
        """Trained weights, so top-K ties are not in play and this must be exact."""
        batched = _input_for(self.engine)
        first = [[tensor.clone() for tensor in triple] for triple in self._async(batched)]
        for attempt in range(3):
            again = self._async(batched)
            for image, (got, want) in enumerate(zip(again, first)):
                _triples_equal(self, got, want, f"attempt {attempt} image {image}")


@unittest.skipIf(_SKIP, _SKIP or "")
class StreamTest(unittest.TestCase):
    """The stream choice must be a pure performance decision, never a correctness one.

    TensorRT inserts its own ``cudaStreamSynchronize`` calls around work submitted to the
    *default* stream — it says so on every ``enqueueV3`` — which would give back the overlap this
    package removes ``run_torch``'s synchronize to obtain. So the pipeline runs on a dedicated
    non-default stream. This pins that the results do not depend on that choice, so the
    optimization can never be blamed for a detection difference.
    """

    @classmethod
    def setUpClass(cls):
        cls.paths = [
            engines.find_bundles()[0] / "models" / role
            for role in ("model.engine", "detector.engine")
        ]
        cls.paths = [path for path in cls.paths if path.exists()]

    def test_results_are_identical_on_either_stream(self):
        """Same engine, same input, both streams — via ``inference_scope`` for the second.

        The scope is not a convenience here. An engine's output-allocator buffers are long-lived
        shared state, so two streams submitting to one engine with no ordering between them race
        on those buffers — which shows up as ``CUDA error 700 (illegal address)`` raised somewhere
        else entirely. ``inference_scope`` supplies the ``wait_stream`` pair that orders them.
        """
        import torch

        from onnx_infer.meta import ModelMeta

        from gpu_infer.engine import AsyncEngine, inference_scope

        for path in self.paths:
            with self.subTest(engine=path.name):
                engine = AsyncEngine(
                    path, ModelMeta.load(path.with_suffix(".meta.json"))
                )
                batched = _input_for(engine)

                on_default = engine.submit(batched)
                default_copy = [
                    on_default.boxes.clone(),
                    on_default.scores.clone(),
                    on_default.labels.clone(),
                ]
                torch.cuda.synchronize()

                with inference_scope(inputs=(batched,)):
                    on_dedicated = engine.submit(batched)
                    dedicated_copy = [
                        on_dedicated.boxes.clone(),
                        on_dedicated.scores.clone(),
                        on_dedicated.labels.clone(),
                    ]
                torch.cuda.synchronize()

                _triples_equal(
                    self, dedicated_copy, default_copy, f"{path.name} across streams"
                )

    def test_the_scope_orders_against_the_callers_stream(self):
        """Work queued before the scope must be visible inside it.

        Without ``wait_stream`` this is a genuine race: a frame decoded on the caller's stream
        can still be mid-write when the engine starts reading it. Enqueue enough slow work to
        make an unordered read observably wrong, then check the value seen inside the scope.
        """
        import torch

        from gpu_infer.engine import inference_scope

        source = torch.zeros(4096, 4096, device="cuda")
        for _ in range(20):  # keep the caller's stream busy
            source = source + 1.0
        expected = 20.0

        with inference_scope(inputs=(source,)):
            observed = float(source[0, 0])
        self.assertEqual(observed, expected)

    def test_the_dedicated_stream_is_not_the_default_one(self):
        """Guards the whole point: a helper that quietly returned the default stream would
        silence nothing and slow everything."""
        import torch

        from gpu_infer.engine import inference_stream

        stream = inference_stream()
        self.assertNotEqual(
            stream.cuda_stream,
            torch.cuda.default_stream(stream.device).cuda_stream,
        )

    def test_the_stream_is_cached_per_device(self):
        from gpu_infer.engine import inference_stream

        self.assertIs(inference_stream(), inference_stream())


@unittest.skipIf(_SKIP, _SKIP or "")
class OutputLayoutTest(unittest.TestCase):
    """Both graph layouts normalize to the same padded shape."""

    @classmethod
    def setUpClass(cls):
        cls.bundles = engines.find_bundles()

    def _engine(self, path):
        from onnx_infer.meta import ModelMeta

        from gpu_infer.engine import AsyncEngine

        return AsyncEngine(path, ModelMeta.load(path.with_suffix(".meta.json")))

    def test_outputs_are_always_batch_by_maxdet(self):
        for path in engines.find_engines():
            with self.subTest(engine=path.name, bundle=path.parent.parent.name):
                engine = self._engine(path)
                outputs = engine.submit(_input_for(engine))
                self.assertEqual(outputs.batch, 1)
                self.assertEqual(list(outputs.boxes.shape), [1, outputs.max_det, 4])
                self.assertEqual(list(outputs.scores.shape), [1, outputs.max_det])
                self.assertEqual(list(outputs.labels.shape), [1, outputs.max_det])
                self.assertEqual(
                    list(outputs.valid_mask().shape), [1, outputs.max_det]
                )

    def test_efficientnms_valid_mask_matches_the_count(self):
        """The ``arange`` mask must select exactly what ``det_boxes[i, :n]`` would."""
        paths = engines.find_engines(efficientnms=True)
        if not paths:
            self.skipTest("no EfficientNMS engine available")
        engine = self._engine(paths[0])
        outputs = engine.submit(_input_for(engine))
        count = int(outputs.num_detections[0])
        mask = outputs.valid_mask()[0]
        self.assertEqual(int(mask.sum()), count)
        self.assertTrue(bool(mask[:count].all()))
        self.assertFalse(bool(mask[count:].any()))

    def test_passthrough_has_no_count_and_is_all_valid(self):
        paths = engines.find_engines(efficientnms=False)
        if not paths:
            self.skipTest("no passthrough engine available")
        engine = self._engine(paths[0])
        outputs = engine.submit(_input_for(engine))
        self.assertIsNone(outputs.num_detections)
        self.assertTrue(bool(outputs.valid_mask().all()))

    def test_passthrough_keeps_every_detection(self):
        """Guards the bug ``_split_passthrough`` documents: these graphs have no batch axis,
        so indexing it returns detection *zero* and silently discards the rest."""
        paths = engines.find_engines(efficientnms=False)
        if not paths:
            self.skipTest("no passthrough engine available")
        engine = self._engine(paths[0])
        outputs = engine.submit(_input_for(engine))
        self.assertGreater(
            outputs.max_det, 1, "a passthrough graph emits its whole query set"
        )


@unittest.skipIf(_SKIP, _SKIP or "")
class SubmitSingleTest(unittest.TestCase):
    """The offset-pointer path used for batch-1 passthrough engines."""

    @classmethod
    def setUpClass(cls):
        from onnx_infer.meta import ModelMeta

        from gpu_infer.engine import AsyncEngine

        cls.path = engines.find_bundles()[0] / "models" / "model.engine"
        cls.meta = ModelMeta.load(cls.path.with_suffix(".meta.json"))
        cls.engine = AsyncEngine(cls.path, cls.meta)

    def test_row_of_a_stack_matches_submitting_that_row_alone(self):
        import torch

        rows = [_input_for(self.engine, batch=1) * (0.3 + 0.15 * i) for i in range(4)]
        stacked = torch.cat(rows, dim=0).contiguous()

        for index, row in enumerate(rows):
            with self.subTest(index=index):
                from_stack = self.engine.submit_single(stacked, index)
                alone = self.engine.submit(row)
                torch.cuda.current_stream().synchronize()
                _triples_equal(
                    self,
                    [from_stack.boxes, from_stack.scores, from_stack.labels],
                    [alone.boxes, alone.scores, alone.labels],
                    f"row {index}",
                )

    def test_the_scratch_fallback_produces_the_same_answer(self):
        """Force the unaligned path and check it agrees with the direct-pointer one."""
        import torch

        from gpu_infer import engine as engine_module

        rows = [_input_for(self.engine, batch=1) * (0.4 + 0.1 * i) for i in range(3)]
        stacked = torch.cat(rows, dim=0).contiguous()

        direct = [self.engine.submit_single(stacked, i) for i in range(len(rows))]
        direct = [[out.boxes.clone(), out.scores.clone(), out.labels.clone()] for out in direct]

        original = engine_module._ADDRESS_ALIGNMENT
        try:
            # An alignment no real stride satisfies, so every row takes the scratch copy.
            engine_module._ADDRESS_ALIGNMENT = 7919
            self.engine._scratch = None
            via_scratch = [self.engine.submit_single(stacked, i) for i in range(len(rows))]
            via_scratch = [[o.boxes, o.scores, o.labels] for o in via_scratch]
            torch.cuda.current_stream().synchronize()
        finally:
            engine_module._ADDRESS_ALIGNMENT = original
            self.engine._scratch = None

        for index, (got, want) in enumerate(zip(via_scratch, direct)):
            _triples_equal(self, got, want, f"scratch row {index}")


@unittest.skipIf(_SKIP, _SKIP or "")
class InheritedGuardsTest(unittest.TestCase):
    """``TrtModel``'s protections must still fire through the async path."""

    @classmethod
    def setUpClass(cls):
        from onnx_infer.meta import ModelMeta

        from gpu_infer.engine import AsyncEngine

        cls.path = engines.find_bundles()[0] / "models" / "model.engine"
        cls.engine = AsyncEngine(
            cls.path, ModelMeta.load(cls.path.with_suffix(".meta.json"))
        )

    def test_a_batch_over_the_profile_raises_instead_of_lying(self):
        """Over-profile submission used to silently return image 0's detections for all.

        TensorRT reports the rejected ``set_input_shape`` through its *logger*, leaves the
        context on its previous shape, and runs anyway. ``_as_engine_input`` raises; this pins
        that the async path routes through it.
        """
        max_batch = self.engine.max_batch
        if max_batch is None:
            self.skipTest("engine profile batch ceiling unreadable")
        with self.assertRaises(ValueError) as caught:
            self.engine.submit(_input_for(self.engine, batch=max_batch + 1))
        self.assertIn("batch profile", str(caught.exception))

    def test_numpy_input_is_rejected(self):
        import numpy as np

        with self.assertRaises(TypeError):
            self.engine.submit(np.zeros((1, 3, 64, 64), dtype=np.float32))

    def test_warmup_leaves_the_allocator_buffers_populated(self):
        """After construction the grow-only allocator should not need to grow again."""
        buffers = self.engine.model._allocator.buffers
        self.assertTrue(buffers, "warm-up should have allocated every output buffer")
        sizes = {name: buffer.numel() for name, buffer in buffers.items()}
        self.engine.submit(_input_for(self.engine))
        for name, before in sizes.items():
            self.assertEqual(
                buffers[name].numel(), before, f"{name} reallocated after warm-up"
            )


if __name__ == "__main__":
    unittest.main()
