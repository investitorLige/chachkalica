"""``TrtModel``'s two entry points and the contracts other code depends on.

``run_torch`` is the new GPU-resident path. ``run`` is the pre-existing numpy one,
now implemented on top of it — and it has out-of-tree callers
(``person_model_test/run_yolox_export_eval.py``,
``inferlica/benchmark/kernel_util.py``) that unpack a bare triple and call
``labels.astype(np.int64)``. Those two facts are what this file pins down.

The third contract is the output allocator's lifetime — that it is built once and its
buffers reused (or the process leaks the GPU until it wedges), and that reuse is
nonetheless invisible to a caller holding results across calls.

Nothing here compares two engine invocations: the shared fixture's engine is not
self-consistent across calls (see ``conftest``). The value-level logic —
``_unpack_efficientnms``, ``_as_engine_input``, and ``run``'s numpy conversion — is
tested directly as pure functions instead, which is both deterministic and more
precise about what is being asserted.

Requires a CUDA GPU + TensorRT; skips cleanly otherwise.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

torch = pytest.importorskip("torch")

from trt_infer.session import _split_passthrough, _unpack_efficientnms  # noqa: E402


# --------------------------------------------------------------- pure output slicing


def test_unpack_efficientnms_slices_each_image_to_its_own_count():
    """The plugin emits fixed-size, zero-padded outputs; only num_detections says
    where each image's real detections stop."""
    counts = torch.tensor([[3], [0], [2]], dtype=torch.int32)
    boxes = torch.arange(3 * 4 * 4, dtype=torch.float32).reshape(3, 4, 4)
    scores = torch.arange(3 * 4, dtype=torch.float32).reshape(3, 4)
    classes = torch.arange(3 * 4, dtype=torch.int32).reshape(3, 4)

    triples = _unpack_efficientnms([counts, boxes, scores, classes], 3)

    assert [triple[0].shape[0] for triple in triples] == [3, 0, 2]
    for i, n in enumerate([3, 0, 2]):
        torch.testing.assert_close(triples[i][0], boxes[i, :n])
        torch.testing.assert_close(triples[i][1], scores[i, :n])
        torch.testing.assert_close(triples[i][2], classes[i, :n])


def test_unpack_efficientnms_accepts_flat_plugin_output():
    """TensorRT hands back num_detections as [B,1] or [B]; both must work."""
    flat = _unpack_efficientnms(
        [torch.tensor([2, 1], dtype=torch.int32),
         torch.zeros(2, 5, 4), torch.zeros(2, 5), torch.zeros(2, 5, dtype=torch.int32)],
        2,
    )
    assert [triple[0].shape[0] for triple in flat] == [2, 1]


def test_split_passthrough_indexes_the_batch_axis():
    boxes = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    scores = torch.arange(2 * 3, dtype=torch.float32).reshape(2, 3)
    labels = torch.arange(2 * 3, dtype=torch.int64).reshape(2, 3)

    triples = _split_passthrough([boxes, scores, labels], 2)

    assert len(triples) == 2
    for i in range(2):
        torch.testing.assert_close(triples[i][0], boxes[i])
        torch.testing.assert_close(triples[i][1], scores[i])
        torch.testing.assert_close(triples[i][2], labels[i])


# ------------------------------------------------------------------ engine plumbing


@pytest.fixture(scope="module")
def model(batched_yolox_engine):
    from onnx_infer.meta import ModelMeta
    from trt_infer.session import TrtModel

    meta = ModelMeta.load(batched_yolox_engine.path.with_suffix(".meta.json"))
    return TrtModel(batched_yolox_engine.path, meta, device="cuda")


@pytest.fixture(scope="module")
def batch(batched_yolox_engine):
    def make(n, seed=0):
        rng = np.random.default_rng(seed)
        return rng.random((n, 3, *batched_yolox_engine.static_hw), dtype=np.float32)

    return make


@pytest.fixture(scope="module")
def profile(batched_yolox_engine):
    return batched_yolox_engine.batch_profile


def test_reads_the_engines_batch_profile(model, profile):
    assert model._max_batch == profile


def test_run_returns_a_bare_numpy_triple_for_one_image(model, batch):
    """The shape every caller written before batching still expects."""
    out = model.run(batch(1))

    assert isinstance(out, list) and len(out) == 3
    boxes, scores, labels = out
    for array in (boxes, scores, labels):
        assert isinstance(array, np.ndarray), type(array)
    assert boxes.ndim == 2 and boxes.shape[1] == 4
    assert scores.shape == (boxes.shape[0],)
    # The out-of-tree callers' literal next move.
    assert labels.astype(np.int64).shape == (boxes.shape[0],)


def test_run_returns_a_list_of_triples_for_a_real_batch(model, batch):
    out = model.run(batch(2))

    assert len(out) == 2
    for triple in out:
        assert len(triple) == 3
        assert all(isinstance(array, np.ndarray) for array in triple)


def test_run_torch_always_returns_a_list_of_cuda_triples(model, batch, profile):
    for n in (1, 2, profile):
        out = model.run_torch(torch.from_numpy(batch(n)).cuda())
        assert len(out) == n, n
        for triple in out:
            assert len(triple) == 3
            for tensor in triple:
                assert torch.is_tensor(tensor) and tensor.is_cuda


def test_run_converts_run_torch_output_without_changing_dtypes(model, monkeypatch):
    """``run``'s only job over ``run_torch`` is the numpy conversion and the
    bare-triple shape. Driven off a stub so it is exact rather than at the mercy of
    the engine's tie-breaking."""
    fake = [
        [torch.ones(2, 4).cuda(), torch.full((2,), 0.5).cuda(),
         torch.tensor([1, 2], dtype=torch.int32).cuda()],
        [torch.zeros(1, 4).cuda(), torch.full((1,), 0.25).cuda(),
         torch.tensor([0], dtype=torch.int32).cuda()],
    ]
    monkeypatch.setattr(model, "run_torch", lambda _batched: fake)

    out = model.run(np.zeros((2, 3, 8, 8), dtype=np.float32))

    assert len(out) == 2
    for triple_out, triple_fake in zip(out, fake):
        for array, tensor in zip(triple_out, triple_fake):
            assert isinstance(array, np.ndarray)
            assert array.dtype == np.dtype(str(tensor.dtype).removeprefix("torch."))
            np.testing.assert_array_equal(array, tensor.cpu().numpy())


def test_run_unwraps_a_single_image_from_run_torch(model, monkeypatch):
    fake = [[torch.ones(2, 4).cuda(), torch.ones(2).cuda(), torch.ones(2).cuda()]]
    monkeypatch.setattr(model, "run_torch", lambda _batched: fake)

    boxes, scores, labels = model.run(np.zeros((1, 3, 8, 8), dtype=np.float32))
    assert boxes.shape == (2, 4) and scores.shape == (2,) and labels.shape == (2,)


# -------------------------------------------------------------------- input coercion


def test_accepts_a_cpu_tensor(model, batch):
    out = model.run_torch(torch.from_numpy(batch(1)))
    assert len(out) == 1 and out[0][0].is_cuda


def test_accepts_a_non_contiguous_input(model, batched_yolox_engine):
    """A sub-view of a larger buffer, which is what person crops are. Note
    ``torch.flip`` would not do: it materializes a contiguous copy."""
    height, width = batched_yolox_engine.static_hw
    strided = torch.rand(2, 3, height + 16, width + 16, device="cuda")[:, :, :height, :width]
    assert not strided.is_contiguous()

    out = model.run_torch(strided)
    assert len(out) == 2


def test_accepts_a_non_float32_input(model, batch):
    out = model.run_torch(torch.from_numpy(batch(1)).cuda().to(torch.float64))
    assert len(out) == 1


def test_run_torch_rejects_numpy(model, batch):
    with pytest.raises(TypeError, match="use run\\(\\) for numpy input"):
        model.run_torch(batch(1))


def test_a_batch_over_the_profile_is_rejected_not_silently_wrong(model, batch, profile):
    """TensorRT reports an oversized set_input_shape through its logger, not an
    exception, then runs at the previous shape — so without this check every image
    past the first comes back holding a slice of the first one's output."""
    with pytest.raises(ValueError, match="batch profile of at most"):
        model.run_torch(torch.from_numpy(batch(profile + 1)).cuda())

    with pytest.raises(ValueError, match="batch profile of at most"):
        model.run(batch(profile + 1))


# ------------------------------------------------------------- output buffer lifetime


def test_reallocate_only_allocates_when_it_needs_more_room():
    """The allocator's whole job. Growing is allowed; re-allocating for a request the
    current buffer already fits is the leak."""
    trt = pytest.importorskip("tensorrt")
    if not torch.cuda.is_available():
        pytest.skip("the allocator's buffers are CUDA tensors")

    from trt_infer.session import _build_allocator_class

    allocator = _build_allocator_class(trt, torch)()

    first = allocator._reallocate("out", 1024)
    assert allocator._reallocate("out", 512) == first    # smaller request: same memory
    assert allocator._reallocate("out", 1024) == first   # exact fit: same memory

    grown = allocator._reallocate("out", 4096)
    assert allocator.buffers["out"].numel() >= 4096
    assert allocator._reallocate("out", 2048) == grown   # and it never shrinks back

    # A null pointer is how this API reports failure, so even a 0-byte request has to
    # come back with real memory behind it.
    assert allocator._reallocate("empty", 0) != 0


def test_the_allocator_is_built_once_and_its_buffers_reused(model, batch, profile):
    """One allocator for the context's life, handing TensorRT the same addresses every
    call. TensorRT's bindings never release a registered allocator, so building one per
    call pins its buffers for the process's lifetime."""
    warm = torch.from_numpy(batch(profile)).cuda()

    model.run_torch(warm)
    allocator = model._allocator
    addresses = {name: buf.data_ptr() for name, buf in allocator.buffers.items()}
    assert addresses, "the engine's outputs never went through the allocator"

    model.run_torch(warm)

    assert model._allocator is allocator
    assert {name: buf.data_ptr() for name, buf in allocator.buffers.items()} == addresses


def test_repeated_inference_stops_allocating_once_the_buffers_are_warm(model, batch, profile):
    """The regression gate: live GPU memory must be flat across a run of calls, not
    climbing by one set of output buffers each time.

    Exact equality, not a tolerance — every allocation ``run_torch`` makes in steady
    state is freed when the call's results are dropped, so a single retained byte here
    is the bug. Pre-fix this grew by 4 buffers per call and wedged the process at ~9GiB
    after ~18h of continuous inference.
    """
    warm = torch.from_numpy(batch(profile)).cuda()
    for _ in range(3):
        model.run_torch(warm)  # let every output buffer reach its final size
    torch.cuda.synchronize()

    baseline = torch.cuda.memory_allocated()
    for _ in range(50):
        model.run_torch(warm)
    torch.cuda.synchronize()

    assert torch.cuda.memory_allocated() == baseline


def test_a_later_call_does_not_overwrite_an_earlier_calls_results(model, batch, profile):
    """What reusing the buffers costs, and why ``run_torch`` copies out of them.

    ``TrtAdapter.predict`` groups images by input shape, runs one batch per group, and
    postprocesses only after every group has finished — so it genuinely holds one
    call's results while the next call runs. Returning views into the reused buffers
    would silently hand the earlier groups the last group's detections.

    Compares the first call against its own snapshot, so the engine's non-determinism
    across invocations (see ``conftest``) doesn't come into it.
    """
    first = model.run_torch(torch.from_numpy(batch(2, seed=1)).cuda())
    snapshot = [[tensor.clone() for tensor in triple] for triple in first]

    model.run_torch(torch.from_numpy(batch(profile, seed=2)).cuda())

    for triple, expected in zip(first, snapshot):
        for tensor, want in zip(triple, expected):
            torch.testing.assert_close(tensor, want)
