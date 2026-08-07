"""The gate on the GPU-resident ``TrtAdapter.predict``.

``test_preprocess_torch`` and ``test_postprocess_torch`` prove each half matches its
numpy twin exactly, in isolation. This file closes the loop on the assembled adapter,
in two layers because the synthetic fixture engine is not deterministic across calls
(see ``conftest``):

* **Decode parity** — run the engine *once*, then decode that single captured output
  both ways (torch and numpy) and require bit-equality. Deterministic by construction,
  and it isolates exactly the code this change introduced.
* **End-to-end parity** — full ``predict`` against the all-numpy route
  (``preprocess`` -> ``run`` -> ``adapt_outputs`` -> ``to_friendy``), on a real trained
  bundle engine whose separated scores make selection stable. Skips when the checkout
  has no exported bundle. This is the one that speaks to "an exported bundle still
  reproduces the training repo's numbers".

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


@pytest.fixture(scope="module")
def adapter(batched_yolox_engine):
    from trt_infer import load_trt_adapter

    trt_adapter, info = load_trt_adapter(batched_yolox_engine.path, "cuda")
    assert info["num_classes"] == 3
    return trt_adapter


@pytest.fixture(scope="module")
def images(batched_yolox_engine):
    height, width = batched_yolox_engine.static_hw
    torch.manual_seed(4)
    return [torch.rand(3, height, width, device="cuda") for _ in range(3)]


def _decode_both_ways(adapter, raw, transform, score_threshold):
    """Decode one captured engine output through the torch path and the numpy path."""
    from onnx_infer.postprocess import to_friendy
    from trt_infer.postprocess_torch import adapt_outputs_torch, to_friendy_torch

    via_torch = to_friendy_torch(
        *adapt_outputs_torch(adapter._handler, raw), transform, score_threshold,
        clip_boxes=adapter.meta.clip_boxes, box_coords=adapter.meta.box_coords,
    )
    boxes, scores, labels = adapter._handler.adapt_outputs(
        [tensor.detach().cpu().numpy() for tensor in raw]
    )
    via_numpy = to_friendy(
        boxes, scores, labels, transform, score_threshold,
        clip_boxes=adapter.meta.clip_boxes, box_coords=adapter.meta.box_coords,
    )
    return via_torch.cpu().numpy(), via_numpy


@pytest.mark.parametrize("score_threshold", [0.0, 0.05, 0.5])
def test_decoding_one_captured_output_matches_numpy(adapter, images, score_threshold):
    from trt_infer.preprocess_torch import preprocess_torch

    prepared = [preprocess_torch(image, adapter.meta) for image in images]
    raws = adapter._model.run_torch(torch.cat([batched for batched, _ in prepared], dim=0))

    for i, (raw, (_batched, transform)) in enumerate(zip(raws, prepared)):
        via_torch, via_numpy = _decode_both_ways(adapter, raw, transform, score_threshold)
        assert via_torch.shape == via_numpy.shape, f"image {i}"
        np.testing.assert_array_equal(via_torch, via_numpy, err_msg=f"image {i}")


def test_predictions_come_back_on_the_cpu(adapter, images):
    """Matches OnnxAdapter and the torch adapters. Everything downstream
    (merge_predictions' hand-rolled NMS, the detector's class filter) is a Python
    loop over tiny tensors that is faster on the host than issuing CUDA syncs."""
    predictions = adapter.predict(images)

    assert len(predictions) == len(images)
    for prediction in predictions:
        assert not prediction.is_cuda
        assert prediction.dtype == torch.float32
        assert prediction.ndim == 2 and prediction.shape[1] == 6


def test_accepts_cpu_tensors_numpy_and_non_contiguous_crops(adapter, images):
    """The three input forms callers actually supply: a GPU crop (a strided view of
    the frame), a host tensor, and a raw ndarray."""
    crop = images[0][:, 13:140, 7:96]
    assert not crop.is_contiguous()

    for label, supplied in (
        ("cuda crop", [crop]),
        ("cpu tensor", [images[0].cpu()]),
        ("numpy", [images[0].cpu().numpy()]),
    ):
        prediction = adapter.predict(supplied, score_threshold=0.05)[0]
        assert prediction.ndim == 2 and prediction.shape[1] == 6, label
        assert not prediction.is_cuda, label


def test_empty_input(adapter):
    assert adapter.predict([]) == []


def test_grouping_beyond_the_batch_profile_is_rejected(adapter, images, batched_yolox_engine):
    """`predict` groups same-shaped images into one engine call, so a caller handing it
    more than the profile allows must get an error rather than duplicated detections."""
    too_many = [images[0]] * (batched_yolox_engine.batch_profile + 1)
    with pytest.raises(ValueError, match="batch profile of at most"):
        adapter.predict(too_many)


# --------------------------------------------------- end-to-end, on trained engines


def _numpy_route(adapter, image, score_threshold):
    """What ``predict`` looked like before it moved to the GPU."""
    from onnx_infer.postprocess import to_friendy
    from onnx_infer.preprocess import preprocess

    batched, transform = preprocess(
        np.asarray(image.detach().cpu().numpy(), dtype=np.float32), adapter.meta
    )
    boxes, scores, labels = adapter._handler.adapt_outputs(adapter._model.run(batched))
    return to_friendy(
        boxes, scores, labels, transform, score_threshold,
        clip_boxes=adapter.meta.clip_boxes, box_coords=adapter.meta.box_coords,
    )


def test_predict_matches_the_numpy_route_on_a_trained_engine(real_bundle_engines):
    """The gate that speaks to production: a real bundle's engines, whose trained
    weights separate the scores enough for selection to be stable."""
    from trt_infer import load_trt_adapter

    torch.manual_seed(0)
    checked = 0
    for engine in real_bundle_engines:
        adapter, _ = load_trt_adapter(engine, "cuda")
        frame = torch.rand(3, 720, 1280, device="cuda")
        # A whole frame plus person-crop-shaped strided views of it.
        cases = [frame, frame[:, :360, :420], frame[:, 90:313, 200:297], frame[:, :17, :5]]

        for i, image in enumerate(cases):
            for score_threshold in (0.0, 0.25):
                got = adapter.predict([image], score_threshold=score_threshold)[0]
                expected = _numpy_route(adapter, image, score_threshold)
                tag = f"{engine.parent.parent.name}/{engine.stem} case {i} @{score_threshold}"
                assert tuple(got.shape) == expected.shape, (
                    f"{tag}: {got.shape[0]} detections vs {expected.shape[0]}"
                )
                np.testing.assert_array_equal(got.numpy(), expected, err_msg=tag)
                checked += 1

    assert checked, "no engines exercised"
