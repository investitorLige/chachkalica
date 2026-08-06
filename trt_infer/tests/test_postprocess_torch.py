"""``to_friendy_torch`` / ``adapt_outputs_torch`` must reproduce their numpy twins.

Bit-exact, for the same reason as ``test_preprocess_torch``: this is the second half
of the claim that decoding engine outputs on the GPU cannot move a bundle's
detections.

The CUDA cases are not redundant with the CPU ones. Torch lowers
``tensor / python_scalar`` into a multiply by the reciprocal *on CUDA only*, which
diverges from numpy's true division by 1-2 ULP — that is why
``postprocess_torch._divisors`` materializes the divisors as tensors, and this test
is what pins that down.

Runs on CPU (and additionally on CUDA when one is present); no TensorRT needed.
"""

import inspect
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

torch = pytest.importorskip("torch")

from onnx_infer.arch import ARCH_REGISTRY, get_handler  # noqa: E402
from onnx_infer.arch.base import PassthroughHandler  # noqa: E402
from onnx_infer.postprocess import FRIENDY_COLUMNS, to_friendy  # noqa: E402
from onnx_infer.preprocess import Transform  # noqa: E402
from trt_infer.postprocess_torch import adapt_outputs_torch, to_friendy_torch  # noqa: E402

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

# scale_x/scale_y that are not exactly representable in float32 are the interesting
# ones — they are where a reciprocal-multiply diverges from a true divide.
TRANSFORMS = [
    Transform(1.0, 1.0, 0, 0, 640, 480),
    Transform(0.5, 0.5, 0, 0, 1280, 960),
    Transform(960 / 1920, 960 / 1080, 0, 0, 1920, 1080),
    Transform(0.3333333333333333, 0.7777777777777778, 0, 0, 333, 777),
    Transform(2.5, 3.125, 0, 0, 256, 128),
]


def _sample(rng, n):
    boxes = (rng.random((n, 4), dtype=np.float32) * 1200.0 - 100.0).astype(np.float32)
    if n:
        # Half well-formed (x2 > x1), half left degenerate on purpose.
        half = n // 2
        boxes[:half, 2:] = boxes[:half, :2] + rng.random((half, 2), dtype=np.float32) * 300.0
    return (
        boxes,
        rng.random(n, dtype=np.float32).astype(np.float32),
        rng.integers(0, 5, size=n).astype(np.int32),
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("box_coords", ["input_pixels", "input_normalized"])
def test_matches_numpy(device, box_coords):
    rng = np.random.default_rng(1234)
    for n, transform, threshold, clip in itertools.product(
        [0, 1, 7, 300], TRANSFORMS, [0.0, 0.05, 0.5, 2.0], [False, True]
    ):
        boxes, scores, labels = _sample(rng, n)
        reference = to_friendy(
            boxes, scores, labels, transform, threshold,
            clip_boxes=clip, box_coords=box_coords,
        )
        got = to_friendy_torch(
            torch.from_numpy(boxes).to(device),
            torch.from_numpy(scores).to(device),
            torch.from_numpy(labels).to(device),
            transform, threshold, clip_boxes=clip, box_coords=box_coords,
        )
        label = f"n={n} scale_x={transform.scale_x} t={threshold} clip={clip} {box_coords}"
        assert got.device.type == device, label
        assert tuple(got.shape) == reference.shape, f"{label}: {tuple(got.shape)} != {reference.shape}"
        np.testing.assert_array_equal(got.cpu().numpy(), reference, err_msg=label)


@pytest.mark.parametrize("device", DEVICES)
def test_empty_input_keeps_the_column_count_and_device(device):
    empty = to_friendy_torch(
        torch.zeros(0, 4, device=device),
        torch.zeros(0, device=device),
        torch.zeros(0, dtype=torch.int64, device=device),
        TRANSFORMS[0], 0.0,
    )
    assert tuple(empty.shape) == (0, len(FRIENDY_COLUMNS))
    assert empty.device.type == device


def test_shares_the_column_definition_rather_than_restating_it():
    import trt_infer.postprocess_torch as postprocess_torch
    import onnx_infer.postprocess as numpy_postprocess

    assert postprocess_torch.FRIENDY_COLUMNS is numpy_postprocess.FRIENDY_COLUMNS


def test_signatures_stay_in_step():
    numpy_sig = inspect.signature(to_friendy)
    torch_sig = inspect.signature(to_friendy_torch)
    assert [
        (p.name, p.kind, p.default) for p in numpy_sig.parameters.values()
    ] == [(p.name, p.kind, p.default) for p in torch_sig.parameters.values()]


@pytest.mark.parametrize("arch", sorted(ARCH_REGISTRY))
def test_every_registered_arch_has_a_torch_twin(arch):
    """Adding a non-passthrough arch should fail here, not silently take the slow path."""
    adapted = adapt_outputs_torch(
        get_handler(arch),
        [torch.zeros(4, 4), torch.zeros(4), torch.zeros(4, dtype=torch.int32)],
    )
    assert adapted is not None, f"{arch} needs a torch adapt_outputs twin"
    boxes, scores, labels = adapted
    assert boxes.shape == (4, 4) and scores.shape == (4,) and labels.shape == (4,)
    assert labels.dtype == torch.int64


def test_declines_a_handler_that_overrides_adapt_outputs():
    class Custom(PassthroughHandler):
        name = "custom"

        def adapt_outputs(self, outputs):
            return outputs[0], outputs[1], outputs[2]

    assert adapt_outputs_torch(Custom(), [None, None, None]) is None


def test_label_cast_truncates_like_numpy():
    values = np.array([-1.9, -0.4, 0.4, 2.9], dtype=np.float32)
    _, _, labels = adapt_outputs_torch(
        get_handler("yolox"),
        [torch.zeros(4, 4), torch.zeros(4), torch.from_numpy(values)],
    )
    np.testing.assert_array_equal(labels.numpy(), values.astype(np.int64))


def test_rejects_too_few_graph_outputs():
    with pytest.raises(ValueError, match="expected 3 graph outputs"):
        adapt_outputs_torch(get_handler("yolox"), [torch.zeros(4, 4), torch.zeros(4)])
