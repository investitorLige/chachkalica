"""``preprocess_torch`` must reproduce ``onnx_infer.preprocess.preprocess`` exactly.

This is the gate on the claim that moving preprocessing to the GPU cannot move an
exported bundle's detections. Equality is asserted **bit for bit**, not within a
tolerance: the torch resize is a verbatim port of the numpy gather precisely so
that it can be.

If a case here ever fails, the fix is to make the torch side match the numpy
reference — not to loosen the assertion and not to swap in ``F.interpolate``. The
numpy version is what ``onnx_infer/tests/test_onnx_parity.py``'s per-arch
tolerances were measured against.

Runs on CPU (and additionally on CUDA when one is present), so it gates in any
environment with torch — no GPU or TensorRT required.
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

from onnx_infer.meta import InputSpec, ModelMeta, Normalize  # noqa: E402
from onnx_infer.preprocess import plan_preprocess, preprocess  # noqa: E402
from trt_infer.preprocess_torch import preprocess_torch  # noqa: E402

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

IMAGENET = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

# Every resize_mode, at sizes that force upscaling, downscaling and exact matches.
MODES = [
    dict(resize_mode="none"),
    dict(resize_mode="square", size=64),
    dict(resize_mode="square", size=960),
    dict(resize_mode="longest_side", max_size=64),
    dict(resize_mode="longest_side", max_size=4096),  # never upscales: a no-op resize
    dict(resize_mode="letterbox", size=64),
    dict(resize_mode="letterbox", size=640),
    dict(resize_mode="letterbox", size=960),
]

SHAPES = [
    (3, 40, 50),      # plain downscale/upscale
    (3, 50, 50),      # square source
    (3, 200, 100),    # tall
    (3, 100, 200),    # wide
    (3, 1, 7),        # single row
    (3, 7, 1),        # single column
    (3, 1, 1),        # single pixel
    (3, 33, 17),      # odd on both axes
    (3, 640, 640),    # already at a canvas size
    (3, 1080, 1920),  # realistic frame
]

# The two configs the reference bundle actually ships, spelled out so a regression
# in either shows up by name rather than as one row of a cross-product.
YOLOX_PERSON = dict(
    spec=dict(resize_mode="letterbox", size=640, multiple=0, pad_value=114.0,
              input_scale="byte"),
    normalize=None,
    layout="bgr",
)
RFDETR_LARGE = dict(
    spec=dict(resize_mode="letterbox", size=960, multiple=0, pad_value=0.0,
              input_scale="unit"),
    normalize=IMAGENET,
    layout="rgb",
)


def _meta(spec_kwargs, normalize, layout):
    spec = InputSpec(**spec_kwargs)
    spec.validate()
    return ModelMeta(
        arch="test", num_classes=1, class_map={0: "x"}, score_threshold=0.0,
        input=spec, normalize=normalize, layout=layout,
    )


def _assert_identical(image, meta, device, label):
    reference, transform_ref = preprocess(image, meta)
    got, transform_got = preprocess_torch(torch.from_numpy(image).to(device), meta)

    assert transform_got == transform_ref, f"{label}: {transform_got} != {transform_ref}"
    assert tuple(got.shape) == reference.shape, f"{label}: {tuple(got.shape)} != {reference.shape}"
    assert got.dtype == torch.float32, f"{label}: dtype {got.dtype}"
    np.testing.assert_array_equal(got.cpu().numpy(), reference, err_msg=label)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", MODES, ids=lambda m: "-".join(str(v) for v in m.values()))
def test_matches_numpy_across_the_meta_surface(device, mode):
    """Full cross-product of every knob meta.json can set, against every shape."""
    rng = np.random.default_rng(0)
    for shape, scale, layout, normalize, multiple, pad in itertools.product(
        SHAPES, ["unit", "byte"], ["rgb", "bgr"], [None, IMAGENET], [0, 32], [0.0, 114.0]
    ):
        if mode["resize_mode"] == "letterbox" and multiple > 1:
            continue  # rejected by plan_preprocess; see test_letterbox_with_multiple_is_rejected
        meta = _meta(
            dict(multiple=multiple, pad_value=pad, input_scale=scale, **mode), normalize, layout
        )
        label = f"{shape} {mode} {scale} {layout} norm={normalize is not None} m={multiple} p={pad}"
        _assert_identical(rng.random(shape, dtype=np.float32), meta, device, label)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("config", [YOLOX_PERSON, RFDETR_LARGE], ids=["yolox-person", "rfdetr-large"])
def test_matches_numpy_for_the_shipped_bundle_configs(device, config):
    rng = np.random.default_rng(7)
    meta = _meta(config["spec"], config["normalize"], config["layout"])
    for shape in [(3, 1080, 1920), (3, 223, 97), (3, 960, 960), (3, 17, 5)]:
        _assert_identical(rng.random(shape, dtype=np.float32), meta, device, f"{config} {shape}")


@pytest.mark.parametrize("device", DEVICES)
def test_accepts_a_non_contiguous_crop(device):
    """Person crops arrive as strided views of the frame, never as dense buffers."""
    rng = np.random.default_rng(3)
    frame = torch.from_numpy(rng.random((3, 480, 640), dtype=np.float32)).to(device)
    crop = frame[:, 100:323, 200:297]
    assert not crop.is_contiguous()

    meta = _meta(RFDETR_LARGE["spec"], RFDETR_LARGE["normalize"], RFDETR_LARGE["layout"])
    got, _ = preprocess_torch(crop, meta)
    reference, _ = preprocess(crop.cpu().numpy(), meta)
    np.testing.assert_array_equal(got.cpu().numpy(), reference)


@pytest.mark.parametrize("device", DEVICES)
def test_rejects_a_non_chw_image(device):
    meta = _meta(dict(resize_mode="none"), None, "rgb")
    with pytest.raises(ValueError, match="CHW image with 3 channels"):
        preprocess_torch(torch.zeros(4, 10, 10, device=device), meta)


def test_letterbox_with_multiple_is_rejected():
    """The one meta combination the two pad mechanisms cannot both satisfy."""
    meta = _meta(dict(resize_mode="letterbox", size=40, multiple=32), None, "rgb")
    with pytest.raises(ValueError, match="letterbox canvas"):
        plan_preprocess(100, 100, meta)


def test_signatures_stay_in_step():
    """A kwarg added to the numpy side must be added here too, not silently skipped."""
    numpy_sig = inspect.signature(preprocess)
    torch_sig = inspect.signature(preprocess_torch)
    assert [
        (p.name, p.kind, p.default) for p in numpy_sig.parameters.values()
    ] == [(p.name, p.kind, p.default) for p in torch_sig.parameters.values()]
