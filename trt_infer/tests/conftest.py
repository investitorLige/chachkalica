"""Shared engine fixture for the GPU-resident runtime tests.

``test_trt_parity.py`` builds its own per-arch engines to gate numeric parity against
the torch adapters. The tests here gate the *runtime plumbing* instead — batching,
the numpy/torch contracts, output slicing — so they only need one engine, and it has
to be an EfficientNMS arch with a batch profile wider than 1 (the passthrough archs
are still built batch-1). YOLOX-nano is the cheapest such engine to build.

**This engine is not deterministic, and no fixture here can make it so.** A random-init
YOLOX emits a score that does not depend on the input at all — the head's stem collapses
to zero, so every one of the 1024 anchors carries the identical bias-only score — and
EfficientNMS's top-K sort over 1024 exact ties resolves differently on each invocation.
``run(x)`` twice does not agree with itself. Biasing or randomizing the head does not
help; only trained weights separate the scores. (``test_trt_parity.py`` lives with the
same property, which is why it compares only a confident top-K within a tolerance.)

Everything downstream is built around that: the tests never compare two engine
invocations. They exercise pure functions directly, decode a *single* captured
invocation two ways, or assert structure. The end-to-end exact gate runs against a real
trained bundle engine instead — see ``real_bundle_engines``.
"""

import json
import sys
from pathlib import Path
from typing import NamedTuple

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BATCH_PROFILE = 4
# Small canvas so the engine builds fast. It must be the size the *exporter* records
# in meta.json, since preprocess letterboxes onto that canvas and the engine's static
# profile has to accept exactly it — so the profile is read back from the meta below
# rather than assumed here.
CANVAS = 256


class BuiltEngine(NamedTuple):
    """The engine plus the profile it was built with.

    Carried on the fixture rather than imported as module constants: this directory
    has no ``__init__.py``, so a test module cannot import from ``conftest``.
    """

    path: Path
    batch_profile: int
    static_hw: tuple


@pytest.fixture(scope="session")
def batched_yolox_engine(tmp_path_factory) -> BuiltEngine:
    """A YOLOX engine built with a batch-4 profile."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("onnx")
    pytest.importorskip("tensorrt")
    if not torch.cuda.is_available():
        pytest.skip("the TensorRT runtime needs a CUDA GPU")

    from friendy_chachkalica.ml.onnx_export.arch.yolox import export_yolox
    from friendy_chachkalica.ml.trt_export.cli import build_engine
    from friendy_chachkalica.registry import build_model

    torch.manual_seed(0)
    adapter = build_model(
        "yolox", num_classes=3, variant="yolox-nano", input_max_size=CANVAS
    )
    # Same head biasing as test_trt_parity.py's yolox fixture, for the same reason:
    # a random-init model otherwise emits nothing to slice. See the module docstring
    # for why this still cannot produce a deterministic engine.
    for obj_conv in adapter.model.head.obj_preds:
        torch.nn.init.constant_(obj_conv.bias, 2.0)
    for cls_conv in adapter.model.head.cls_preds:
        torch.nn.init.normal_(cls_conv.bias, mean=0.0, std=2.0)
    adapter.score_threshold = 0.05
    adapter.eval()

    out_dir = tmp_path_factory.mktemp("batched_yolox")
    onnx_path = out_dir / "model.onnx"
    meta = export_yolox(
        adapter, num_classes=3, params={},
        class_map={0: "a", 1: "b", 2: "c"}, onnx_path=onnx_path,
    )
    onnx_path.with_suffix(".meta.json").write_text(json.dumps(meta))

    # The exporter owns the canvas size; the engine profile must match what preprocess
    # will actually produce, so take it from the meta rather than restating it.
    canvas = int(meta["input"]["size"])
    static_hw = (canvas, canvas)

    engine_path = build_engine(
        onnx_path, out_dir / "model.engine", precision="fp32", adapter=adapter,
        min_hw=static_hw, opt_hw=static_hw, max_hw=static_hw, workspace_gb=2.0,
        min_batch=1, opt_batch=BATCH_PROFILE, max_batch=BATCH_PROFILE,
    )
    return BuiltEngine(Path(engine_path), BATCH_PROFILE, static_hw)


@pytest.fixture(scope="session")
def real_bundle_engines():
    """Trained engines from an exported bundle, if this checkout has one.

    Trained weights separate the scores, which makes selection stable and lets the
    end-to-end ``predict`` gate assert bit-exactness — the thing the synthetic fixture
    above structurally cannot support. Skips when no bundle has been exported.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("tensorrt")
    if not torch.cuda.is_available():
        pytest.skip("the TensorRT runtime needs a CUDA GPU")

    # The repo layout, plus the trainer container's, where data/ is mounted at /app/data.
    roots = [REPO_ROOT / "chachkalica" / "data", Path("/app/data")]
    engines = sorted(
        {
            path
            for root in roots
            if root.is_dir()
            for path in root.glob("bundles/*/models/*.engine")
        }
    )
    if not engines:
        pytest.skip("no exported bundle with .engine artifacts in this checkout")
    return engines
