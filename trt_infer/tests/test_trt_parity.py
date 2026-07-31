"""End-to-end parity: the torch training adapter vs the TensorRT runtime adapter.

The gate for every architecture. Exports the adapter to ONNX (reusing the ONNX
exporters), compiles it to a TensorRT engine (``trt_export``), then asserts the
:class:`TrtAdapter` reproduces the torch adapter's confident detections.

All registered archs build and run:

* **rtdetr, rfdetr** — compile straight from the standard ONNX (fixed-size DETR
  top-k).
* **retinanet, yolox** — their standard ONNX bakes data-dependent NMS that TRT
  can't compile, so ``trt_export`` re-exports a raw-output graph and appends the
  ``EfficientNMS_TRT`` plugin; the runtime auto-detects the plugin outputs.
* **fasterrcnn** — two-stage; ``trt_export`` re-exports a raw-output graph with a
  *fixed top-K* RPN (no proposal NMS, so RoIAlign sees a static count) and
  per-class boxes, then appends ``EfficientNMS_TRT``. Because the fixed-K RPN
  makes the engine an approximation of the torch model (and this fixture is
  random-init, so all scores tie), its test gates on the engine building, loading,
  and emitting structurally valid detections — not bit-exact box parity.

We compare the **confident top-K** detections: every DETR-family top-k (and the
EfficientNMS candidate sort) has a low-confidence tail whose near-ties break
differently across backends (a model property, not a runtime bug), so the tail is
excluded from the gate. Tight numeric parity on the confident set is what matters
and is what real trained checkpoints exercise (validated to ~1e-3–4e-3).

Requires a CUDA GPU + TensorRT; skips cleanly otherwise.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("onnx")
pytest.importorskip("tensorrt")

if not torch.cuda.is_available():
    pytest.skip("TensorRT parity needs a CUDA GPU", allow_module_level=True)

from friendy_chachkalica.registry import build_model  # noqa: E402
from friendy_chachkalica.ml.onnx_export.arch.fasterrcnn import export_fasterrcnn  # noqa: E402
from friendy_chachkalica.ml.onnx_export.arch.retinanet import export_retinanet  # noqa: E402
from friendy_chachkalica.ml.onnx_export.arch.yolox import export_yolox  # noqa: E402
from friendy_chachkalica.ml.onnx_export.arch.rtdetr import export_rtdetr  # noqa: E402
from friendy_chachkalica.ml.onnx_export.arch.rfdetr import export_rfdetr  # noqa: E402
from friendy_chachkalica.ml.trt_export.cli import build_engine  # noqa: E402
from trt_infer import load_trt_adapter  # noqa: E402


def _assert_topk_parity(torch_pred, trt_pred, k=10, atol=5e-3):
    """The K highest-confidence detections agree (label + box/score within atol)."""
    torch_pred = np.asarray(torch_pred, dtype=np.float32)
    trt_pred = np.asarray(trt_pred, dtype=np.float32)
    n = min(k, torch_pred.shape[0], trt_pred.shape[0])
    assert n >= 1, f"nothing to compare: torch={torch_pred.shape[0]} trt={trt_pred.shape[0]}"

    top_t = torch_pred[np.argsort(-torch_pred[:, 4])][:n]
    top_x = trt_pred[np.argsort(-trt_pred[:, 4])]
    used = set()
    for row in top_t:
        best_j, best_d = None, np.inf
        for j, cand in enumerate(top_x):
            if j in used or int(cand[5]) != int(row[5]):
                continue
            dist = float(np.abs(row[:5] - cand[:5]).sum())
            if dist < best_d:
                best_d, best_j = dist, j
        assert best_j is not None, f"no TRT twin for confident detection {row}"
        assert best_d <= atol, f"confident detection off by L1 {best_d:.2e} (> {atol}): {row}"
        used.add(best_j)


def _export(adapter, exporter, out_dir, num_classes=3):
    onnx_path = out_dir / "model.onnx"
    meta = exporter(
        adapter, num_classes=num_classes, params={},
        class_map={i: chr(ord("a") + i) for i in range(num_classes)},
        onnx_path=onnx_path,
    )
    onnx_path.with_suffix(".meta.json").write_text(json.dumps(meta))
    return onnx_path


def _build(onnx_path, out_dir, *, adapter, static_hw):
    """Build an FP32 engine at a fixed profile (adapter drives the EfficientNMS archs)."""
    return build_engine(
        onnx_path, out_dir / "model.engine", precision="fp32", adapter=adapter,
        min_hw=static_hw, opt_hw=static_hw, max_hw=static_hw, workspace_gb=2.0,
    )


# --------------------------------------------------------------------------- RetinaNet


@pytest.fixture(scope="module")
def retinanet_engine(tmp_path_factory):
    torch.manual_seed(0)
    adapter = build_model("retinanet", num_classes=3)
    torch.nn.init.normal_(adapter.model.head.classification_head.cls_logits.bias, mean=-2.0, std=2.0)
    adapter.eval()
    out_dir = tmp_path_factory.mktemp("retinanet")
    onnx_path = _export(adapter, export_retinanet, out_dir)
    return adapter, _build(onnx_path, out_dir, adapter=adapter, static_hw=(512, 512))


def test_retinanet_parity(retinanet_engine):
    adapter, engine = retinanet_engine
    trt_adapter, info = load_trt_adapter(engine, "cuda")
    assert info["num_classes"] == 3
    torch.manual_seed(1)
    image = torch.rand(3, 512, 512)
    torch_pred = adapter.predict([image])[0].detach().cpu().numpy()
    torch_pred = torch_pred[torch_pred[:, 4] >= 0.05]
    trt_pred = trt_adapter.predict([image], score_threshold=0.05)[0].detach().cpu().numpy()
    # Looser tol on this synthetic random-init fixture: torchvision's per-level
    # top-1000 candidate pre-selection (which EfficientNMS doesn't replicate)
    # picks slightly different boxes among the many near-tied scores of an
    # untrained head. Real trained checkpoints match to ~1e-4 (no near-ties).
    _assert_topk_parity(torch_pred, trt_pred, k=5, atol=5e-2)


# --------------------------------------------------------------------------- YOLOX


@pytest.fixture(scope="module")
def yolox_engine(tmp_path_factory):
    torch.manual_seed(0)
    adapter = build_model("yolox", num_classes=3, variant="yolox-nano")
    for obj_conv in adapter.model.head.obj_preds:
        torch.nn.init.constant_(obj_conv.bias, 2.0)
    for cls_conv in adapter.model.head.cls_preds:
        torch.nn.init.normal_(cls_conv.bias, mean=0.0, std=2.0)
    adapter.score_threshold = 0.05
    adapter.eval()
    out_dir = tmp_path_factory.mktemp("yolox")
    onnx_path = _export(adapter, export_yolox, out_dir)
    return adapter, _build(onnx_path, out_dir, adapter=adapter, static_hw=(512, 512))


def test_yolox_parity(yolox_engine):
    adapter, engine = yolox_engine
    trt_adapter, info = load_trt_adapter(engine, "cuda")
    assert info["num_classes"] == 3
    torch.manual_seed(1)
    image = torch.rand(3, 512, 512)
    torch_pred = adapter.predict([image], score_threshold=0.05)[0].detach().cpu().numpy()
    trt_pred = trt_adapter.predict([image], score_threshold=0.05)[0].detach().cpu().numpy()
    _assert_topk_parity(torch_pred, trt_pred, k=10, atol=5e-3)


# --------------------------------------------------------------------------- RT-DETR


@pytest.fixture(scope="module")
def rtdetr_engine(tmp_path_factory):
    pytest.importorskip("transformers")
    torch.manual_seed(0)
    # The adapter stretches every image onto a square input_max_size canvas, so
    # the engine's static profile has to be that canvas — pin it to 512 (rather
    # than the 640 default) and feed a 512 image, which keeps the resize a no-op
    # on both sides and the engine small.
    adapter = build_model("rtdetr", num_classes=3, weights=None, input_max_size=512)
    inner = adapter.model.model
    torch.nn.init.normal_(inner.enc_score_head.weight, mean=0.0, std=0.2)
    torch.nn.init.constant_(inner.enc_score_head.bias, 0.0)
    torch.nn.init.normal_(inner.decoder.class_embed[-1].weight, mean=0.0, std=0.15)
    torch.nn.init.constant_(inner.decoder.class_embed[-1].bias, -2.0)
    adapter.eval()
    out_dir = tmp_path_factory.mktemp("rtdetr")
    onnx_path = _export(adapter, export_rtdetr, out_dir)
    return adapter, _build(onnx_path, out_dir, adapter=None, static_hw=(512, 512))


def test_rtdetr_parity(rtdetr_engine):
    adapter, engine = rtdetr_engine
    trt_adapter, info = load_trt_adapter(engine, "cuda")
    assert info["num_classes"] == 3
    torch.manual_seed(1)
    image = torch.rand(3, 512, 512)
    torch_pred = adapter.predict([image], score_threshold=0.0)[0].detach().cpu().numpy()
    trt_pred = trt_adapter.predict([image], score_threshold=0.0)[0].detach().cpu().numpy()
    _assert_topk_parity(torch_pred, trt_pred, k=10, atol=5e-3)


# --------------------------------------------------------------------------- RF-DETR


@pytest.fixture(scope="module")
def rfdetr_engine(tmp_path_factory):
    pytest.importorskip("rfdetr")
    torch.manual_seed(0)
    adapter = build_model("rfdetr", num_classes=3, variant="nano", weights=False, resolution=224)
    adapter.eval()
    out_dir = tmp_path_factory.mktemp("rfdetr")
    onnx_path = _export(adapter, export_rfdetr, out_dir)
    return adapter, _build(onnx_path, out_dir, adapter=None, static_hw=(224, 224))


def test_rfdetr_parity(rfdetr_engine):
    adapter, engine = rfdetr_engine
    trt_adapter, info = load_trt_adapter(engine, "cuda")
    assert info["num_classes"] == 3
    torch.manual_seed(1)
    image = torch.rand(3, 224, 224)
    torch_pred = adapter.predict([image], score_threshold=0.0)[0].detach().cpu().numpy()
    trt_pred = trt_adapter.predict([image], score_threshold=0.0)[0].detach().cpu().numpy()
    _assert_topk_parity(torch_pred, trt_pred, k=10, atol=5e-3)


# --------------------------------------------------------------------------- Faster R-CNN


def test_fasterrcnn_engine(tmp_path):
    """Build + run gate for the two-stage EfficientNMS export.

    The engine approximates the torch model (fixed top-K RPN replaces the variable
    proposal NMS — see trt_export/arch/fasterrcnn.py), and on this random-init
    fixture every class score ties, so per-box parity is meaningless. We instead
    assert the engine builds, loads, and emits structurally valid detections
    (finite, in-range boxes; known class ids; scores above the fused threshold).
    """
    torch.manual_seed(0)
    # The 320 mobilenet variant is the lightest to build/export/compile.
    adapter = build_model("fasterrcnn", num_classes=3, variant="mobilenet_v3_large_320_fpn")
    # Suppress background (class 0) so the random-init detector emits foreground
    # boxes above threshold; otherwise there is nothing to check.
    cls_score = adapter.model.roi_heads.box_predictor.cls_score
    with torch.no_grad():
        torch.nn.init.constant_(cls_score.bias, 0.0)
        cls_score.bias[0] = -4.0
    adapter.eval()

    onnx_path = _export(adapter, export_fasterrcnn, tmp_path)
    # EfficientNMS arch: prep re-exports from the adapter, so pass it through.
    engine = _build(onnx_path, tmp_path, adapter=adapter, static_hw=(320, 320))

    trt_adapter, info = load_trt_adapter(engine, "cuda")
    assert info["num_classes"] == 3
    torch.manual_seed(1)
    image = torch.rand(3, 320, 320)
    trt_pred = trt_adapter.predict([image], score_threshold=0.05)[0].detach().cpu().numpy()

    assert trt_pred.ndim == 2 and trt_pred.shape[1] == 6
    assert trt_pred.shape[0] >= 1, "engine produced no detections"
    assert np.isfinite(trt_pred).all()
    xywh, conf, cls = trt_pred[:, :4], trt_pred[:, 4], trt_pred[:, 5]
    assert (conf >= 0.05 - 1e-4).all() and (conf <= 1.0 + 1e-4).all()
    assert set(np.unique(cls.astype(int))).issubset({0, 1, 2})
    assert (xywh[:, 2] > 0).all() and (xywh[:, 3] > 0).all()          # positive w/h
    assert (xywh[:, :2] >= -0.05).all() and (xywh[:, :2] <= 1.05).all()  # centers in-frame
