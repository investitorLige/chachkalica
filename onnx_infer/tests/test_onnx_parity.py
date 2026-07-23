"""End-to-end parity: the torch training adapter vs the ONNX service adapter.

The gate for every architecture. Builds an adapter, exports it, then asserts the
:class:`OnnxAdapter` reproduces the adapter's ``predict`` output (boxes/scores
within tolerance, labels exact) on the same images.

Skips cleanly when torch / torchvision / onnx / onnxruntime aren't installed, so
the pure-numpy tests still run in a minimal environment.
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
pytest.importorskip("onnxruntime")

from friendy_chachkalica.registry import build_model  # noqa: E402
from friendy_chachkalica.ml.onnx_export.arch.fasterrcnn import export_fasterrcnn  # noqa: E402
from friendy_chachkalica.ml.onnx_export.arch.retinanet import export_retinanet  # noqa: E402
from friendy_chachkalica.ml.onnx_export.arch.yolox import export_yolox  # noqa: E402
from friendy_chachkalica.ml.onnx_export.arch.rtdetr import export_rtdetr  # noqa: E402
from friendy_chachkalica.ml.onnx_export.arch.rfdetr import export_rfdetr  # noqa: E402
from onnx_infer import load_onnx_adapter  # noqa: E402


def _assert_parity(torch_pred, onnx_pred, min_dets=1, atol=1e-3):
    """Every torch detection has a unique ONNX twin (same label, box+score within
    ``atol``), and counts match. Order-independent: the two runners can emit the
    same set in a different row order, and near-tied scores make row-align
    unreliable, so we greedily nearest-neighbour match instead.
    """
    torch_pred = np.asarray(torch_pred, dtype=np.float32)
    onnx_pred = np.asarray(onnx_pred, dtype=np.float32)
    assert torch_pred.shape[0] == onnx_pred.shape[0], (
        f"detection count differs: torch={torch_pred.shape[0]} onnx={onnx_pred.shape[0]}"
    )
    assert torch_pred.shape[0] >= min_dets, (
        f"trivial comparison: only {torch_pred.shape[0]} detections (expected >= {min_dets})"
    )

    used = set()
    for row in torch_pred:
        best_j, best_d = None, np.inf
        for j, cand in enumerate(onnx_pred):
            if j in used or int(cand[5]) != int(row[5]):
                continue
            dist = np.abs(row[:5] - cand[:5]).sum()  # box (4) + score
            if dist < best_d:
                best_d, best_j = dist, j
        assert best_j is not None, f"no ONNX match for torch detection {row}"
        assert best_d <= atol, f"closest ONNX match off by L1 {best_d:.2e} (> {atol}) for {row}"
        used.add(best_j)


@pytest.fixture(scope="module")
def retinanet_export(tmp_path_factory):
    torch.manual_seed(0)
    adapter = build_model("retinanet", num_classes=3)  # random weights, no download
    # RetinaNet's classification head is prior-init'd to ~0.01 scores, so random
    # weights would emit zero detections above the score floor — a trivial
    # (0 == 0) comparison. Widen the head bias so the model actually fires,
    # giving a real multi-detection set to check parity against.
    torch.nn.init.normal_(adapter.model.head.classification_head.cls_logits.bias, mean=-2.0, std=2.0)
    adapter.eval()
    out_dir = tmp_path_factory.mktemp("retinanet")
    onnx_path = out_dir / "model.onnx"
    meta = export_retinanet(
        adapter, num_classes=3, params={},
        class_map={0: "a", 1: "b", 2: "c"}, onnx_path=onnx_path,
    )
    onnx_path.with_suffix(".meta.json").write_text(json.dumps(meta))
    return adapter, onnx_path


@pytest.mark.parametrize("hw", [(480, 640), (512, 512)])
def test_retinanet_parity(retinanet_export, hw):
    adapter, onnx_path = retinanet_export
    onnx_adapter, info = load_onnx_adapter(onnx_path, "cpu")
    assert info["num_classes"] == 3
    assert info["train_classes"] == {0: "a", 1: "b", 2: "c"}

    torch.manual_seed(1)
    image = torch.rand(3, *hw)
    threshold = 0.05
    torch_pred = adapter.predict([image])[0].detach().cpu().numpy()
    torch_pred = torch_pred[torch_pred[:, 4] >= threshold]  # match the graph's score floor
    onnx_pred = onnx_adapter.predict([image], score_threshold=threshold)[0].detach().cpu().numpy()

    _assert_parity(torch_pred, onnx_pred, min_dets=10)


# --------------------------------------------------------------------------- Faster R-CNN


@pytest.fixture(scope="module")
def fasterrcnn_export(tmp_path_factory):
    torch.manual_seed(0)
    # From scratch, offline, smallest/fastest variant. Unlike RetinaNet/YOLOX,
    # torchvision's FastRCNNPredictor has no prior-probability bias init (plain
    # nn.Linear default init), so a random-init model's box-head softmax scores
    # land near-uniform across classes — well above any real score floor. No head
    # surgery needed to get a non-trivial multi-detection set.
    adapter = build_model(
        "fasterrcnn", num_classes=3, variant="mobilenet_v3_large_320_fpn", weights=False,
    )
    adapter.eval()
    out_dir = tmp_path_factory.mktemp("fasterrcnn")
    onnx_path = out_dir / "model.onnx"
    meta = export_fasterrcnn(
        adapter, num_classes=3, params={},
        class_map={0: "a", 1: "b", 2: "c"}, onnx_path=onnx_path,
    )
    onnx_path.with_suffix(".meta.json").write_text(json.dumps(meta))
    return adapter, onnx_path


@pytest.mark.parametrize("hw", [(480, 640), (512, 512)])
def test_fasterrcnn_parity(fasterrcnn_export, hw):
    adapter, onnx_path = fasterrcnn_export
    onnx_adapter, info = load_onnx_adapter(onnx_path, "cpu")
    assert info["num_classes"] == 3
    assert info["train_classes"] == {0: "a", 1: "b", 2: "c"}

    torch.manual_seed(1)
    image = torch.rand(3, *hw)
    threshold = 0.05
    torch_pred = adapter.predict([image])[0].detach().cpu().numpy()
    torch_pred = torch_pred[torch_pred[:, 4] >= threshold]  # match the graph's score floor
    onnx_pred = onnx_adapter.predict([image], score_threshold=threshold)[0].detach().cpu().numpy()

    _assert_parity(torch_pred, onnx_pred, min_dets=5)


# --------------------------------------------------------------------------- YOLOX


@pytest.fixture(scope="module")
def yolox_export(tmp_path_factory):
    torch.manual_seed(0)
    adapter = build_model("yolox", num_classes=3, variant="yolox-nano")  # random init
    # YOLOX inits its obj/cls heads with a prior-prob bias (~0.01), so a random
    # model emits zero detections above any real floor — a trivial (0 == 0)
    # comparison. Bias the obj heads high (fire everywhere) and give the cls heads
    # spread so argmax varies, yielding a rich multi-detection + real-NMS set.
    for obj_conv in adapter.model.head.obj_preds:
        torch.nn.init.constant_(obj_conv.bias, 2.0)  # sigmoid(2) ~ 0.88
    for cls_conv in adapter.model.head.cls_preds:
        torch.nn.init.normal_(cls_conv.bias, mean=0.0, std=2.0)
    adapter.score_threshold = 0.05
    adapter.eval()
    out_dir = tmp_path_factory.mktemp("yolox")
    onnx_path = out_dir / "model.onnx"
    meta = export_yolox(
        adapter, num_classes=3, params={},
        class_map={0: "a", 1: "b", 2: "c"}, onnx_path=onnx_path,
    )
    onnx_path.with_suffix(".meta.json").write_text(json.dumps(meta))
    return adapter, onnx_path


@pytest.mark.parametrize("hw", [(640, 640), (480, 640), (512, 512)])
def test_yolox_parity(yolox_export, hw):
    adapter, onnx_path = yolox_export
    onnx_adapter, info = load_onnx_adapter(onnx_path, "cpu")
    assert info["num_classes"] == 3

    torch.manual_seed(1)
    image = torch.rand(3, *hw)
    threshold = 0.05
    torch_pred = adapter.predict([image], score_threshold=threshold)[0].detach().cpu().numpy()
    onnx_pred = onnx_adapter.predict([image], score_threshold=threshold)[0].detach().cpu().numpy()

    _assert_parity(torch_pred, onnx_pred, min_dets=5)


# --------------------------------------------------------------------------- RT-DETR


@pytest.fixture(scope="module")
def rtdetr_export(tmp_path_factory):
    pytest.importorskip("transformers")
    torch.manual_seed(0)
    adapter = build_model("rtdetr", num_classes=3, weights=None)  # from scratch, offline
    # A random-init RT-DETR has an untrained `enc_score_head`, so every one of the
    # ~8400 encoder proposals scores the identical constant ln(1/num_classes). The
    # encoder's query selection — `topk(enc_outputs_class.max(-1), num_queries)` —
    # is then a topk over a fully-tied field, which torch and onnxruntime break
    # differently: they select *different* (equally valid) query sets, cascading
    # into completely scrambled decoder boxes. That is tie ambiguity, not an export
    # defect (the graph is bit-identical up to the topk — verified). So we spread
    # both heads just enough to de-tie the scores without saturating sigmoid (a
    # 256-dim dot product amplifies the weight std ~16x, so keep std small): the
    # encoder topk and the final scores become well-separated and backend-stable,
    # giving a genuine multi-detection parity check. Mirrors the retinanet/yolox
    # head surgery above.
    inner = adapter.model.model
    torch.nn.init.normal_(inner.enc_score_head.weight, mean=0.0, std=0.2)
    torch.nn.init.constant_(inner.enc_score_head.bias, 0.0)
    torch.nn.init.normal_(inner.decoder.class_embed[-1].weight, mean=0.0, std=0.15)
    torch.nn.init.constant_(inner.decoder.class_embed[-1].bias, -2.0)
    adapter.eval()
    out_dir = tmp_path_factory.mktemp("rtdetr")
    onnx_path = out_dir / "model.onnx"
    meta = export_rtdetr(
        adapter, num_classes=3, params={},
        class_map={0: "a", 1: "b", 2: "c"}, onnx_path=onnx_path,
    )
    onnx_path.with_suffix(".meta.json").write_text(json.dumps(meta))
    return adapter, onnx_path


@pytest.mark.parametrize(
    "hw",
    [
        (512, 512),   # <= max_size, multiple of 32: no resize, no pad (byte-identical input)
        (480, 640),   # ditto
        (704, 512),   # longest > 640: exercises longest-side resize + pad-to-32
    ],
)
def test_rtdetr_parity(rtdetr_export, hw):
    adapter, onnx_path = rtdetr_export
    onnx_adapter, info = load_onnx_adapter(onnx_path, "cpu")
    assert info["num_classes"] == 3

    torch.manual_seed(1)
    image = torch.rand(3, *hw)
    threshold = 0.5
    torch_pred = adapter.predict([image], score_threshold=threshold)[0].detach().cpu().numpy()
    onnx_pred = onnx_adapter.predict([image], score_threshold=threshold)[0].detach().cpu().numpy()

    # RT-DETR is NMS-free and returns normalized boxes; a resized case leans on the
    # service's resize matching torch's F.interpolate, so allow a slightly looser tol.
    atol = 1e-3 if hw[0] <= 640 and hw[1] <= 640 else 5e-3
    _assert_parity(torch_pred, onnx_pred, min_dets=5, atol=atol)


# --------------------------------------------------------------------------- RF-DETR


@pytest.fixture(scope="module")
def rfdetr_export(tmp_path_factory):
    pytest.importorskip("rfdetr")
    torch.manual_seed(0)
    # From scratch, offline, smallest variant at a small resolution for speed.
    # Unlike RT-DETR, a random-init RF-DETR does NOT hit topk-tie degeneracy: its
    # class head is prior-bias initialized and the box/class heads are continuous,
    # so the (internal two-stage + final) top-k selections are well-separated and
    # backend-stable. No head surgery needed.
    adapter = build_model("rfdetr", num_classes=3, variant="nano", weights=False, resolution=224)
    adapter.eval()
    out_dir = tmp_path_factory.mktemp("rfdetr")
    onnx_path = out_dir / "model.onnx"
    meta = export_rfdetr(
        adapter, num_classes=3, params={},
        class_map={0: "a", 1: "b", 2: "c"}, onnx_path=onnx_path,
    )
    onnx_path.with_suffix(".meta.json").write_text(json.dumps(meta))
    return adapter, onnx_path


@pytest.mark.parametrize("hw", [(480, 640), (512, 512), (720, 480)])
def test_rfdetr_parity(rfdetr_export, hw):
    adapter, onnx_path = rfdetr_export
    onnx_adapter, info = load_onnx_adapter(onnx_path, "cpu")
    assert info["num_classes"] == 3

    torch.manual_seed(1)
    image = torch.rand(3, *hw)
    # A random-init RF-DETR's prior-biased head keeps every score well below any
    # real floor, so a positive threshold yields zero detections (a trivial
    # comparison). Compare the full post-top-k set (threshold 0) instead — a rich,
    # non-trivial check of box decode + top-k + background drop + square resize.
    threshold = 0.0
    torch_pred = adapter.predict([image], score_threshold=threshold)[0].detach().cpu().numpy()
    onnx_pred = onnx_adapter.predict([image], score_threshold=threshold)[0].detach().cpu().numpy()

    # RF-DETR always applies an aspect-changing square resize, so parity leans on
    # the service's numpy resize matching torch's bilinear F.interpolate; allow the
    # same looser tol as the RT-DETR resized cases.
    _assert_parity(torch_pred, onnx_pred, min_dets=20, atol=5e-3)
