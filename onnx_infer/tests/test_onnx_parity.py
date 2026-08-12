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
from friendy_chachkalica.ml.onnx_export.arch.ecdet import export_ecdet  # noqa: E402
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
        (512, 512),   # square: uniform upscale to the 640 canvas
        (480, 640),   # landscape: per-axis stretch (one axis already on the canvas)
        (704, 512),   # both axes stretched, one down and one up
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

    # RT-DETR is NMS-free and returns normalized boxes. Every case stretches onto
    # the square canvas, so all of them lean on the service's bilinear resize
    # matching torch's F.interpolate — hence one uniformly looser tolerance.
    _assert_parity(torch_pred, onnx_pred, min_dets=5, atol=5e-3)


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


# --------------------------------------------------------------------------- ECDet


@pytest.fixture(scope="module")
def ecdet_export(tmp_path_factory):
    torch.manual_seed(0)
    # From scratch, offline, smallest variant at a small canvas for speed. 320 is a
    # multiple of 32, which the stride-8/16/32 encoder requires.
    adapter = build_model(
        "ecdet", num_classes=3, variant="ecdet-s", weights=False, input_max_size=320
    )
    # Like RT-DETR (and unlike RF-DETR), a random-init ECDet needs its score heads
    # de-tied: upstream zero-inits every bbox-head output layer and prior-biases the
    # class heads, so the flattened top-k over `queries * classes` runs on a nearly
    # tied field and torch/onnxruntime pick different — equally valid — rows, which
    # reads as scrambled boxes rather than the export defect it is not. Spreading
    # the per-layer heads separates the scores enough to be backend-stable.
    with torch.no_grad():
        for head in adapter.model.decoder.dec_score_head:
            if hasattr(head, "weight"):
                torch.nn.init.normal_(head.weight, mean=0.0, std=0.2)
                torch.nn.init.normal_(head.bias, mean=0.0, std=0.5)
    adapter.eval()
    out_dir = tmp_path_factory.mktemp("ecdet")
    onnx_path = out_dir / "model.onnx"
    meta = export_ecdet(
        adapter, num_classes=3, params={},
        class_map={0: "a", 1: "b", 2: "c"}, onnx_path=onnx_path,
    )
    onnx_path.with_suffix(".meta.json").write_text(json.dumps(meta))
    return adapter, onnx_path


def test_ecdet_exports_when_num_queries_is_below_the_top_k(tmp_path):
    """A config that lowers ``num_queries`` must still export.

    ``num_top_queries`` (300) normally sits below ``queries * classes``, but
    ``config_gen`` already injects ``num_queries: 25`` for rtdetr in the
    people_detect_first pipeline, and the same params on ecdet invert it
    (25 * 3 = 75 < 300). The adapter clamps; the exporter has to clamp the same way
    or ``topk`` raises mid-trace.
    """
    torch.manual_seed(0)
    adapter = build_model(
        "ecdet", num_classes=3, variant="ecdet-s", weights=False, input_max_size=320,
        ECTransformer={"num_queries": 25},
    )
    adapter.eval()
    onnx_path = tmp_path / "model.onnx"
    meta = export_ecdet(
        adapter, num_classes=3, params={},
        class_map={0: "a", 1: "b", 2: "c"}, onnx_path=onnx_path,
    )
    onnx_path.with_suffix(".meta.json").write_text(json.dumps(meta))

    onnx_adapter, _ = load_onnx_adapter(onnx_path, "cpu")
    torch.manual_seed(1)
    image = torch.rand(3, 320, 320)
    torch_pred = adapter.predict([image], score_threshold=0.0)[0].detach().cpu().numpy()
    onnx_pred = onnx_adapter.predict([image], score_threshold=0.0)[0].detach().cpu().numpy()
    assert torch_pred.shape[0] == 75, torch_pred.shape
    _assert_parity(torch_pred, onnx_pred, min_dets=20, atol=5e-3)


def test_ecdet_export_does_not_mutate_the_adapter(ecdet_export):
    """``deploy()`` folds conv blocks and swaps the heads past ``eval_idx`` for
    ``nn.Identity``, so the exporter must run on a deepcopy — otherwise exporting
    silently truncates the model the trainer would go on to use."""
    adapter, _ = ecdet_export
    heads = adapter.model.decoder.dec_score_head
    assert all(hasattr(head, "weight") for head in heads), (
        "export replaced the adapter's score heads with Identity — deploy() leaked"
    )


@pytest.mark.parametrize(
    "hw",
    [
        (320, 320),  # already on the canvas: no resize at all
        (240, 320),  # landscape: per-axis stretch (one axis already on the canvas)
        (256, 256),  # square: uniform upscale
        (400, 300),  # both axes stretched, one down and one up
    ],
)
def test_ecdet_parity(ecdet_export, hw):
    adapter, onnx_path = ecdet_export
    onnx_adapter, info = load_onnx_adapter(onnx_path, "cpu")
    assert info["num_classes"] == 3

    torch.manual_seed(1)
    image = torch.rand(3, *hw)
    # Compare the full post-top-k set: it is a much richer check of box decode +
    # top-k + the normalized-coords contract than the handful of boxes that clear a
    # positive floor, and it is the set the exported graph actually emits.
    threshold = 0.0
    torch_pred = adapter.predict([image], score_threshold=threshold)[0].detach().cpu().numpy()
    onnx_pred = onnx_adapter.predict([image], score_threshold=threshold)[0].detach().cpu().numpy()

    # ECDet stretches onto its square canvas like RT-DETR, so parity leans on the
    # service's numpy resize matching torch's bilinear F.interpolate — same looser
    # tolerance. This also guards the clip contract: the graph emits normalized
    # boxes whose sigmoid-decoded centres can push past the canvas edge, and
    # to_friendy must clamp them to [0,1] exactly as the adapter's clip_xyxy does.
    _assert_parity(torch_pred, onnx_pred, min_dets=20, atol=5e-3)
