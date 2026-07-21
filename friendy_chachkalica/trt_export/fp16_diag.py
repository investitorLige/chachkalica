"""Per-arch FP16 diagnostic: which archs build + keep parity at fp16, and where.

⚠️ RANDOM-INIT CAVEAT: the fixtures below are UNTRAINED. Trust the ``build`` column and
overflow findings, but NOT the ``L1`` column as a verdict on fp16 *accuracy* for the
selection-heavy archs (rtdetr, fasterrcnn). Untrained weights make all scores ~tied, so
top-K / NMS selection flips under any fp16 perturbation and the L1 blows up (rtdetr ~1.9,
fasterrcnn ~0.84) — a fixture artifact, not a real defect. On trained weights fp16 parity
is at the gate (rtdetr ~1e-4, fasterrcnn 0.012-0.035). Gate fp16 on a TRAINED checkpoint
with ``trained_fp16_gate.py`` before flooring an arch to fp32. See [[trt-fp16-per-arch]].


Companion to ``trt_infer/tests/test_trt_parity.py`` — reuses the same per-arch
fixture setup (seeds / bias inits that yield confident detections) but, instead of
asserting fp32 parity, builds each arch at **both** fp32 and fp16 and prints one
row per arch:

    build      did the fp16 engine build at all (else the TRT/build error)
    precision  what the .engine.json sidecar actually recorded (fp16 / fp32)
    method     how fp16 was obtained (builder_flag | onnxruntime | ...)
    L1 vs f32  top-K parity L1 of the fp16 engine against the fp32 engine
    L1 vs torch top-K parity L1 of the fp16 engine against the torch adapter

That separates the two failure modes cleanly: an fp16 build that errors shows up
under ``build``; one that builds but is numerically wrong shows a large ``L1``.
Run on a CUDA + TensorRT box:

    python -m friendy_chachkalica.trt_export.fp16_diag
    python -m friendy_chachkalica.trt_export.fp16_diag --archs rtdetr retinanet
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from friendy_chachkalica.onnx_export.arch.fasterrcnn import export_fasterrcnn  # noqa: E402
from friendy_chachkalica.onnx_export.arch.retinanet import export_retinanet  # noqa: E402
from friendy_chachkalica.onnx_export.arch.rfdetr import export_rfdetr  # noqa: E402
from friendy_chachkalica.onnx_export.arch.rtdetr import export_rtdetr  # noqa: E402
from friendy_chachkalica.onnx_export.arch.yolox import export_yolox  # noqa: E402
from friendy_chachkalica.registry import build_model  # noqa: E402
from friendy_chachkalica.trt_export.cli import build_engine  # noqa: E402
from trt_infer import load_trt_adapter  # noqa: E402


def _topk_l1(a, b, k=10):
    """Max per-detection L1 over the K most confident, label-matched. inf if unpaired."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return float("inf")
    n = min(k, a.shape[0])
    top_a = a[np.argsort(-a[:, 4])][:n]
    top_b = b[np.argsort(-b[:, 4])]
    used, worst = set(), 0.0
    for row in top_a:
        best_j, best_d = None, np.inf
        for j, cand in enumerate(top_b):
            if j in used or int(cand[5]) != int(row[5]):
                continue
            d = float(np.abs(row[:5] - cand[:5]).sum())
            if d < best_d:
                best_d, best_j = d, j
        if best_j is None:
            return float("inf")
        used.add(best_j)
        worst = max(worst, best_d)
    return worst


# --- per-arch setup: (build_adapter, exporter, static_hw, pass_adapter, score_thr) ---


def _setup_retinanet():
    torch.manual_seed(0)
    a = build_model("retinanet", num_classes=3)
    torch.nn.init.normal_(a.model.head.classification_head.cls_logits.bias, mean=-2.0, std=2.0)
    a.eval()
    return a, export_retinanet, (512, 512), True, 0.05


def _setup_yolox():
    torch.manual_seed(0)
    a = build_model("yolox", num_classes=3, variant="yolox-nano")
    for obj_conv in a.model.head.obj_preds:
        torch.nn.init.constant_(obj_conv.bias, 2.0)
    for cls_conv in a.model.head.cls_preds:
        torch.nn.init.normal_(cls_conv.bias, mean=0.0, std=2.0)
    a.score_threshold = 0.05
    a.eval()
    return a, export_yolox, (512, 512), True, 0.05


def _setup_rtdetr():
    torch.manual_seed(0)
    a = build_model("rtdetr", num_classes=3, weights=None)
    inner = a.model.model
    torch.nn.init.normal_(inner.enc_score_head.weight, mean=0.0, std=0.2)
    torch.nn.init.constant_(inner.enc_score_head.bias, 0.0)
    torch.nn.init.normal_(inner.decoder.class_embed[-1].weight, mean=0.0, std=0.15)
    torch.nn.init.constant_(inner.decoder.class_embed[-1].bias, -2.0)
    a.eval()
    return a, export_rtdetr, (512, 512), False, 0.0


def _setup_rfdetr():
    torch.manual_seed(0)
    a = build_model("rfdetr", num_classes=3, variant="nano", weights=False, resolution=224)
    a.eval()
    return a, export_rfdetr, (224, 224), False, 0.0


def _setup_fasterrcnn():
    torch.manual_seed(0)
    a = build_model("fasterrcnn", num_classes=3, variant="mobilenet_v3_large_320_fpn")
    cls_score = a.model.roi_heads.box_predictor.cls_score
    with torch.no_grad():
        torch.nn.init.constant_(cls_score.bias, 0.0)
        cls_score.bias[0] = -4.0
    a.eval()
    # Option B: fasterrcnn now compiles via EfficientNMS surgery (prep_fasterrcnn),
    # so build_engine needs the torch adapter, like retinanet/yolox.
    return a, export_fasterrcnn, (320, 320), True, 0.05


SETUPS = {
    "yolox": _setup_yolox,
    "rfdetr": _setup_rfdetr,
    "rtdetr": _setup_rtdetr,
    "retinanet": _setup_retinanet,
    "fasterrcnn": _setup_fasterrcnn,
}


def _export(adapter, exporter, out_dir, num_classes=3):
    onnx_path = out_dir / "model.onnx"
    meta = exporter(
        adapter, num_classes=num_classes, params={},
        class_map={i: chr(ord("a") + i) for i in range(num_classes)},
        onnx_path=onnx_path,
    )
    onnx_path.with_suffix(".meta.json").write_text(json.dumps(meta))
    return onnx_path


def _predict(engine, image, score_thr):
    adapter, _ = load_trt_adapter(engine, "cuda")
    return adapter.predict([image], score_threshold=score_thr)[0].detach().cpu().numpy()


def diagnose(name, node_block=None):
    setup = SETUPS[name]
    adapter, exporter, hw, pass_adapter, score_thr = setup()
    row = {"arch": name, "build": "?", "precision": "-", "method": "-",
           "l1_vs_fp32": None, "l1_vs_torch": None}
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        onnx_path = _export(adapter, exporter, out)
        build_adapter = adapter if pass_adapter else None

        torch.manual_seed(1)
        image = torch.rand(3, *hw)
        torch_pred = adapter.predict([image], score_threshold=score_thr)[0].detach().cpu().numpy()

        # fp32 reference
        try:
            eng32 = build_engine(onnx_path, out / "m32.engine", precision="fp32",
                                 adapter=build_adapter, min_hw=hw, opt_hw=hw, max_hw=hw, workspace_gb=2.0)
            pred32 = _predict(eng32, image, score_thr)
        except Exception as e:  # noqa: BLE001
            row["build"] = f"fp32 FAILED: {type(e).__name__}: {e}"
            return row

        # fp16 under test
        try:
            eng16 = build_engine(onnx_path, out / "m16.engine", precision="fp16",
                                 adapter=build_adapter, min_hw=hw, opt_hw=hw, max_hw=hw,
                                 node_block_substrings=node_block, workspace_gb=2.0)
        except Exception as e:  # noqa: BLE001
            row["build"] = f"FP16 BUILD FAILED: {type(e).__name__}: {e}"
            return row

        prov = json.loads((Path(str(eng16) + ".json")).read_text())
        row["build"] = "ok"
        row["precision"] = prov.get("precision", "?")
        row["method"] = prov.get("fp16_method") or "-"
        try:
            pred16 = _predict(eng16, image, score_thr)
            row["l1_vs_fp32"] = _topk_l1(pred32, pred16)
            row["l1_vs_torch"] = _topk_l1(torch_pred, pred16)
        except Exception as e:  # noqa: BLE001
            row["build"] = f"fp16 INFER FAILED: {type(e).__name__}: {e}"
    return row


# Candidate fp32 keep-lists to try per arch, cheapest (fewest ops kept fp32) first.
# Add/trim these as the results narrow the fragile region.
SWEEP_CANDIDATES = {
    "rtdetr": [
        [],
        ["Where"],
        ["Where", "Sigmoid"],
        ["Where", "Sigmoid", "Div", "Sub"],
        ["Where", "Sigmoid", "Div", "Sub", "Add", "Mul"],
    ],
    "fasterrcnn": [
        [],
        ["Mul", "Add"],
        ["Where"],
        ["Mul", "Add", "Where"],
        ["Mul", "Add", "Sub", "Div", "Where"],
    ],
    "rfdetr": [
        [],
        ["Sigmoid"],
        ["Sigmoid", "Div", "Sub"],
        ["Sigmoid", "Add", "Mul", "Div", "Sub"],
    ],
}


def sweep(name):
    """Build fp32 once, then fp16 with each candidate keep-list; report L1 per set.

    Explicit precision='fp16' bypasses the fp32 floor, so this measures recovery
    even for archs currently on the floor. extra_fp32_ops extends the (empty, during
    tuning) per-arch registry list, so each candidate == the effective keep-list.
    """
    adapter, exporter, hw, pass_adapter, score_thr = SETUPS[name]()
    candidates = SWEEP_CANDIDATES.get(name, [[], ["Where"], ["Sigmoid"]])
    rows = []
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        onnx_path = _export(adapter, exporter, out)
        build_adapter = adapter if pass_adapter else None
        torch.manual_seed(1)
        image = torch.rand(3, *hw)
        torch_pred = adapter.predict([image], score_threshold=score_thr)[0].detach().cpu().numpy()

        eng32 = build_engine(onnx_path, out / "m32.engine", precision="fp32",
                             adapter=build_adapter, min_hw=hw, opt_hw=hw, max_hw=hw, workspace_gb=2.0)
        pred32 = _predict(eng32, image, score_thr)

        for i, ops in enumerate(candidates):
            label = ",".join(ops) if ops else "(none)"
            try:
                eng16 = build_engine(out / "model.onnx", out / f"m16_{i}.engine", precision="fp16",
                                     adapter=build_adapter, min_hw=hw, opt_hw=hw, max_hw=hw,
                                     extra_fp32_ops=ops, workspace_gb=2.0)
                pred16 = _predict(eng16, image, score_thr)
                l1_32 = _topk_l1(pred32, pred16)
                l1_t = _topk_l1(torch_pred, pred16)
                rows.append((label, l1_32, l1_t, ""))
            except Exception as e:  # noqa: BLE001
                rows.append((label, None, None, f"{type(e).__name__}: {e}"))
            print(f"[sweep {name}] fp32-keep=[{label}] -> "
                  f"L1 vs fp32 {_fmt(rows[-1][1])} vs torch {_fmt(rows[-1][2])} {rows[-1][3]}")

    print(f"\n{'='*80}\nSWEEP {name}: fp32 keep-list -> top-K parity L1\n{'-'*80}")
    print(f"{'fp32-kept ops':<40} {'L1 vs fp32':<12} {'L1 vs torch':<12} note")
    for label, l1_32, l1_t, err in rows:
        print(f"{label[:39]:<40} {_fmt(l1_32):<12} {_fmt(l1_t):<12} {err}")
    print("=" * 80)
    print("Pick the SHORTEST list whose L1 drops under your gate; put it in "
          "arch/__init__.py ARCH_FP16_OP_BLOCK and drop the arch from UNTRUSTED_FP16.")


def _fmt(v):
    if v is None:
        return "-"
    if v == float("inf"):
        return "inf"
    return f"{v:.2e}"


def main():
    ap = argparse.ArgumentParser(description="Per-arch FP16 build + parity diagnostic")
    ap.add_argument("--archs", nargs="*", default=list(SETUPS), help="subset to test")
    ap.add_argument("--sweep", help="arch to sweep fp32 keep-lists for (instead of the table)")
    ap.add_argument("--node-block", help="comma-separated node-name substrings to keep fp32 (fp16 build)")
    args = ap.parse_args()
    node_block = [s for s in (args.node_block or "").split(",") if s] or None

    if not torch.cuda.is_available():
        print("ERROR: needs a CUDA GPU + TensorRT", file=sys.stderr)
        sys.exit(1)

    if args.sweep:
        if args.sweep not in SETUPS:
            print(f"unknown arch {args.sweep!r}; known: {list(SETUPS)}", file=sys.stderr)
            sys.exit(1)
        sweep(args.sweep)
        return

    rows = []
    for name in args.archs:
        if name not in SETUPS:
            print(f"unknown arch {name!r}; known: {list(SETUPS)}", file=sys.stderr)
            continue
        print(f"\n===== {name} =====")
        try:
            rows.append(diagnose(name, node_block=node_block))
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            rows.append({"arch": name, "build": f"SETUP FAILED: {type(e).__name__}: {e}",
                         "precision": "-", "method": "-", "l1_vs_fp32": None, "l1_vs_torch": None})

    print("\n" + "=" * 92)
    print(f"{'arch':<12} {'build':<28} {'precision':<10} {'method':<14} {'L1 vs fp32':<12} {'L1 vs torch'}")
    print("-" * 92)
    for r in rows:
        print(f"{r['arch']:<12} {r['build'][:27]:<28} {r['precision']:<10} "
              f"{r['method'][:13]:<14} {_fmt(r['l1_vs_fp32']):<12} {_fmt(r['l1_vs_torch'])}")
    print("=" * 92)
    print("Read: build!=ok -> build-time failure. precision=fp32 with build=ok -> silently "
          "fell back (shouldn't now). Large L1 vs fp32 -> numeric fp16 failure in this arch.")


if __name__ == "__main__":
    main()
