"""Find where an arch's ONNX graph overflows fp16 — no TensorRT needed.

rtdetr builds an fp16 engine that's numerically broken (top-K L1 ~1.9), and op-type
block-listing in ``fp16_cast`` does nothing (see [[trt-fp16-per-arch]]). To locate
the fragile region we run the **fp32** graph once with every intermediate tensor
exposed as an output, then flag tensors whose max |value| exceeds fp16's finite
ceiling (65504) — those saturate to ``inf`` the moment the graph is cast to fp16,
which is what wrecks the downstream decode. Each flagged tensor is mapped back to
its producing node (op type + name) so the fix can keep exactly those nodes fp32
(node-level block list), which op-*type* blocking can't target.

Runs on CPU onnxruntime (fp32 execution, fully supported) in seconds. Usage:

    python -m friendy_chachkalica.trt_export.fp16_overflow_probe --arch rtdetr
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from friendy_chachkalica.registry import build_model  # noqa: E402

FP16_MAX = 65504.0


def _setup(arch):
    """Return (adapter, exporter, hw) reproducing fp16_diag's per-arch fixtures."""
    torch.manual_seed(0)
    if arch == "rtdetr":
        from friendy_chachkalica.onnx_export.arch.rtdetr import export_rtdetr
        a = build_model("rtdetr", num_classes=3, weights=None)
        inner = a.model.model
        torch.nn.init.normal_(inner.enc_score_head.weight, mean=0.0, std=0.2)
        torch.nn.init.constant_(inner.enc_score_head.bias, 0.0)
        torch.nn.init.normal_(inner.decoder.class_embed[-1].weight, mean=0.0, std=0.15)
        torch.nn.init.constant_(inner.decoder.class_embed[-1].bias, -2.0)
        a.eval()
        return a, export_rtdetr, (512, 512)
    if arch == "rfdetr":
        from friendy_chachkalica.onnx_export.arch.rfdetr import export_rfdetr
        a = build_model("rfdetr", num_classes=3, variant="nano", weights=False, resolution=224)
        a.eval()
        return a, export_rfdetr, (224, 224)
    if arch == "fasterrcnn":
        # NOTE: probes the STANDARD onnx (pre-EfficientNMS-surgery). The prep'd
        # trt.onnx has a TRT plugin node ORT can't run, but the box-delta decode /
        # RPN math where fp16 would overflow lives in this shared graph.
        from friendy_chachkalica.onnx_export.arch.fasterrcnn import export_fasterrcnn
        a = build_model("fasterrcnn", num_classes=3, variant="mobilenet_v3_large_320_fpn")
        cls_score = a.model.roi_heads.box_predictor.cls_score
        with torch.no_grad():
            torch.nn.init.constant_(cls_score.bias, 0.0)
            cls_score.bias[0] = -4.0
        a.eval()
        return a, export_fasterrcnn, (320, 320)
    raise SystemExit(f"probe not wired for arch {arch!r} (add a _setup branch)")


def _export(adapter, exporter, out_dir, num_classes=3):
    import json
    onnx_path = out_dir / "model.onnx"
    meta = exporter(
        adapter, num_classes=num_classes, params={},
        class_map={i: chr(ord("a") + i) for i in range(num_classes)},
        onnx_path=onnx_path,
    )
    onnx_path.with_suffix(".meta.json").write_text(json.dumps(meta))
    return onnx_path, meta


def probe(arch, top=40):
    import onnx
    import onnxruntime as ort

    adapter, exporter, hw = _setup(arch)
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        onnx_path, meta = _export(adapter, exporter, out)
        input_name = meta.get("input_name", "pixel_values")

        model = onnx.load(str(onnx_path))
        # Map every intermediate tensor -> the node that produces it.
        producer = {}
        for node in model.graph.node:
            for o in node.output:
                if o:
                    producer[o] = (node.op_type, node.name)
        existing = {o.name for o in model.graph.output}
        added = [name for name in producer if name not in existing]
        for name in added:
            model.graph.output.append(onnx.helper.make_empty_tensor_value_info(name))

        sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        torch.manual_seed(1)
        img = torch.rand(1, 3, *hw).numpy().astype(np.float32)
        out_names = [o.name for o in sess.get_outputs()]
        vals = sess.run(out_names, {input_name: img})

    rows = []
    for name, arr in zip(out_names, vals):
        a = np.asarray(arr)
        if a.dtype.kind != "f" or a.size == 0:
            continue
        finite = a[np.isfinite(a)]
        mx = float(np.abs(finite).max()) if finite.size else float("inf")
        n_nonfinite = int(a.size - finite.size)
        op, nn = producer.get(name, ("<graph-input>", ""))
        rows.append((mx, n_nonfinite, name, op, nn))

    over = [r for r in rows if r[0] > FP16_MAX or r[1] > 0]
    over.sort(key=lambda r: (r[1], r[0]), reverse=True)

    print(f"\n{'='*100}\n{arch}: intermediate tensors that overflow fp16 (|x| > {FP16_MAX:.0f} or non-finite)")
    print(f"scanned {len(rows)} float tensors; {len(over)} overflow\n{'-'*100}")
    print(f"{'max|x|':>12}  {'#nonfin':>8}  {'op_type':<16} tensor / node")
    for mx, nnf, name, op, nn in over[:top]:
        mxs = "inf" if mx == float("inf") else f"{mx:.3e}"
        print(f"{mxs:>12}  {nnf:>8}  {op:<16} {name}  [{nn}]")
    print("=" * 100)
    if over:
        from collections import Counter
        c = Counter(op for _, _, _, op, _ in over)
        print("overflowing op_types:", ", ".join(f"{k}×{v}" for k, v in c.most_common()))
        print("Fix: keep these NODES fp32 via a node_block_list in fp16_cast (node names above),\n"
              "or trace them to the smallest upstream op whose output first exceeds 65504.")
    else:
        print("No fp16 overflow found — the failure is precision loss, not overflow "
              "(different fix: the accumulation/softmax path, not a clamp).")


def main():
    ap = argparse.ArgumentParser(description="Locate fp16 overflow in an arch's ONNX graph")
    ap.add_argument("--arch", default="rtdetr")
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args()
    probe(args.arch, top=args.top)


if __name__ == "__main__":
    main()
