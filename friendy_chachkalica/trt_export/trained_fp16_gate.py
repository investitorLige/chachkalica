"""Trained-checkpoint FP16 parity gate — the defensible complement to ``fp16_diag``.

``fp16_diag`` builds each arch from a **random-init** fixture. That is fine for the
build/overflow checks it was written for, but it gives a FALSE NEGATIVE on FP16
*accuracy* for the selection-heavy two-stage / DETR archs: with untrained weights every
score is ~tied, so top-K / NMS selection is a coin-flip that any FP16 perturbation flips,
inflating the parity L1 (fasterrcnn 0.84, rtdetr 1.9). A *trained* model has
well-separated scores, so selection is stable and FP16 parity collapses to the gate
(measured: fasterrcnn 0.012-0.035, rtdetr ~1e-4). See [[trt-fp16-per-arch]].

This gate therefore measures FP16 parity the honest way: from a **real trained
checkpoint** on **real images**. It builds the arch's FP32 and FP16 engines (the normal
``build_engine`` path — EfficientNMS surgery for two-stage/single-stage-NMS archs,
passthrough for DETR) and asserts the FP16 engine's confident top-K detections match the
FP32 engine's within a tolerance.

Usage (run in the trainer docker; needs a CUDA GPU + TensorRT — see
[[trt-build-test-harness]])::

    python -m friendy_chachkalica.trt_export.trained_fp16_gate \
        --checkpoint runs/foo/best.pt --images path/to/val/images \
        --score-threshold 0.3 --gate 0.05

Exit code is non-zero if the worst confident-detection L1 exceeds ``--gate`` (so it can
be wired into CI once trained checkpoints are available to it).
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


def _topk_l1(a, b, k=10):
    """Worst per-detection L1 over the K most-confident, label-matched detections.

    Returns ``inf`` if a confident torch/fp32 detection has no same-label twin in the
    other set (a hard miss), 0.0 if either side is empty (nothing to compare).
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return 0.0
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


def _load_images(image_dir: Path, limit: int, adapter, score_threshold: float, hw):
    """Load images resized to ``hw``, keep those the trained model fires on (>=1 det).

    Resizing to the engine's fixed profile size keeps the input inside the optimization
    profile and makes the torch/fp32/fp16 comparison apples-to-apples (all see the same
    tensor). ``hw`` is ``(H, W)``.
    """
    from torchvision.io import read_image
    from torchvision.transforms.v2 import functional as TF

    kept = []
    exts = ("*.jpg", "*.jpeg", "*.png")
    paths = sorted(p for ext in exts for p in image_dir.glob(ext))
    for p in paths:
        t = read_image(str(p)).float() / 255.0
        if t.shape[0] == 1:
            t = t.repeat(3, 1, 1)
        elif t.shape[0] == 4:
            t = t[:3]
        t = TF.resize(t, [hw[0], hw[1]], antialias=True)
        pred = adapter.predict([t], score_threshold=score_threshold)[0].detach().cpu().numpy()
        if pred.shape[0] >= 1:
            kept.append((p.name, t, pred))
        if len(kept) >= limit:
            break
    return kept


def main() -> None:
    ap = argparse.ArgumentParser(description="Trained-checkpoint FP16 parity gate")
    ap.add_argument("--checkpoint", required=True, help="Path to a trained .pt checkpoint")
    ap.add_argument("--images", required=True, help="Directory of real (in-domain) images")
    ap.add_argument("--score-threshold", type=float, default=0.3)
    ap.add_argument("--gate", type=float, default=5e-2, help="Max allowed confident top-K L1")
    ap.add_argument("--limit", type=int, default=8, help="Max images with detections to gate on")
    ap.add_argument("--hw", default="640x640",
                    help="Fixed engine input HxW (e.g. 640x640). Images are resized to it and "
                         "the profile is pinned there — a dynamic profile can defeat the fp16 "
                         "build's shape-op tactics (esp. the two-stage transform).")
    ap.add_argument("--workspace-gb", type=float, default=3.0)
    args = ap.parse_args()
    hw = tuple(int(v) for v in args.hw.lower().replace("×", "x").split("x"))

    import torch  # noqa: F401 (imported for the CUDA guard / side effects)

    if not torch.cuda.is_available():
        print("ERROR: needs a CUDA GPU + TensorRT", file=sys.stderr)
        sys.exit(2)

    from friendy_chachkalica.trt_export.cli import _load_adapter, build_engine
    from trt_infer import load_trt_adapter

    ckpt = Path(args.checkpoint)
    image_dir = Path(args.images)
    adapter = _load_adapter(ckpt)

    kept = _load_images(image_dir, args.limit, adapter, args.score_threshold, hw)
    if not kept:
        print(f"ERROR: no image in {image_dir} produced a detection at "
              f"score>={args.score_threshold}; pick in-domain images or lower the threshold",
              file=sys.stderr)
        sys.exit(2)
    print(f"[gate] {len(kept)} images with detections")

    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        eng32 = build_engine(ckpt, out / "fp32.engine", precision="fp32", adapter=adapter,
                             min_hw=hw, opt_hw=hw, max_hw=hw, workspace_gb=args.workspace_gb)
        eng16 = build_engine(ckpt, out / "fp16.engine", precision="fp16", adapter=adapter,
                             min_hw=hw, opt_hw=hw, max_hw=hw, workspace_gb=args.workspace_gb)
        # Guard the silent-fallback trap: if the fp16 build produced no engine, build_engine
        # falls back to fp32, and comparing fp32-vs-fp32 would falsely PASS. Refuse that.
        import json
        prov16 = json.loads((Path(str(eng16) + ".json")).read_text())
        if prov16.get("precision") != "fp16":
            print(f"\n[gate] FAIL: fp16 build fell back to {prov16.get('precision')} "
                  f"(no fp16 engine at profile {hw}); cannot validate fp16 parity. Try a "
                  f"different --hw (a dynamic/large profile can defeat the fp16 shape-op tactics).",
                  file=sys.stderr)
            sys.exit(1)
        a32, _ = load_trt_adapter(eng32, "cuda")
        a16, _ = load_trt_adapter(eng16, "cuda")

        worst = 0.0
        for name, img, torch_pred in kept:
            p32 = a32.predict([img], score_threshold=args.score_threshold)[0].detach().cpu().numpy()
            p16 = a16.predict([img], score_threshold=args.score_threshold)[0].detach().cpu().numpy()
            l1_16_32 = _topk_l1(p32, p16)
            l1_16_t = _topk_l1(torch_pred, p16)
            worst = max(worst, l1_16_32)
            print(f"[gate] {name}: dets torch={torch_pred.shape[0]} fp32={p32.shape[0]} "
                  f"fp16={p16.shape[0]} | L1(fp16 vs fp32)={l1_16_32:.4g} "
                  f"L1(fp16 vs torch)={l1_16_t:.4g}")

    passed = worst <= args.gate
    print(f"\n[gate] worst confident L1(fp16 vs fp32) = {worst:.4g}  gate = {args.gate:.4g}  "
          f"-> {'PASS' if passed else 'FAIL'}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
