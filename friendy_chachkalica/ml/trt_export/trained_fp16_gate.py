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

    python -m friendy_chachkalica.ml.trt_export.trained_fp16_gate \
        --checkpoint runs/foo/best.pt --images path/to/val/images \
        --score-threshold 0.3 --gate 0.05

``--precision`` picks the *candidate* precision being gated (``fp16`` by default,
``bf16`` for a ModelOpt AutoCast BF16 engine). ``--cast-backend autocast`` gates
ModelOpt's mixed FP32/FP16 graph instead of this repo's default full FP16 cast.

``--reference`` picks what the candidate is held to: the FP32 **engine** built from
the same graph (default, and the historical behaviour — it isolates precision from
every other difference) or **torch**, the model itself. That choice is not academic:
on D-FINE the FP32 engine is itself unfaithful — it loses confident detections the
torch model finds — so gating against it fails the AutoCast FP16 engine that
actually reproduces torch. This gate now always measures the FP32 engine against
torch as well and says so loudly when the reference is the thing that is broken.

Exit code is non-zero if the worst confident-detection L1 exceeds ``--gate`` (so it can
be wired into CI once trained checkpoints are available to it).
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _topk_l1(a, b, k=10):
    """Worst per-detection L1 over the K most-confident, label-matched detections.

    Returns ``inf`` if a confident reference (``a``) detection has no same-label twin
    in ``b`` — including the case where ``b`` is empty, which is the *worst* outcome
    an fp16 engine can produce, not a free pass. ``b`` being empty used to return 0.0
    under "nothing to compare", which made a total fp16 collapse (fp32 finds 4
    people, fp16 finds none, on every image) score a perfect 0.0 and PASS the gate.
    0.0 is only correct when the *reference* side is empty: then there is genuinely
    no confident detection to hold fp16 to.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape[0] == 0:
        return 0.0
    if b.shape[0] == 0:
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


def _calibration_npz(checkpoint: Path, kept, out_dir: Path):
    """Write the gate's own images as an AutoCast calibration ``.npz``, or None.

    Uses the exported graph's ``meta.json`` (Contract B) so the tensors are
    preprocessed exactly as the graph expects — the same path ``onnx_infer`` takes
    at inference. Returns ``None`` (AutoCast falls back to random data at the right
    shape) if the meta or onnx_infer is unavailable rather than failing the build.
    """
    import json

    import numpy as np

    onnx_path = checkpoint.with_suffix(".onnx")
    meta_path = onnx_path.with_suffix(".meta.json")
    if not meta_path.exists():
        return None
    try:
        from onnx_infer.meta import ModelMeta
        from onnx_infer.preprocess import preprocess
    except ImportError:
        return None

    meta = ModelMeta.from_dict(json.loads(meta_path.read_text()))
    dest = Path(out_dir) / "calib"
    dest.mkdir(parents=True, exist_ok=True)
    for i, (_, image, _) in enumerate(kept):
        batch, _ = preprocess(image.numpy(), meta)
        np.savez(dest / f"calib_{i:02d}.npz", pixel_values=np.asarray(batch, dtype=np.float32))
    return str(dest)


def main() -> None:
    ap = argparse.ArgumentParser(description="Trained-checkpoint low-precision parity gate")
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
    ap.add_argument("--precision", choices=["fp16", "bf16"], default="fp16",
                    help="Candidate precision to gate against the fp32 engine")
    ap.add_argument("--cast-backend", choices=["auto", "graph_cast", "autocast"], default="auto",
                    help="How the low-precision graph is produced on strongly-typed TRT")
    ap.add_argument("--reference", choices=["fp32", "torch"], default="fp32",
                    help="What the candidate is held to (default: the fp32 engine). Use "
                         "'torch' when the fp32 engine itself does not reproduce the model.")
    ap.add_argument("--engine-dir", default=None,
                    help="Keep the built engines here (default: a temp dir, discarded)")
    args = ap.parse_args()
    cand = args.precision
    hw = tuple(int(v) for v in args.hw.lower().replace("×", "x").split("x"))

    import torch  # noqa: F401 (imported for the CUDA guard / side effects)

    if not torch.cuda.is_available():
        print("ERROR: needs a CUDA GPU + TensorRT", file=sys.stderr)
        sys.exit(2)

    from friendy_chachkalica.ml.trt_export.cli import _load_adapter, build_engine
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

    import contextlib

    keep_dir = Path(args.engine_dir) if args.engine_dir else None
    if keep_dir:
        keep_dir.mkdir(parents=True, exist_ok=True)
    ctx = contextlib.nullcontext(str(keep_dir)) if keep_dir else tempfile.TemporaryDirectory()
    with ctx as td:
        out = Path(td)
        eng32 = out / f"{ckpt.stem}.fp32.engine"
        engc = out / f"{ckpt.stem}.{cand}.engine"
        if not eng32.exists():
            build_engine(ckpt, eng32, precision="fp32", adapter=adapter,
                         min_hw=hw, opt_hw=hw, max_hw=hw, workspace_gb=args.workspace_gb)
        if not engc.exists():
            # ModelOpt AutoCast decides which nodes may drop to low precision from
            # the magnitudes it observes, and observes RANDOM inputs when given
            # nothing. This gate already holds real in-domain images at exactly the
            # engine's input size — hand them over, so the fp32/low-precision split
            # is chosen on the same data the verdict is measured on.
            build_engine(ckpt, engc, precision=cand, adapter=adapter,
                         min_hw=hw, opt_hw=hw, max_hw=hw, workspace_gb=args.workspace_gb,
                         cast_backend=args.cast_backend,
                         autocast_options={"calibration_data": _calibration_npz(ckpt, kept, out)})
        # Guard the silent-fallback trap: if the candidate build produced no engine,
        # build_engine may fall back to fp32 (fp16 only — bf16 raises), and comparing
        # fp32-vs-fp32 would falsely PASS. Refuse that.
        import json
        provc = json.loads((Path(str(engc) + ".json")).read_text())
        if provc.get("precision") != cand:
            print(f"\n[gate] FAIL: {cand} build fell back to {provc.get('precision')} "
                  f"(no {cand} engine at profile {hw}); cannot validate {cand} parity. Try a "
                  f"different --hw (a dynamic/large profile can defeat the low-precision "
                  f"shape-op tactics).", file=sys.stderr)
            sys.exit(1)
        a32, _ = load_trt_adapter(eng32, "cuda")
        a16, _ = load_trt_adapter(engc, "cuda")

        worst_vs_fp32 = worst_vs_torch = worst_ref_drift = 0.0
        for name, img, torch_pred in kept:
            p32 = a32.predict([img], score_threshold=args.score_threshold)[0].detach().cpu().numpy()
            p16 = a16.predict([img], score_threshold=args.score_threshold)[0].detach().cpu().numpy()
            l1_16_32 = _topk_l1(p32, p16)
            l1_16_t = _topk_l1(torch_pred, p16)
            # The reference's own fidelity, measured every run: an fp32 engine that
            # has already lost the model's detections makes any verdict against it
            # meaningless, and that is not hypothetical (see the module docstring).
            l1_32_t = _topk_l1(torch_pred, p32)
            worst_vs_fp32 = max(worst_vs_fp32, l1_16_32)
            worst_vs_torch = max(worst_vs_torch, l1_16_t)
            worst_ref_drift = max(worst_ref_drift, l1_32_t)
            print(f"[gate] {name}: dets torch={torch_pred.shape[0]} fp32={p32.shape[0]} "
                  f"{cand}={p16.shape[0]} | L1({cand} vs fp32)={l1_16_32:.4g} "
                  f"L1({cand} vs torch)={l1_16_t:.4g} L1(fp32 vs torch)={l1_32_t:.4g}")

    worst = worst_vs_torch if args.reference == "torch" else worst_vs_fp32
    print(f"\n[gate] worst confident L1(fp32 engine vs torch) = {worst_ref_drift:.4g}   "
          f"L1({cand} vs torch) = {worst_vs_torch:.4g}   "
          f"L1({cand} vs fp32) = {worst_vs_fp32:.4g}")
    if worst_ref_drift > args.gate and args.reference == "fp32":
        print(f"[gate] WARNING: the FP32 engine itself misses the torch model by "
              f"{worst_ref_drift:.4g}, past the {args.gate:.4g} gate. A verdict against it "
              f"says nothing about {cand}; re-run with --reference torch.")
    passed = worst <= args.gate
    print(f"[gate] verdict vs {args.reference}: {worst:.4g}  gate = {args.gate:.4g}  "
          f"-> {'PASS' if passed else 'FAIL'}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
