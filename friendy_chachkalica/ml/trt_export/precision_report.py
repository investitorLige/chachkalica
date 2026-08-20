"""Compare a model's runtimes and precisions on real images, and time them.

``trained_fp16_gate`` answers one yes/no question (does the low-precision engine
still find the same confident detections as the FP32 engine). This is the wider
instrument behind that verdict: run PyTorch, onnxruntime and any number of
TensorRT engines over the same images, then report *what changed* — detection
counts at the operating threshold, label-matched score and box deltas by
confidence band, how many confident detections cross below the threshold, an
agreement mAP against a chosen reference runtime, and engine-only latency.

Why a reference runtime is a parameter and not "the FP32 engine": on D-FINE this
tool showed that a TensorRT FP32 engine — TF32 disabled, strictly FP32 — already
scores detections systematically lower than PyTorch and onnxruntime, which agree
with each other to ±0.002. Judging FP16 against that engine hides the larger
error underneath it. The default reference is therefore ``torch-fp32``.

Usage::

    python -m friendy_chachkalica.ml.trt_export.precision_report compare \\
        --checkpoint best.pt --onnx best.onnx --images val/images \\
        --engine fp32=out/m.fp32.engine --engine bf16=out/m.bf16.engine \\
        --save-preds preds.npz

    python -m friendy_chachkalica.ml.trt_export.precision_report latency \\
        --engine fp32=out/m.fp32.engine --engine bf16=out/m.bf16.engine

    # re-analyse saved detections without a GPU or any engine
    python -m friendy_chachkalica.ml.trt_export.precision_report compare \\
        --from-preds preds.npz --reference torch-fp32

Detection rows are Friendy format — ``(cx, cy, w, h, score, label)`` with the box
normalized to the model input (see ``metrics._prepare_prediction``). Geometry here
converts to xyxy first, and box deltas are reported in input pixels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LOW = 0.05        # pull the whole tail so nothing hides behind the threshold
OPERATING = 0.30  # the threshold a deployment would actually run at


# --------------------------------------------------------------------------- geometry


def to_xyxy(rows: np.ndarray) -> np.ndarray:
    """``(N,6)`` Friendy cxcywh-normalized rows -> ``(N,4)`` normalized xyxy."""
    cx, cy, w, h = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of every xyxy box in ``a`` against every one in ``b``."""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0).astype(np.float32)


def match(ref: np.ndarray, cand: np.ndarray):
    """Greedy label-matched nearest pairing -> ``[(ref_row, cand_row or None), ...]``."""
    used, pairs = set(), []
    for r in ref[np.argsort(-ref[:, 4])]:
        best_j, best_d = None, np.inf
        for j, c in enumerate(cand):
            if j in used or int(c[5]) != int(r[5]):
                continue
            d = float(np.abs(r[:4] - c[:4]).sum())
            if d < best_d:
                best_d, best_j = d, j
        if best_j is not None:
            used.add(best_j)
        pairs.append((r, None if best_j is None else cand[best_j]))
    return pairs


# --------------------------------------------------------------------------- metrics


def compare(ref_preds, cand_preds, *, side: int) -> dict:
    """Paired per-detection stats of one runtime against the reference runtime."""
    bands = {"s>=0.70": (0.70, 1.01), "0.50-0.70": (0.50, 0.70),
             "0.30-0.50": (0.30, 0.50), "0.05-0.30": (LOW, 0.30)}
    rows = []
    for ref, cand in zip(ref_preds, cand_preds):
        if not len(ref):
            continue
        for r, c in match(ref, cand):
            rows.append({
                "s_ref": float(r[4]),
                "s_cand": None if c is None else float(c[4]),
                "dscore": None if c is None else float(c[4]) - float(r[4]),
                "box_l1": None if c is None else float(
                    np.abs(to_xyxy(r[None]) - to_xyxy(c[None])).sum() * side),
            })

    stats = {"paired_total": len(rows), "bands": {}}
    for name, (lo, hi) in bands.items():
        band = [r for r in rows if lo <= r["s_ref"] < hi]
        if not band:
            continue
        kept = [r for r in band if r["s_cand"] is not None]
        ds = [abs(r["dscore"]) for r in kept]
        db = [r["box_l1"] for r in kept]
        stats["bands"][name] = {
            "n": len(band),
            "unmatched": len(band) - len(kept),
            "dscore_med": float(np.median(ds)) if ds else None,
            "dscore_max": float(max(ds)) if ds else None,
            "box_l1_px_med": float(np.median(db)) if db else None,
            "box_l1_px_max": float(max(db)) if db else None,
        }

    confident = [r for r in rows if r["s_ref"] >= OPERATING]
    kept = [r for r in confident if r["s_cand"] is not None]
    stats["confident_n"] = len(confident)
    stats["unmatched"] = len(confident) - len(kept)
    stats["crossed_below_operating"] = sum(
        1 for r in confident if r["s_cand"] is None or r["s_cand"] < OPERATING)
    stats["dscore_gt_0.10"] = sum(1 for r in kept if abs(r["dscore"]) > 0.10)
    if kept:
        signed = [r["dscore"] for r in kept]
        stats["confident_dscore_med"] = float(np.median([abs(d) for d in signed]))
        # Signed mean matters: rounding noise is symmetric, a systematic bias is not.
        stats["confident_dscore_signed_mean"] = float(np.mean(signed))
        stats["confident_box_l1_px_med"] = float(np.median([r["box_l1"] for r in kept]))
        stats["confident_box_l1_px_max"] = float(max(r["box_l1"] for r in kept))
    return stats


def ap_at_iou(ref_preds, cand_preds, iou_thr: float) -> float:
    """COCO-style 101-point AP of the candidate against the reference's confident
    detections, averaged over the classes the reference actually produced.

    Written here rather than reusing ``metrics.evaluate_detection`` because that
    function's contract is normalized-cxcywh *predictions* against absolute-pixel
    *targets*, while both sides here are the same Friendy rows.
    """
    by_class: Dict[int, dict] = {}
    for ref, cand in zip(ref_preds, cand_preds):
        gt = ref[ref[:, 4] >= OPERATING]
        for cls in set(gt[:, 5].astype(int)) | set(cand[:, 5].astype(int)):
            entry = by_class.setdefault(int(cls), {"gt": 0, "rows": []})
            g = gt[gt[:, 5].astype(int) == cls]
            c = cand[cand[:, 5].astype(int) == cls]
            entry["gt"] += len(g)
            matched = np.zeros(len(g), dtype=bool)
            ious = iou_matrix(to_xyxy(c), to_xyxy(g))
            for i in np.argsort(-c[:, 4]):
                hit = False
                if len(g):
                    for j in np.argsort(-ious[i]):
                        if ious[i, j] < iou_thr:
                            break
                        if not matched[j]:
                            matched[j] = hit = True
                            break
                entry["rows"].append((float(c[i, 4]), hit))

    aps = []
    for entry in by_class.values():
        if entry["gt"] == 0:
            continue
        ordered = sorted(entry["rows"], key=lambda r: -r[0])
        tp = np.cumsum([1 if hit else 0 for _, hit in ordered])
        fp = np.cumsum([0 if hit else 1 for _, hit in ordered])
        recall = tp / entry["gt"]
        precision = tp / np.maximum(tp + fp, 1)
        ap = 0.0
        for r in np.linspace(0, 1, 101):
            mask = recall >= r
            ap += (precision[mask].max() if mask.any() else 0.0) / 101
        aps.append(ap)
    return float(np.mean(aps)) if aps else 0.0


def agreement_map(ref_preds, cand_preds) -> dict:
    """mAP50 / mAP50-95 of a runtime scored against the reference runtime's output.

    Not a COCO mAP — there is no ground truth here — but it collapses box drift,
    score drift and lost detections into the numbers an eval report already speaks
    in. The reference's detections above the operating threshold are the "labels".
    """
    thrs = np.arange(0.5, 0.96, 0.05)
    return {
        "map50": round(ap_at_iou(ref_preds, cand_preds, 0.5), 4),
        "map50_95": round(float(np.mean([ap_at_iou(ref_preds, cand_preds, float(t))
                                         for t in thrs])), 4),
    }


# --------------------------------------------------------------------------- runtimes


def load_images(image_dir: Path, hw, limit: Optional[int] = None):
    """``[(name, CHW float [0,1] tensor resized to hw), ...]``."""
    from torchvision.io import read_image
    from torchvision.transforms.v2 import functional as TF

    out = []
    paths = sorted(p for ext in ("*.jpg", "*.jpeg", "*.png") for p in image_dir.glob(ext))
    for p in paths[: limit or None]:
        t = read_image(str(p)).float() / 255.0
        if t.shape[0] == 1:
            t = t.repeat(3, 1, 1)
        elif t.shape[0] == 4:
            t = t[:3]
        out.append((p.name, TF.resize(t, [hw[0], hw[1]], antialias=True)))
    return out


def measure_latency(model, iters: int, warmup: int, hw) -> dict:
    """Engine-only latency, the way D-FINE's own trt_benchmark measures it.

    Fixed ``[1,3,H,W]`` input, long warmup, one synchronize per iteration. Median
    and p95 as well as the mean, because the mean alone hides the tail.
    """
    import torch

    x = torch.zeros((1, 3, *hw), dtype=torch.float32, device="cuda")
    for _ in range(warmup):
        model.run_torch(x)
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        model.run_torch(x)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    t = np.array(times)
    return {"mean_ms": round(float(t.mean()), 3), "median_ms": round(float(np.median(t)), 3),
            "p95_ms": round(float(np.percentile(t, 95)), 3),
            "fps": round(1000.0 / float(np.median(t)), 1)}


def _engine_meta(path: Path) -> dict:
    meta = {"engine": path.name, "size_mb": round(path.stat().st_size / 1e6, 1)}
    sidecar = Path(str(path) + ".json")
    if sidecar.exists():
        info = json.loads(sidecar.read_text())
        meta.update({
            "built_precision": info.get("precision"),
            "method": info.get("precision_method"),
            "tf32": info.get("tf32"),
            "autocast": info.get("autocast"),
            "engine_tensor_dtypes": (info.get("inspection") or {}).get("tensors"),
        })
    return meta


# --------------------------------------------------------------------------- reporting


def build_report(preds, ref_name, metas, *, images_n, side, latencies=None) -> dict:
    report = {"reference": ref_name, "images": images_n,
              "operating_threshold": OPERATING, "runtimes": {}}
    for name in preds:
        entry = {
            "detections_at_operating": sum(int((p[:, 4] >= OPERATING).sum()) for p in preds[name]),
            "meta": metas.get(name, {}),
        }
        if name != ref_name:
            entry["vs_reference"] = compare(preds[ref_name], preds[name], side=side)
            entry["agreement"] = agreement_map(preds[ref_name], preds[name])
        if latencies and name in latencies:
            entry["latency"] = latencies[name]
        report["runtimes"][name] = entry
    return report


def print_report(report: dict) -> None:
    print(f"\nreference = {report['reference']}, {report['images']} images, "
          f"operating threshold {report['operating_threshold']}")
    header = (f"{'runtime':<22} {'dets':>6} {'lat ms':>7} {'fps':>7} {'crossed':>8} "
              f"{'|ds|>.1':>8} {'|ds| med':>9} {'ds mean':>9} {'box px':>7} "
              f"{'mAP50':>7} {'mAP50-95':>9}")
    print(header)
    for name, entry in report["runtimes"].items():
        vs, ag, lat = entry.get("vs_reference", {}), entry.get("agreement", {}), entry.get("latency", {})

        def f(value, nd=3):
            return f"{value:.{nd}f}" if isinstance(value, float) else str(value)

        print(f"{name:<22} {entry['detections_at_operating']:>6} "
              f"{f(lat.get('median_ms', '-'), 2):>7} {f(lat.get('fps', '-'), 1):>7} "
              f"{str(vs.get('crossed_below_operating', '-')):>8} "
              f"{str(vs.get('dscore_gt_0.10', '-')):>8} "
              f"{f(vs.get('confident_dscore_med', '-')):>9} "
              f"{f(vs.get('confident_dscore_signed_mean', '-')):>9} "
              f"{f(vs.get('confident_box_l1_px_med', '-'), 2):>7} "
              f"{f(ag.get('map50', '-')):>7} {f(ag.get('map50_95', '-')):>9}")


def _parse_engines(values: List[str]) -> Dict[str, Path]:
    out = {}
    for value in values or []:
        if "=" not in value:
            raise SystemExit(f"--engine expects name=path, got {value!r}")
        name, path = value.split("=", 1)
        out[name] = Path(path)
    return out


def cmd_compare(args) -> None:
    engines = _parse_engines(args.engine)
    side = args.side

    if args.from_preds:
        z = np.load(args.from_preds, allow_pickle=True)
        names = sorted({k.split("|")[0] for k in z.files if "|" in k})
        preds = {
            name: [np.asarray(z[f"{name}|{i}"], dtype=np.float32)
                   for i in sorted(int(k.split("|")[1]) for k in z.files
                                   if k.startswith(name + "|"))]
            for name in names
        }
        metas = {name: _engine_meta(path) for name, path in engines.items() if path.exists()}
        ref = args.reference if args.reference in preds else names[0]
        report = build_report(preds, ref, metas, images_n=len(preds[ref]), side=side)
    else:
        import torch

        from .cli import _load_adapter

        images = load_images(Path(args.images), (side, side), args.limit)
        if not images:
            raise SystemExit(f"no images in {args.images}")
        preds, metas, adapters = {}, {}, {}

        def run(name, fn):
            rows = []
            for _, t in images:
                with torch.no_grad():
                    res = fn(t)
                if hasattr(res, "detach"):
                    res = res.detach().cpu().numpy()
                rows.append(np.asarray(res, dtype=np.float32).reshape(-1, 6))
            preds[name] = rows
            print(f"[report] {name:<22} detections>={OPERATING}: "
                  f"{sum(int((r[:, 4] >= OPERATING).sum()) for r in rows)}", flush=True)

        if args.checkpoint:
            torch_adapter = _load_adapter(Path(args.checkpoint))
            torch_adapter.model.cuda()
            run("torch-fp32", lambda t: torch_adapter.predict([t.cuda()], score_threshold=LOW)[0])
        if args.onnx:
            from onnx_infer import load_onnx_adapter

            ort_adapter, _ = load_onnx_adapter(Path(args.onnx), device=args.onnx_device)
            run("onnxruntime-fp32", lambda t: ort_adapter.predict([t], score_threshold=LOW)[0])
        for name, path in engines.items():
            if not path.exists():
                print(f"[report] skipping {name}: {path} not found")
                continue
            from trt_infer import load_trt_adapter

            adapter, _ = load_trt_adapter(path, "cuda")
            adapters[name] = adapter
            metas[name] = _engine_meta(path)
            run(name, lambda t, a=adapter: a.predict([t], score_threshold=LOW)[0])

        if args.save_preds:
            np.savez(args.save_preds,
                     **{f"{n}|{i}": rows for n, out in preds.items() for i, rows in enumerate(out)})
            print(f"[report] wrote raw detections -> {args.save_preds}")

        latencies = {}
        if args.iters:
            for name, adapter in adapters.items():
                latencies[name] = measure_latency(adapter._model, args.iters, args.warmup,
                                                  (side, side))
        ref = args.reference if args.reference in preds else next(iter(preds))
        report = build_report(preds, ref, metas, images_n=len(images), side=side,
                              latencies=latencies)

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, default=str))
        print(f"[report] wrote {args.out}")
    print_report(report)


def cmd_latency(args) -> None:
    import torch

    from trt_infer import load_trt_adapter

    engines = {n: p for n, p in _parse_engines(args.engine).items() if p.exists()}
    if not engines:
        raise SystemExit("no engines given (or none exist)")

    # The first engine loaded in a process pays TensorRT's one-time runtime init
    # (cuDNN/cuBLAS/plugin registry). Load one and throw it away so no measured
    # engine absorbs that tax.
    first = next(iter(engines.values()))
    prime, _ = load_trt_adapter(first, "cuda")
    measure_latency(prime._model, 20, 10, (args.side, args.side))
    del prime
    torch.cuda.empty_cache()

    out = {}
    for name, path in engines.items():
        adapter, _ = load_trt_adapter(path, "cuda")
        out[name] = {**measure_latency(adapter._model, args.iters, args.warmup,
                                       (args.side, args.side)),
                     **_engine_meta(path)}
        print(f"{name:<22} median {out[name]['median_ms']:>7.3f} ms  "
              f"p95 {out[name]['p95_ms']:>7.3f}  {out[name]['fps']:>7.1f} fps  "
              f"{out[name]['size_mb']:>6.1f} MB", flush=True)
        del adapter
        torch.cuda.empty_cache()

    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2, default=str))
        print(f"wrote {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--engine", action="append", metavar="NAME=PATH",
                        help="A TensorRT engine to include (repeatable)")
    common.add_argument("--side", type=int, default=640, help="Square input side (default 640)")
    common.add_argument("--out", help="Write the report JSON here")

    c = sub.add_parser("compare", parents=[common], help="Numerical comparison across runtimes")
    c.add_argument("--checkpoint", help="Trained .pt — adds the torch-fp32 runtime")
    c.add_argument("--onnx", help="Exported .onnx — adds the onnxruntime-fp32 runtime")
    c.add_argument("--onnx-device", default="cpu")
    c.add_argument("--images", help="Directory of real images")
    c.add_argument("--limit", type=int, default=None)
    c.add_argument("--reference", default="torch-fp32",
                   help="Runtime everything is compared against (default torch-fp32; a "
                        "TensorRT fp32 engine is not necessarily a faithful reference)")
    c.add_argument("--iters", type=int, default=0, help="Latency iterations (0 = skip)")
    c.add_argument("--warmup", type=int, default=200)
    c.add_argument("--save-preds", help="Write every runtime's raw detections to this .npz")
    c.add_argument("--from-preds", help="Skip inference; re-analyse a saved .npz")
    c.set_defaults(func=cmd_compare)

    l = sub.add_parser("latency", parents=[common], help="Engine-only latency for each engine")
    l.add_argument("--iters", type=int, default=500)
    l.add_argument("--warmup", type=int, default=200)
    l.set_defaults(func=cmd_latency)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
