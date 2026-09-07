#!/usr/bin/env python3
"""Compare a built ``.engine`` against the ONNX it came from, on real images.

    python build/verify_engine.py bundle/models/rtmo.engine \
        --onnx bundle/models/rtmo.trt.onnx --images path/to/images

Run this after every build, before the bundle goes anywhere near a camera. A
TensorRT build is a *recompilation*: it can silently pick a tactic that changes
numbers, and the only honest check is running the same preprocessed input
through both runtimes and diffing the outputs. onnxruntime CPU is the reference
because it shares nothing with TensorRT but the graph.

What it reports, per image and in aggregate:

  * **detection count**, both runtimes. A mismatch is the finding — it means NMS
    or the score floor landed differently, and nothing below it is meaningful.
  * **max |Δbox|** in input pixels, and **max |Δscore|**. Sub-pixel and ~1e-4 is
    a normal fp32 recompilation; whole pixels or 1e-2 scores is not.
  * **max |Δkeypoint|**, same idea, over all 17 joints of every person.
  * **posture agreement** — the label each runtime's keypoints produce through
    ``_classify_posture``. This is the number that actually matters downstream,
    because the label is all the app keeps.

Rows are compared in graph order. Both runtimes gather ``dets`` and
``keypoints`` by the same post-NMS indices, so row *i* is the same person in
both — unless the counts already disagree, in which case the comparison is
skipped for that image rather than silently aligned.

Requires: numpy, pillow, onnxruntime, tensorrt, torch.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pose_runtime as pr  # noqa: E402


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("engine", help="The .engine to verify.")
    parser.add_argument("--onnx", default=None,
                        help="Reference graph (default: the engine path with "
                             "'.trt.onnx' in place of '.engine').")
    parser.add_argument("--images", required=True, help="Image, or directory of images.")
    parser.add_argument("--conf", type=float, default=None,
                        help="Confidence floor (default: meta.json's).")
    parser.add_argument("--box-tol", type=float, default=1.0,
                        help="Fail if any box coordinate differs by more than this "
                             "many input pixels.")
    parser.add_argument("--score-tol", type=float, default=1e-2,
                        help="Fail if any score differs by more than this.")
    return parser.parse_args(argv)


def _default_onnx(engine_path: Path) -> Path:
    return engine_path.with_name(engine_path.stem + ".trt.onnx")


def main(argv=None) -> int:
    args = _parse_args(argv)

    engine_path = Path(args.engine)
    onnx_path = Path(args.onnx) if args.onnx else _default_onnx(engine_path)
    for path in (engine_path, onnx_path):
        if not path.exists():
            print(f"FAIL: no such file: {path}", file=sys.stderr)
            return 2

    paths = pr.iter_images(args.images)
    if not paths:
        print(f"FAIL: no images found at {args.images}", file=sys.stderr)
        return 2

    import tensorrt as trt

    print(f"tensorrt     {trt.__version__}")
    trt_session, meta = pr.load_session(engine_path, device="cuda")
    ort_session, _meta = pr.load_session(onnx_path, device="cpu")
    threshold = meta.score_threshold if args.conf is None else args.conf
    print(f"engine       {engine_path.name}")
    print(f"reference    {onnx_path.name} (onnxruntime CPU)")
    print(f"conf         {threshold}\n")

    header = f"{'image':<34}{'trt':>5}{'ort':>5}{'Δbox':>9}{'Δscore':>10}{'Δkpt':>9}  posture"
    print(header)
    print("-" * len(header))

    worst_box = worst_score = worst_kpt = 0.0
    count_mismatches, posture_mismatches, compared = 0, 0, 0

    for path in paths:
        frame = pr.load_image_bgr(path)
        trt_people, (trt_dets, trt_kpts), _t = pr.run_image(trt_session, meta, frame, threshold)
        ort_people, (ort_dets, ort_kpts), _t2 = pr.run_image(ort_session, meta, frame, threshold)

        if trt_dets.shape[0] != ort_dets.shape[0]:
            count_mismatches += 1
            print(f"{path.name[:33]:<34}{trt_dets.shape[0]:>5}{ort_dets.shape[0]:>5}"
                  f"{'—':>9}{'—':>10}{'—':>9}  COUNT MISMATCH")
            continue

        d_box = float(np.abs(trt_dets[:, :4] - ort_dets[:, :4]).max()) if trt_dets.size else 0.0
        d_score = float(np.abs(trt_dets[:, 4] - ort_dets[:, 4]).max()) if trt_dets.size else 0.0
        d_kpt = float(np.abs(trt_kpts[..., :2] - ort_kpts[..., :2]).max()) if trt_kpts.size else 0.0
        worst_box, worst_score = max(worst_box, d_box), max(worst_score, d_score)
        worst_kpt = max(worst_kpt, d_kpt)
        compared += 1

        trt_postures = [person.posture for person in trt_people]
        ort_postures = [person.posture for person in ort_people]
        agree = trt_postures == ort_postures
        posture_mismatches += 0 if agree else 1
        label = ", ".join(f"{trt_postures.count(name)}x {name}"
                          for name in sorted(set(trt_postures))) or "-"
        if not agree:
            label = f"DISAGREE trt={trt_postures} ort={ort_postures}"

        print(f"{path.name[:33]:<34}{len(trt_people):>5}{len(ort_people):>5}"
              f"{d_box:>9.3f}{d_score:>10.2e}{d_kpt:>9.3f}  {label}")

    print("-" * len(header))
    print(f"compared     {compared}/{len(paths)} images")
    print(f"worst Δbox   {worst_box:.3f} px (tolerance {args.box_tol})")
    print(f"worst Δscore {worst_score:.2e} (tolerance {args.score_tol})")
    print(f"worst Δkpt   {worst_kpt:.3f} px")

    failed = False
    if count_mismatches:
        print(f"\nFAIL: {count_mismatches} image(s) produced different detection counts.")
        failed = True
    if worst_box > args.box_tol:
        print(f"\nFAIL: box delta {worst_box:.3f} px exceeds {args.box_tol}.")
        failed = True
    if worst_score > args.score_tol:
        print(f"\nFAIL: score delta {worst_score:.2e} exceeds {args.score_tol}.")
        failed = True
    if posture_mismatches:
        # Not a hard failure on its own: a person sitting exactly on a threshold
        # can flip on a 1e-4 keypoint difference. It is still worth looking at —
        # if it happens on more than a stray image, the thresholds are too tight
        # for that camera, not the build.
        print(f"\nWARN: {posture_mismatches} image(s) disagreed on at least one "
              f"posture label. Check whether those people sit near a threshold "
              f"(vendor/onnx_infer/arch/rtmo.py) before blaming the engine.")

    print("\nFAIL" if failed else "\nOK: the engine matches its ONNX within tolerance.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
