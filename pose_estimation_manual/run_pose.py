#!/usr/bin/env python3
"""Run the RTMO pose detector over images. Keypoints kept, posture classified.

    python run_pose.py photo.jpg
    python run_pose.py path/to/images --draw --out results
    python run_pose.py photo.jpg --model bundle/models/rtmo.trt.onnx --device cpu

Arguments
    images          Required, positional. An image, or a directory searched
                    recursively for .jpg/.jpeg/.png/.bmp/.webp/.tif/.tiff.
    --model PATH    .engine (TensorRT, GPU) or .onnx (onnxruntime, CPU-capable).
                    Default: bundle/models/rtmo.engine.
    --conf FLOAT    Confidence floor (default: meta.json's score_threshold, 0.4).
                    The graph already dropped everything under 0.15 itself.
    --device STR    "cuda" (default) or "cpu". "cpu" only works with --model *.onnx.
    --out DIR       Where poses.json is written (default: "results").
    --draw          Also write annotated copies under "<out>/annotated/".
    --quiet         Don't print per-image lines.

Writes ``<out>/poses.json``::

    {"model": "...", "conf": 0.4, "images": [
      {"image": "a.jpg", "width": 1920, "height": 1080, "people": [
        {"score": 0.91, "posture": "standing", "posture_id": 0,
         "box": [x1, y1, x2, y2],              pixels, xyxy, in the source image
         "box_normalized": [x1, y1, x2, y2],   the same box over [0, 1]
         "keypoints": {"nose": [x, y, score], ... 17 COCO joints ...}}
      ]}
    ]}

The app itself keeps only ``posture`` (as a class label) and the box — see
README.md, "What the app actually stores". This script exists because the
keypoints are worth looking at when a posture call is wrong.

Requires: numpy, pillow, and either tensorrt+torch (.engine) or onnxruntime (.onnx).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pose_runtime as pr  # noqa: E402

DEFAULT_MODEL = ROOT / "bundle" / "models" / "rtmo.engine"

# Posture -> BGR-ish RGB colour for --draw. Lying is red on purpose: it is the
# one this bundle exists to notice.
POSTURE_COLORS = {
    "standing": (60, 200, 90),
    "sitting": (250, 190, 60),
    "lying": (235, 70, 60),
}


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("images")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="results")
    parser.add_argument("--draw", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def _draw(frame_bgr, people, out_path: Path) -> None:
    from PIL import Image, ImageDraw

    rgb = frame_bgr[:, :, ::-1]
    image = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(image)
    for person in people:
        color = POSTURE_COLORS.get(person.posture, (200, 200, 200))
        x1, y1, x2, y2 = person.box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1 + 4, max(0.0, y1 - 12)), f"{person.posture} {person.score:.2f}",
                  fill=color)
        kpts = person.keypoints
        for a, b in pr.SKELETON:
            if kpts[a, 2] >= pr.KPT_SCORE_MIN and kpts[b, 2] >= pr.KPT_SCORE_MIN:
                draw.line([kpts[a, 0], kpts[a, 1], kpts[b, 0], kpts[b, 1]],
                          fill=color, width=2)
        for x, y, score in kpts:
            if score >= pr.KPT_SCORE_MIN:
                draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=color)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def main(argv=None) -> int:
    args = _parse_args(argv)

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"FAIL: no such model: {model_path}", file=sys.stderr)
        return 2
    if model_path.suffix.lower() == ".engine" and args.device.lower().startswith("cpu"):
        print("FAIL: a TensorRT .engine is GPU-only. Pass --model "
              "bundle/models/rtmo.trt.onnx for a CPU run.", file=sys.stderr)
        return 2

    paths = pr.iter_images(args.images)
    if not paths:
        print(f"FAIL: no images found at {args.images}", file=sys.stderr)
        return 2

    started = time.monotonic()
    session, meta = pr.load_session(model_path, device=args.device)
    threshold = meta.score_threshold if args.conf is None else args.conf
    print(f"loaded       {model_path.name} arch={meta.arch} "
          f"input={meta.input.resize_mode}/{meta.input.size} conf={threshold} "
          f"in {time.monotonic() - started:.1f}s")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    records, timings = [], []

    for path in paths:
        frame = pr.load_image_bgr(path)
        height, width = frame.shape[0], frame.shape[1]
        began = time.monotonic()
        people, _raw, _transform = pr.run_image(session, meta, frame, threshold)
        timings.append((time.monotonic() - began) * 1000)
        records.append({
            "image": str(path),
            "width": width,
            "height": height,
            "people": [person.to_dict(width, height) for person in people],
        })
        if not args.quiet:
            counts = {}
            for person in people:
                counts[person.posture] = counts.get(person.posture, 0) + 1
            summary = ", ".join(f"{n}x {name}" for name, n in sorted(counts.items())) or "-"
            print(f"  {path.name:<40} {len(people):>2} people  {summary:<32} "
                  f"{timings[-1]:>6.0f} ms")
        if args.draw:
            _draw(frame, people, out_dir / "annotated" / path.name)

    payload = {"model": str(model_path), "conf": threshold, "images": records}
    (out_dir / "poses.json").write_text(json.dumps(payload, indent=2) + "\n")
    total_people = sum(len(record["people"]) for record in records)
    print(f"\nwrote        {out_dir / 'poses.json'} — {len(records)} images, "
          f"{total_people} people")
    if len(timings) > 1:
        print(f"inference    {min(timings[1:]):.0f} ms best of {len(timings) - 1} "
              f"(the first includes warm-up: {timings[0]:.0f} ms)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
