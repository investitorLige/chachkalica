#!/usr/bin/env python3
"""Run this bundle's detection pipeline on images.

    python infer.py path/to/images
    python infer.py path/to/images --conf 0.4 --save-images

Arguments:
    images            Required, positional. A single image file or a directory,
                       searched recursively for .jpg/.jpeg/.png/.bmp/.webp/.tif/.tiff.

    --out DIR          Where results are written (default: "results", created if
                        missing). See "Output" below for what ends up in it.

    --conf FLOAT        Confidence threshold for detections. Default: the value the
                        bundle was exported with (see pipeline.json). Lower = more,
                        noisier detections; higher = fewer, more confident ones.

    --detector-conf FLOAT   Confidence threshold for the person detector, only for
                        pipelines that crop people before running the model(s).
                        Default: the bundle's exported value. Ignored for pipelines
                        that don't detect people first.

    --save-images        Also write annotated copies of each image (boxes + labels
                        burned in) under "<out>/annotated/".

    --device STR         "auto" (default), "cpu", "cuda", "cuda:1", ...

    --format {onnx,engine}  Assert the bundle carries this format; the bundle only
                        ever carries one (whichever it was exported with), so this
                        just fails loudly if you guessed wrong rather than
                        picking anything.

    --batch-size INT     Images per forward pass. Default: the bundle's exported
                        value, possibly capped by the artifacts' profile (TRT
                        engines are built for a fixed max batch).

Every other setting — which models run, tiling geometry, person-crop expansion, the
NMS that merges overlapping regions — is fixed in ``pipeline.json`` and was tuned
together with the weights; changing it changes what the pipeline detects.

Output ("<out>/detections.json"):
    {
      "bundle": {...},              # name/version of the exported bundle
      "format": "onnx" | "engine",
      "conf": 0.4,                  # confidence threshold actually used
      "detector_conf": 0.5 | null,
      "device": "cuda:0",
      "classes": {"0": "hardhat", ...},
      "images": [
        {
          "image": "relative/path.jpg",
          "width": 1920, "height": 1080,
          "detections": [
            {
              "class_id": 0, "class_name": "hardhat", "confidence": 0.93,
              "box_xyxy": [x1, y1, x2, y2],     # pixel coords in the source image
              "box_xywhn": [cx, cy, w, h]        # normalized 0..1, center + size
            }, ...
          ]
        }, ...
      ]
    }
    With --save-images, annotated copies are additionally written to
    "<out>/annotated/<relative image path>".

Requires: torch, numpy, pillow, pyyaml, onnxruntime (see requirements.txt).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

BUNDLE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BUNDLE_DIR / "runtime"))

import bundle_manifest as manifest_contract  # noqa: E402
from chachak.config import pipeline_config_from_dict  # noqa: E402
from chachak.detector import load_detector  # noqa: E402
from chachak.infer import load_checkpoint_adapter  # noqa: E402
from chachak._friendy import (  # noqa: E402
    remap_raw_predictions_to_eval_classes,
    resolve_device,
)
from chachak.boxes import merge_predictions  # noqa: E402
from chachak.pipeline import _frame_size  # noqa: E402
from chachak.preview import _load_image_tensor, _to_box_dicts  # noqa: E402
from chachak.registry import build_pipeline  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Distinguishable at small sizes and on both light and dark photo content; cycled
# by class id so a class keeps its color across images and runs.
PALETTE = [
    (232, 62, 74), (46, 158, 234), (58, 190, 120), (247, 176, 46),
    (166, 106, 232), (38, 198, 196), (240, 120, 180), (150, 170, 60),
]


def _list_images(source: Path) -> List[Path]:
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise SystemExit(f"[infer] No such file or directory: {source}")
    found = sorted(
        path for path in source.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not found:
        raise SystemExit(f"[infer] No images found under {source}")
    return found


def _needs_detector(pipeline_name: str, chain: List[str]) -> bool:
    people = {"people_detect_first", "batch_people"}
    if pipeline_name in people:
        return True
    return pipeline_name == "chain" and any(child in people for child in chain)


def build_runtime(config, device, detector_path: Optional[Path]):
    """Load every bundled model (plus the detector) and build one pipeline each.

    Returns ``[(pipeline, class_map), ...]``: one entry per model. A bundle with
    ``extra_models`` runs each through the same pipeline and merges their
    predictions per frame, mirroring the training repo's combined-eval path — only
    valid because such models were exported with disjoint class names.
    """
    detector = None
    if detector_path is not None:
        detector = load_detector(
            detector_path,
            device,
            person_class_name=config.detector.person_class_name,
            person_class_id=config.detector.person_class_id,
            score_threshold=config.detector.score_threshold,
            batch_size=config.detector.batch_size,
        )

    # One person-box cache shared by every model's pipeline: the detector is frozen,
    # so re-running it per model on the same frame would only burn time.
    box_cache: Dict[Any, Any] = {}
    runtimes = []
    for path in [config.model_checkpoint, *config.extra_checkpoints]:
        adapter, info = load_checkpoint_adapter(path, device)
        classes = info.get("train_classes") or {}
        if not classes:
            raise SystemExit(
                f"[infer] {path.name} carries no class names in its meta.json, so its "
                f"predictions cannot be mapped onto the pipeline's class space. "
                f"Re-export the bundle from a checkpoint that records class names."
            )
        runtimes.append(
            (build_pipeline(config, adapter, device, detector, box_cache=box_cache), classes)
        )
    return runtimes


def run_batch(runtimes, config, images, image_paths):
    """Predictions for one batch of frames, in the pipeline's class space."""
    targets = [{"image_path": str(path)} for path in image_paths]
    frame_sizes = [_frame_size(image) for image in images]

    per_model = []
    for pipeline, classes in runtimes:
        per_model.append([
            remap_raw_predictions_to_eval_classes(
                prediction.detach().cpu(), classes, config.classes
            )
            for prediction in pipeline.process_batch(images, targets)
        ])

    if len(per_model) == 1:
        return per_model[0]
    return [
        merge_predictions(
            [per_model[m][i] for m in range(len(per_model))],
            *frame_sizes[i],
            config.merge_nms_iou,
        )
        for i in range(len(images))
    ]


def _detections(prediction, class_map, width: int, height: int) -> List[Dict[str, Any]]:
    """Friendy ``(N, 6)`` rows -> JSON dicts with both pixel and normalized boxes."""
    results = []
    for box in _to_box_dicts(prediction, class_map):
        cx, cy, w, h = box["cx"], box["cy"], box["w"], box["h"]
        results.append(
            {
                "class_id": box["class_id"],
                "class_name": box["class_name"],
                "confidence": round(float(box["confidence"]), 5),
                # xyxy in the source image's own pixels — what a consumer draws or
                # crops with. Rounded to 0.1px; the underlying box is normalized.
                "box_xyxy": [
                    round((cx - w / 2) * width, 1),
                    round((cy - h / 2) * height, 1),
                    round((cx + w / 2) * width, 1),
                    round((cy + h / 2) * height, 1),
                ],
                "box_xywhn": [round(cx, 6), round(cy, 6), round(w, 6), round(h, 6)],
            }
        )
    results.sort(key=lambda item: item["confidence"], reverse=True)
    return results


def _annotate(image_path: Path, detections: List[Dict[str, Any]], out_path: Path) -> None:
    from PIL import Image, ImageDraw

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    # Scale line/text with the image so a 4K frame isn't annotated with hairlines.
    width = max(2, round(min(image.size) / 400))
    for detection in detections:
        x1, y1, x2, y2 = detection["box_xyxy"]
        color = PALETTE[detection["class_id"] % len(PALETTE)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        label = f"{detection['class_name']} {detection['confidence']:.2f}"
        text_box = draw.textbbox((x1, y1), label)
        pad = width
        # Flip the caption inside the frame when the box touches the top edge.
        text_h = text_box[3] - text_box[1] + 2 * pad
        top = y1 - text_h if y1 - text_h >= 0 else y1
        draw.rectangle(
            [x1, top, x1 + (text_box[2] - text_box[0]) + 2 * pad, top + text_h], fill=color
        )
        draw.text((x1 + pad, top + pad), label, fill=(255, 255, 255))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run this bundle's detection pipeline on images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("images", help="Image file, or a directory searched recursively")
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Confidence threshold for detections (default: the bundle's exported value)",
    )
    parser.add_argument("--out", default="results", help="Directory for detections.json")
    parser.add_argument(
        "--save-images", action="store_true", help="Also write annotated copies of each image"
    )
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, cuda:1, ...")
    parser.add_argument("--format", choices=manifest_contract.FORMATS, default=None,
                        help="Assert the bundle carries this artifact format (default: "
                             "whichever it carries); errors out if it carries the other")
    parser.add_argument("--batch-size", type=int, default=None, help="Images per forward pass")
    parser.add_argument(
        "--detector-conf",
        type=float,
        default=None,
        help="Person-detector threshold, for pipelines that crop people "
             "(default: the bundle's exported value)",
    )
    args = parser.parse_args()

    manifest = manifest_contract.load_manifest(BUNDLE_DIR / "pipeline.json")
    # The flat manifest carries exactly one artifact set — resolve_format just
    # reports which format it is (from the model path's extension) and raises if
    # --format asks for the other one; there is nothing to fall back to.
    fmt = manifest_contract.resolve_format(manifest, args.format)

    paths = manifest_contract.artifact_paths(manifest, BUNDLE_DIR, fmt)
    if not _needs_detector(manifest.get("pipeline"), manifest.get("chain") or []):
        paths["detector"] = None

    output_dir = Path(args.out).resolve()
    request = manifest_contract.pipeline_request_from_manifest(
        manifest,
        paths,
        fmt=fmt,
        images=args.images,
        output_dir=output_dir,
        conf=args.conf,
        detector_conf=args.detector_conf,
        device=args.device,
        batch_size=args.batch_size,
    )
    # Validated by the same loader the training repo uses, so a bundle can never
    # run a pipeline configuration that repo would reject.
    config = pipeline_config_from_dict(request, BUNDLE_DIR)

    for line in manifest_contract.describe(manifest):
        print(f"[infer] {line}")
    print(f"[infer] format={fmt} conf={config.score_threshold}")
    exported_batch = manifest.get("infer_batch_size")
    if args.batch_size is None and exported_batch and config.infer_batch_size < exported_batch:
        print(
            f"[infer] batch capped at {config.infer_batch_size} (from {exported_batch}) by the "
            f"{fmt} artifacts' profile — throughput only, detections are unaffected"
        )

    device = resolve_device(config.device)
    runtimes = build_runtime(config, device, paths["detector"])

    source = Path(args.images).resolve()
    image_paths = _list_images(source)
    print(f"[infer] {len(image_paths)} image(s) from {source}")

    output_dir.mkdir(parents=True, exist_ok=True)
    class_map = manifest_contract.class_map(manifest)
    records = []
    total = 0
    batch = max(1, config.infer_batch_size)
    for start in range(0, len(image_paths), batch):
        chunk = image_paths[start : start + batch]
        images = [_load_image_tensor(path, device) for path in chunk]
        predictions = run_batch(runtimes, config, images, chunk)
        for path, image, prediction in zip(chunk, images, predictions):
            frame_w, frame_h = _frame_size(image)
            detections = _detections(prediction.detach().cpu(), class_map, frame_w, frame_h)
            total += len(detections)
            relative = path.relative_to(source) if source.is_dir() else Path(path.name)
            records.append(
                {
                    "image": str(relative),
                    "width": frame_w,
                    "height": frame_h,
                    "detections": detections,
                }
            )
            if args.save_images:
                _annotate(path, detections, output_dir / "annotated" / relative)
        print(f"[infer] {min(start + batch, len(image_paths))}/{len(image_paths)} images, {total} detections")

    result_path = output_dir / "detections.json"
    result_path.write_text(
        json.dumps(
            {
                "bundle": manifest.get("bundle"),
                "format": fmt,
                "conf": config.score_threshold,
                "detector_conf": (
                    config.detector.score_threshold if paths["detector"] is not None else None
                ),
                "device": str(device),
                "classes": {str(key): value for key, value in class_map.items()},
                "images": records,
            },
            indent=2,
        )
    )
    print(f"[infer] Wrote {result_path} ({len(records)} images, {total} detections)")
    if args.save_images:
        print(f"[infer] Wrote annotated images to {output_dir / 'annotated'}")


if __name__ == "__main__":
    main()
