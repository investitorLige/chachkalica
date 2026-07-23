"""Materialize a person-crop pipeline's train/val-loss samples to disk once.

``people_detect_first``/``batch_people`` normally run the frozen person
detector live, per batch, every epoch (:func:`cropping.crop_batch`) — wasted
work, since the detector's output for a given image never changes. This module
runs that same crop step exactly once per dataset per run and writes the
resulting person crops + remapped YOLO labels to a run-scoped cache directory,
so the rest of training reads them back through the ordinary
``num_workers``-parallel :class:`~data.YoloDetectionDataset`/``DataLoader``
path, with no detector in the loop at all.

Only the *loss* consumers (train, and validation loss) can use this: they only
ever need crop images + crop-local targets, never a merge back to full-frame
coordinates. Validation mAP and test prediction still need full frames (they
crop, predict with the *current* epoch's weights, and merge per-crop
predictions back for scoring) — those are sped up separately, via the
per-image person-box cache in ``chachak.pipeline.Pipeline`` (see
``build_run_pipeline``'s ``box_cache``), not by this module.

Alongside the crop images/labels, this also writes ``manifest.jsonl`` (one JSON
object per crop) recording exactly where each crop came from — the source
image, and its pixel offset/size within that source frame. That's what lets
:func:`remap_crop_boxes_to_frame` (and ``scripts/visualize_crop_cache.py``, built
on it) later lift a crop's boxes — ground truth or a model's predictions on the
crop — back onto the original full frame for a human-readable check, the same
geometry :func:`chachak.boxes.remap_local_preds_to_frame` already does for the
live pipeline.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader

try:
    from ..config import DatasetConfig
    from ..data import build_dataset, detection_collate_fn
    from .cropping import crop_batch_regions
except ImportError:  # run as a flat script
    from config import DatasetConfig
    from data import build_dataset, detection_collate_fn
    from preprocess.cropping import crop_batch_regions


_PRECOMPUTE_BATCH_SIZE = 8
_PRECOMPUTE_NUM_WORKERS = 4
MANIFEST_FILENAME = "manifest.jsonl"


def _tensor_to_pil_image(image: torch.Tensor):
    import numpy as np
    from PIL import Image

    array = (image.clamp(0, 1) * 255.0).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(np.ascontiguousarray(array))


def _write_yolo_label(path: Path, target: dict) -> None:
    boxes = target["boxes"]
    labels = target["labels"]
    height, width = (int(value) for value in target["orig_size"].tolist())
    lines = []
    for (x1, y1, x2, y2), label in zip(boxes.tolist(), labels.tolist()):
        cx = (x1 + x2) / 2.0 / width
        cy = (y1 + y2) / 2.0 / height
        box_w = (x2 - x1) / width
        box_h = (y2 - y1) / height
        lines.append(f"{int(label)} {cx:.6f} {cy:.6f} {box_w:.6f} {box_h:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def _crop_name(image_path: str, images_root: Path, index: int) -> str:
    """A filesystem-safe, collision-resistant stem for one crop.

    Prefers the path relative to the dataset's own images root (so two
    identically-named images in different subfolders don't collide); falls
    back to the bare stem if the path isn't actually under ``images_root``
    (shouldn't happen for a real dataset, but keeps this from raising).
    """
    resolved = Path(image_path).resolve()
    try:
        relative = resolved.relative_to(images_root.resolve())
        base = str(relative.with_suffix("")).replace("/", "__").replace("\\", "__")
    except ValueError:
        base = resolved.stem
    return f"{base}__p{index}"


def build_crop_cache(
    dataset_config: DatasetConfig,
    pipeline: Any,
    device: torch.device,
    cache_dir: Path,
) -> DatasetConfig:
    """Run ``dataset_config`` through ``pipeline`` once, caching person crops.

    Writes ``cache_dir/images/*.jpg`` + ``cache_dir/labels/*.txt`` (crop-local
    YOLO labels) + ``cache_dir/manifest.jsonl`` (one JSON record per crop, see
    module docstring), and returns a new :class:`DatasetConfig` (same
    classes/role/augmentation/weight) pointing at the crop images/labels.
    Idempotent: if ``cache_dir/.done`` already exists (written only after a
    complete, uninterrupted pass), the existing cache — manifest included — is
    reused as-is without touching the detector again.
    """
    images_dir = cache_dir / "images"
    labels_dir = cache_dir / "labels"
    manifest_path = cache_dir / MANIFEST_FILENAME
    done_marker = cache_dir / ".done"

    if done_marker.exists():
        print(f"[crop_cache] Reusing existing crop cache for {dataset_config.name}: {cache_dir}")
        return _cached_dataset_config(dataset_config, images_dir, labels_dir)

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    source_dataset = build_dataset(dataset_config)
    loader = DataLoader(
        source_dataset,
        batch_size=_PRECOMPUTE_BATCH_SIZE,
        shuffle=False,
        num_workers=_PRECOMPUTE_NUM_WORKERS,
        collate_fn=detection_collate_fn,
    )

    images_root = Path(dataset_config.images)
    per_image_counts: dict = {}
    total_crops = 0
    started = time.perf_counter()
    print(
        f"[crop_cache] Precomputing person crops: dataset={dataset_config.name} "
        f"role={dataset_config.role} images={len(source_dataset)} -> {cache_dir}"
    )
    with open(manifest_path, "w") as manifest_file:
        for batch_index, (images, targets) in enumerate(loader, start=1):
            images = [image.to(device) for image in images]
            for crop_image, crop_target, source_target, offset, crop_size, frame_size in crop_batch_regions(
                images, targets, pipeline
            ):
                source_path = source_target.get("image_path")
                index = per_image_counts.get(source_path, 0)
                per_image_counts[source_path] = index + 1
                name = _crop_name(source_path, images_root, index)
                _tensor_to_pil_image(crop_image).save(images_dir / f"{name}.jpg", quality=95)
                _write_yolo_label(labels_dir / f"{name}.txt", crop_target)
                manifest_file.write(
                    json.dumps(
                        {
                            "crop_image": f"images/{name}.jpg",
                            "crop_label": f"labels/{name}.txt",
                            "source_image_path": source_path,
                            "offset": list(offset),
                            "crop_size": list(crop_size),
                            "source_frame_size": list(frame_size),
                        }
                    )
                    + "\n"
                )
                total_crops += 1
            if batch_index % 25 == 0:
                print(f"[crop_cache] ...{batch_index} batch(es), {total_crops} crop(s) so far")

    done_marker.write_text("ok\n")
    elapsed = time.perf_counter() - started
    print(
        f"[crop_cache] Done: {total_crops} crop(s) from {len(source_dataset)} image(s) "
        f"in {elapsed:.1f}s -> {cache_dir}"
    )
    return _cached_dataset_config(dataset_config, images_dir, labels_dir)


def _cached_dataset_config(dataset_config: DatasetConfig, images_dir: Path, labels_dir: Path) -> DatasetConfig:
    return replace(
        dataset_config,
        name=f"{dataset_config.name}__crops",
        images=images_dir,
        labels=labels_dir,
    )


def read_manifest(cache_dir: Path) -> List[Dict[str, Any]]:
    """Load ``cache_dir/manifest.jsonl`` as a list of per-crop records."""
    manifest_path = Path(cache_dir) / MANIFEST_FILENAME
    with open(manifest_path) as manifest_file:
        return [json.loads(line) for line in manifest_file if line.strip()]


def read_crop_label_as_preds(label_path: Path) -> torch.Tensor:
    """A crop's YOLO label file as a Friendy ``(N, 6)`` tensor, confidence=1.0.

    Matches the ``[cx, cy, w, h, confidence, class_id]`` shape
    :func:`chachak.boxes.remap_local_preds_to_frame` expects, normalized to the
    crop's own size (as written by :func:`_write_yolo_label`) — ground-truth
    boxes get a placeholder confidence of 1.0 since there's no real score to
    carry. A model's own predictions on a crop are already in this shape and
    don't need this helper.
    """
    label_path = Path(label_path)
    if not label_path.is_file():
        return torch.zeros((0, 6))
    rows = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        class_id = float(parts[0])
        cx, cy, box_w, box_h = (float(value) for value in parts[1:5])
        rows.append([cx, cy, box_w, box_h, 1.0, class_id])
    return torch.tensor(rows, dtype=torch.float32) if rows else torch.zeros((0, 6))


def remap_crop_boxes_to_frame(
    manifest_entry: Dict[str, Any], boxes: torch.Tensor
) -> torch.Tensor:
    """Lift one crop's Friendy ``(N, 6)`` boxes back to full-frame-normalized coords.

    ``boxes`` — ground truth (see :func:`read_crop_label_as_preds`) or a model's
    own predictions on the crop — must already be normalized to the crop's own
    size (``manifest_entry["crop_size"]``). Uses ``manifest_entry``'s
    ``offset``/``crop_size``/``source_frame_size`` (see :func:`build_crop_cache`)
    with the exact geometry the live pipeline uses to merge per-crop predictions
    back onto the frame (:func:`chachak.boxes.remap_local_preds_to_frame`).
    """
    try:
        from chachak.boxes import remap_local_preds_to_frame
    except ImportError:  # chachak not yet on sys.path (see ml.train._ensure_chachak_importable)
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
        from chachak.boxes import remap_local_preds_to_frame

    crop_w, crop_h = manifest_entry["crop_size"]
    frame_w, frame_h = manifest_entry["source_frame_size"]
    offset = tuple(manifest_entry["offset"])
    return remap_local_preds_to_frame(boxes, offset, crop_w, crop_h, frame_w, frame_h)
