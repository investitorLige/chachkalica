#!/usr/bin/env python3
"""Compose a weapon dataset out of person crops, falling back to whole frames.

Given one or more YOLO source datasets (``<root>/<name>/{classes.txt,images,labels}``),
this runs the frozen YOLOX person detector over every image and emits, per image:

* **one crop per person who is holding something labelled** — the person box padded
  outward by ``--expand-ratio`` and then stretched, edge by edge, until it also
  covers every weapon box that sits on that person. A weapon counts as "on" a
  person when at least ``--min-inside-fraction`` of the *weapon box's own area*
  falls inside the (unexpanded) person box, so a gun that pokes out of the box
  still claims its holder — the window then grows out to the gun's far edge, and
  the crop never contains a clipped version of the object it was built for.
* **the whole frame**, whenever some label is *not* claimed by any person (no
  person box overlaps it enough, or the detector found no people at all), and for
  images with no labels at all — negatives pass straight through. A frame with
  both a claimed and an unclaimed weapon yields both the crop and the whole frame,
  since dropping either would lose a real training signal.

Every emitted sample carries *all* labels visible in it, not just the ones that
built the window: an unlabelled weapon in frame is worse than a redundant one, so
a second weapon caught by a crop is kept (clipped, if at least
``--min-visible-fraction`` of it survives the clip). This is the same floor
``preprocess.cropping`` uses for train-time person crops.

Source class *names* drive the merge, not ids: each source's own ``classes.txt``
is read, each name is normalized (lowercased, then passed through ``--alias``)
and looked up in ``--classes``. A name that resolves to nothing is dropped and
counted, which is how ``background`` negatives and stray out-of-range ids fall
away without special-casing.

Requires the repo root on ``PYTHONPATH`` (it imports ``chachak`` alongside
``friendy_chachkalica``) and a GPU for the TensorRT detector — i.e. the trainer
image. Example, both weapon prototypes:

    docker run --rm --gpus all --shm-size=2g \\
      -v "$PWD":/repo -v "$PWD/chachkalica/data":/app/data -v "$PWD/models":/app/models:ro \\
      -e PYTHONPATH=/repo chackalica-trainer \\
      python /repo/friendy_chachkalica/scripts/build_weapon_person_crop_dataset.py \\
        --source-root /app/data/source --out-name prototype_training_weapon \\
        --datasets ns_weapon_detection ns_weapon_detection_web \\
        --classes handgun rifle knife bat --alias gun=handgun \\
        --person-box-cache /app/data/_dbg/person_boxes
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader

from chachak.boxes import expand_box, xywhn_preds_to_xyxy
from chachak.detector import load_detector

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
MANIFEST_FILENAME = "manifest.jsonl"
REPORT_FILENAME = "build_report.json"

Box = List[float]
Label = Tuple[int, Box]


# ──────────────────────────────── class merging ────────────────────────────────

class ClassMerger:
    """Maps a source dataset's class ids onto the output class list, by name."""

    def __init__(self, out_classes: Sequence[str], aliases: Dict[str, str]) -> None:
        self.out_classes = list(out_classes)
        self._by_name = {name.strip().lower(): index for index, name in enumerate(self.out_classes)}
        self._aliases = {key.strip().lower(): value.strip().lower() for key, value in aliases.items()}

    def out_id(self, source_name: str) -> Optional[int]:
        key = source_name.strip().lower()
        return self._by_name.get(self._aliases.get(key, key))

    def id_table(self, source_classes: Sequence[str]) -> Dict[int, Optional[int]]:
        """``{source class id: output class id or None}`` for one dataset."""
        return {index: self.out_id(name) for index, name in enumerate(source_classes)}


def read_classes_file(path: Path) -> List[str]:
    """Class names from a ``classes.txt``, ignoring blanks and ``#`` comments.

    The comment skip matters: one source ships a leading ``# tools: bbox, ...``
    line, and counting it as a class would shift every id by one.
    """
    if not path.is_file():
        return []
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


# ─────────────────────────────────── geometry ──────────────────────────────────

def read_yolo_labels(path: Path, width: int, height: int) -> List[Tuple[int, Box]]:
    """A YOLO ``.txt`` as ``[(source_class_id, [x1, y1, x2, y2])]`` in pixels."""
    if not path.is_file():
        return []
    rows: List[Tuple[int, Box]] = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
            cx, cy, box_w, box_h = (float(value) for value in parts[1:5])
        except ValueError:
            continue
        rows.append((
            class_id,
            [
                (cx - box_w / 2) * width,
                (cy - box_h / 2) * height,
                (cx + box_w / 2) * width,
                (cy + box_h / 2) * height,
            ],
        ))
    return rows


def box_area(box: Box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def intersection_area(a: Box, b: Box) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def inside_fraction(inner: Box, outer: Box) -> float:
    """Fraction of ``inner``'s own area that falls inside ``outer``."""
    area = box_area(inner)
    return intersection_area(inner, outer) / area if area > 0 else 0.0


def union_box(a: Box, b: Box) -> Box:
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def integer_window(box: Box, width: int, height: int) -> Tuple[int, int, int, int]:
    """Snap a window to whole pixels, rounding *outward* and clipped to the frame.

    Outward (floor/ceil rather than round) so a weapon box the window was just
    grown to cover cannot be shaved back off by the rounding.
    """
    x1 = int(min(max(math.floor(box[0]), 0), max(width - 1, 0)))
    y1 = int(min(max(math.floor(box[1]), 0), max(height - 1, 0)))
    x2 = int(min(max(math.ceil(box[2]), x1 + 1), width))
    y2 = int(min(max(math.ceil(box[3]), y1 + 1), height))
    return x1, y1, x2, y2


def labels_in_window(
    labels: Sequence[Label],
    window: Tuple[int, int, int, int],
    min_visible_fraction: float,
) -> List[Label]:
    """Every label visible in ``window``, clipped, in window-local pixels.

    A label whose surviving area is under ``min_visible_fraction`` of its
    original is dropped rather than kept as a sliver.
    """
    wx1, wy1, wx2, wy2 = window
    kept: List[Label] = []
    for class_id, box in labels:
        area = box_area(box)
        if area <= 0:
            continue
        x1, y1 = max(box[0], wx1), max(box[1], wy1)
        x2, y2 = min(box[2], wx2), min(box[3], wy2)
        if x2 <= x1 or y2 <= y1:
            continue
        if ((x2 - x1) * (y2 - y1)) / area < min_visible_fraction:
            continue
        kept.append((class_id, [x1 - wx1, y1 - wy1, x2 - wx1, y2 - wy1]))
    return kept


def yolo_lines(labels: Sequence[Label], width: int, height: int) -> List[str]:
    lines = []
    for class_id, (x1, y1, x2, y2) in labels:
        box_w, box_h = (x2 - x1) / width, (y2 - y1) / height
        if box_w <= 0 or box_h <= 0:
            continue
        cx, cy = (x1 + x2) / 2.0 / width, (y1 + y2) / 2.0 / height
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {box_w:.6f} {box_h:.6f}")
    return lines


# ─────────────────────────────── window planning ───────────────────────────────

def plan_windows(
    labels: Sequence[Label],
    person_boxes: Sequence[Box],
    width: int,
    height: int,
    expand_ratio: float,
    min_inside_fraction: float,
) -> Tuple[List[Tuple[int, int, int, int]], List[int]]:
    """``(crop windows, indices of labels no person claimed)``.

    One window per person who claims at least one label. The window starts as the
    person box padded by ``expand_ratio`` and is then unioned with each claimed
    label's *full* box, so a weapon that only partly overlapped the person is
    still framed whole. Windows that come out identical (two heavily overlapping
    person boxes around the same person) are emitted once.
    """
    windows: List[Tuple[int, int, int, int]] = []
    claimed: set = set()
    for person_box in person_boxes:
        hits = [
            index
            for index, (_class_id, box) in enumerate(labels)
            if inside_fraction(box, person_box) >= min_inside_fraction
        ]
        if not hits:
            continue
        window = expand_box(person_box, expand_ratio, width, height)
        for index in hits:
            window = union_box(window, labels[index][1])
        windows.append(integer_window(window, width, height))
        claimed.update(hits)

    deduped = list(dict.fromkeys(windows))
    unclaimed = [index for index in range(len(labels)) if index not in claimed]
    return deduped, unclaimed


# ──────────────────────────────── source reading ───────────────────────────────

class SourceImages(torch.utils.data.Dataset):
    """Decodes one source dataset's images (and raw labels) in DataLoader workers."""

    def __init__(self, images_dir: Path, labels_dir: Path, relative_paths: Sequence[Path]) -> None:
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.relative_paths = list(relative_paths)

    def __len__(self) -> int:
        return len(self.relative_paths)

    def __getitem__(self, index: int) -> dict:
        relative = self.relative_paths[index]
        path = self.images_dir / relative
        try:
            # exif_transpose before anything reads the size: the source labels are
            # normalized against the displayed orientation, as data.py assumes too.
            image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        except OSError as exc:
            return {"relative": relative, "error": str(exc)}
        array = np.asarray(image)
        height, width = array.shape[:2]
        return {
            "relative": relative,
            "array": array,
            "width": width,
            "height": height,
            "raw_labels": read_yolo_labels(
                (self.labels_dir / relative).with_suffix(".txt"), width, height
            ),
        }


def list_images(images_dir: Path) -> List[Path]:
    return sorted(
        path.relative_to(images_dir)
        for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def identity_collate(batch: List[dict]) -> List[dict]:
    return batch


# ──────────────────────────────── person boxes ─────────────────────────────────

class PersonBoxCache:
    """Per-source-dataset JSON of raw (unexpanded) person boxes, in frame pixels.

    Keyed by the image's path relative to the dataset's ``images/``. Cached boxes
    are detector output only — expansion and weapon-union happen downstream — so a
    second build with a different ``--expand-ratio`` still reuses the GPU work,
    while a different detector or score threshold correctly invalidates it.
    """

    def __init__(self, directory: Optional[Path], detector_key: str) -> None:
        self.directory = directory
        self.detector_key = detector_key
        self._boxes: Dict[str, List[Box]] = {}
        self._path: Optional[Path] = None
        self._dirty = False

    def open(self, dataset_name: str) -> None:
        self._boxes, self._dirty = {}, False
        if self.directory is None:
            self._path = None
            return
        self._path = self.directory / f"{dataset_name}.json"
        if not self._path.is_file():
            return
        try:
            payload = json.loads(self._path.read_text())
        except (OSError, ValueError):
            return
        if payload.get("detector_key") != self.detector_key:
            print(f"[build] person-box cache for {dataset_name} was built differently; recomputing")
            return
        self._boxes = payload.get("boxes") or {}
        print(f"[build] reusing {len(self._boxes)} cached person-box row(s) for {dataset_name}")

    def get(self, key: str) -> Optional[List[Box]]:
        return self._boxes.get(key)

    def put(self, key: str, boxes: List[Box]) -> None:
        # 0.1px is well under the rounding the crop window applies anyway, and it
        # keeps a 56k-image cache in the low megabytes.
        self._boxes[key] = [[round(value, 1) for value in box] for box in boxes]
        self._dirty = True

    def flush(self) -> None:
        if self._path is None or not self._dirty:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps({"detector_key": self.detector_key, "boxes": self._boxes})
        )
        self._dirty = False


def detect_person_boxes(detector, samples: Sequence[dict], device: torch.device) -> List[List[Box]]:
    """Raw person boxes per sample, in each sample's own frame pixels."""
    if not samples:
        return []
    images = [
        torch.from_numpy(np.ascontiguousarray(sample["array"])).permute(2, 0, 1).to(
            device=device, dtype=torch.float32
        ).div_(255.0)
        for sample in samples
    ]
    with torch.no_grad():
        predictions = detector.predict(images)
    return [
        xywhn_preds_to_xyxy(preds, sample["width"], sample["height"]).tolist()
        for preds, sample in zip(predictions, samples)
    ]


# ───────────────────────────────── emission ────────────────────────────────────

def output_stem(dataset_name: str, relative: Path) -> str:
    """A stem for a sample pulled out of ``dataset_name``.

    Sources merge into one flat ``images/``, so the source name has to be in the
    stem — except where the files already carry it (the RadojkaLakic exports name
    themselves), which would otherwise double it up.

    Two sources that hold the *same* image can still land on one stem. That is not
    assumed to be benign: :func:`build_one_source` refuses the second write and
    says so, rather than letting one source quietly overwrite another.
    """
    base = str(relative.with_suffix("")).replace("/", "__").replace("\\", "__")
    return base if base.startswith(dataset_name) else f"{dataset_name}__{base}"


class LinkSource:
    """An already-built sibling dataset whose identical images can be hardlinked.

    Two prototypes over the same sources differ only in their class mapping, so
    their pixels are byte-identical and only the ``.txt`` files change. Linking
    instead of re-encoding halves the disk. The window and source path recorded in
    the sibling's manifest are checked before any link, so a build whose geometry
    actually *did* change falls back to writing real pixels rather than silently
    adopting the wrong crop.
    """

    def __init__(self, dataset_dir: Optional[Path]) -> None:
        self.images_dir = None if dataset_dir is None else dataset_dir / "images"
        self._records: Dict[str, dict] = {}
        if dataset_dir is None:
            return
        manifest = dataset_dir / MANIFEST_FILENAME
        if not manifest.is_file():
            print(f"[build] --link-images-from has no {MANIFEST_FILENAME}; ignoring it")
            self.images_dir = None
            return
        with open(manifest) as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    self._records[record["image"]] = record
        print(f"[build] {len(self._records)} linkable image(s) from {dataset_dir}")

    def try_link(self, name: str, destination: Path, source_image: str, window: Optional[list]) -> bool:
        if self.images_dir is None:
            return False
        record = self._records.get(name)
        if record is None or record.get("source_image") != source_image or record.get("window") != window:
            return False
        candidate = self.images_dir / name
        if not candidate.is_file():
            return False
        try:
            os.link(candidate, destination)
        except OSError:
            shutil.copy2(candidate, destination)
        return True


def save_crop(array: np.ndarray, window: Tuple[int, int, int, int], destination: Path, quality: int) -> None:
    x1, y1, x2, y2 = window
    Image.fromarray(np.ascontiguousarray(array[y1:y2, x1:x2])).save(destination, quality=quality)


# ─────────────────────────────────── preview ───────────────────────────────────

_PREVIEW_COLORS = [(239, 83, 80), (66, 165, 245), (102, 187, 106), (255, 202, 40), (171, 71, 188)]


def save_preview(image_path: Path, labels: Sequence[Label], classes: Sequence[str], destination: Path) -> None:
    """Draw an emitted sample's own labels back onto it, for eyeballing."""
    from PIL import ImageDraw

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width = max(2, round(min(image.size) / 300))
    for class_id, (x1, y1, x2, y2) in labels:
        color = _PREVIEW_COLORS[class_id % len(_PREVIEW_COLORS)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        name = classes[class_id] if class_id < len(classes) else str(class_id)
        draw.text((x1 + width, y1 + width), name, fill=color)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, quality=90)


# ──────────────────────────────────── build ────────────────────────────────────

def build_dataset(args: argparse.Namespace) -> dict:
    source_root = Path(args.source_root)
    out_dir = Path(args.out_root or source_root) / args.out_name
    images_out, labels_out = out_dir / "images", out_dir / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    (out_dir / "classes.txt").write_text("\n".join(args.classes) + "\n")

    merger = ClassMerger(args.classes, dict(alias.split("=", 1) for alias in args.alias))
    link_source = LinkSource(Path(args.link_images_from) if args.link_images_from else None)
    preview_dir = Path(args.preview_dir) if args.preview_dir else out_dir.parent / f"{out_dir.name}_preview"

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    detector = load_detector(
        args.detector,
        device,
        person_class_id=args.person_class_id,
        score_threshold=args.person_score_threshold,
        batch_size=args.detector_batch_size,
    )
    box_cache = PersonBoxCache(
        Path(args.person_box_cache) if args.person_box_cache else None,
        detector_key=f"{Path(args.detector).name}@{args.person_score_threshold}",
    )

    report = {
        "output": str(out_dir),
        "classes": list(args.classes),
        "aliases": list(args.alias),
        "expand_ratio": args.expand_ratio,
        "min_inside_fraction": args.min_inside_fraction,
        "min_visible_fraction": args.min_visible_fraction,
        "person_score_threshold": args.person_score_threshold,
        "datasets": [],
    }
    previews_left = args.preview
    # Output stem -> the source image that claimed it, shared across every source
    # so a later dataset can never overwrite an earlier one's sample.
    claimed_stems: Dict[str, str] = {}
    started = time.perf_counter()

    with open(out_dir / MANIFEST_FILENAME, "w") as manifest:
        for dataset_name in args.datasets:
            report["datasets"].append(
                build_one_source(
                    dataset_name=dataset_name,
                    dataset_dir=source_root / dataset_name,
                    out_dir=out_dir,
                    manifest=manifest,
                    merger=merger,
                    detector=detector,
                    device=device,
                    box_cache=box_cache,
                    link_source=link_source,
                    args=args,
                    preview_dir=preview_dir,
                    preview_budget=previews_left,
                    claimed_stems=claimed_stems,
                )
            )
            previews_left = max(0, previews_left - report["datasets"][-1]["previews_written"])

    report["elapsed_seconds"] = round(time.perf_counter() - started, 1)
    report["totals"] = {
        key: sum(entry[key] for entry in report["datasets"])
        for key in ("source_images", "crops", "frames", "boxes_written", "boxes_dropped", "unreadable", "stem_collisions")
    }
    (out_dir / REPORT_FILENAME).write_text(json.dumps(report, indent=2) + "\n")
    return report


def build_one_source(
    *,
    dataset_name: str,
    dataset_dir: Path,
    out_dir: Path,
    manifest,
    merger: ClassMerger,
    detector,
    device: torch.device,
    box_cache: PersonBoxCache,
    link_source: LinkSource,
    args: argparse.Namespace,
    preview_dir: Path,
    preview_budget: int,
    claimed_stems: Dict[str, str],
) -> dict:
    images_dir, labels_dir = dataset_dir / "images", dataset_dir / "labels"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"{dataset_dir} has no images/ directory")

    source_classes = read_classes_file(dataset_dir / "classes.txt")
    id_table = merger.id_table(source_classes)
    relative_paths = list_images(images_dir)
    if args.limit:
        relative_paths = relative_paths[: args.limit]

    print(
        f"[build] {dataset_name}: {len(relative_paths)} image(s), "
        f"classes {source_classes} -> "
        f"{ {source_classes[i]: (merger.out_classes[o] if o is not None else None) for i, o in id_table.items()} }"
    )

    loader = DataLoader(
        SourceImages(images_dir, labels_dir, relative_paths),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=identity_collate,
    )
    box_cache.open(dataset_name)

    stats = Counter()
    unknown_ids = Counter()
    previews_written = 0
    images_out, labels_out = out_dir / "images", out_dir / "labels"

    for batch_index, batch in enumerate(loader, start=1):
        samples = []
        for sample in batch:
            if "error" in sample:
                stats["unreadable"] += 1
                print(f"[build] WARNING: skipping unreadable {images_dir / sample['relative']}: {sample['error']}")
                continue
            samples.append(sample)
        stats["source_images"] += len(samples)

        # Only images the cache has never seen reach the detector.
        needing_detection = [s for s in samples if box_cache.get(str(s["relative"])) is None]
        fresh = detect_person_boxes(detector, needing_detection, device)
        for sample, boxes in zip(needing_detection, fresh):
            box_cache.put(str(sample["relative"]), boxes)

        for sample in samples:
            key = str(sample["relative"])
            width, height = sample["width"], sample["height"]
            person_boxes = box_cache.get(key) or []

            labels: List[Label] = []
            for source_id, box in sample["raw_labels"]:
                out_id = id_table.get(source_id)
                if out_id is None:
                    unknown_ids[source_id] += 1
                    stats["boxes_dropped"] += 1
                    continue
                labels.append((out_id, box))

            windows, unclaimed = plan_windows(
                labels, person_boxes, width, height, args.expand_ratio, args.min_inside_fraction
            )

            emitted: List[Tuple[str, Optional[Tuple[int, int, int, int]], List[Label], int, int]] = []
            for index, window in enumerate(windows):
                x1, y1, x2, y2 = window
                emitted.append((
                    f"{output_stem(dataset_name, sample['relative'])}__p{index}",
                    window,
                    labels_in_window(labels, window, args.min_visible_fraction),
                    x2 - x1,
                    y2 - y1,
                ))
            if unclaimed or not labels:
                emitted.append((
                    output_stem(dataset_name, sample["relative"]),
                    None,
                    labels_in_window(labels, (0, 0, width, height), args.min_visible_fraction),
                    width,
                    height,
                ))

            for stem, window, sample_labels, out_w, out_h in emitted:
                image_name = f"{stem}.jpg"
                destination = images_out / image_name
                source_image = f"{dataset_name}/images/{sample['relative']}"
                owner = claimed_stems.get(stem)
                if owner is not None:
                    # Two sources holding the same image (one of them exporting it
                    # under a name that already carries the other's dataset prefix)
                    # would otherwise silently overwrite each other, leaving the
                    # manifest describing files that are no longer there.
                    stats["stem_collisions"] += 1
                    if stats["stem_collisions"] <= 5:
                        print(
                            f"[build] WARNING: {dataset_name} would overwrite {image_name}, "
                            f"already written from {owner}; skipping this copy"
                        )
                    continue
                claimed_stems[stem] = source_image
                if not link_source.try_link(
                    image_name, destination, source_image, list(window) if window else None
                ):
                    if window is None:
                        # Whole frames are copied byte for byte: re-encoding a JPEG
                        # that is already correct only loses quality.
                        shutil.copy2(images_dir / sample["relative"], destination)
                    else:
                        save_crop(sample["array"], window, destination, args.jpeg_quality)
                lines = yolo_lines(sample_labels, out_w, out_h)
                (labels_out / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
                manifest.write(json.dumps({
                    "image": image_name,
                    "label": f"{stem}.txt",
                    "kind": "crop" if window else "frame",
                    "source_dataset": dataset_name,
                    "source_image": source_image,
                    "window": list(window) if window else None,
                    "source_frame_size": [width, height],
                    "boxes": len(lines),
                }) + "\n")
                stats["crops" if window else "frames"] += 1
                stats["boxes_written"] += len(lines)

                if previews_written < preview_budget and sample_labels:
                    save_preview(
                        destination, sample_labels, merger.out_classes,
                        preview_dir / image_name,
                    )
                    previews_written += 1

        if batch_index % 50 == 0:
            box_cache.flush()
            print(
                f"[build] {dataset_name}: {stats['source_images']} image(s) -> "
                f"{stats['crops']} crop(s) + {stats['frames']} frame(s)"
            )

    box_cache.flush()
    if unknown_ids:
        print(
            f"[build] {dataset_name}: dropped {stats['boxes_dropped']} box(es) whose class "
            f"did not map: {dict(unknown_ids)} (source classes {source_classes})"
        )
    print(
        f"[build] {dataset_name}: done — {stats['crops']} crop(s) + {stats['frames']} frame(s) "
        f"from {stats['source_images']} image(s), {stats['boxes_written']} box(es)"
    )
    return {
        "name": dataset_name,
        "source_classes": source_classes,
        "unmapped_class_ids": {str(key): value for key, value in unknown_ids.items()},
        "previews_written": previews_written,
        **{key: stats[key] for key in
           ("source_images", "crops", "frames", "boxes_written", "boxes_dropped", "unreadable", "stem_collisions")},
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source-root", default="/app/data/source", help="Directory holding the source datasets.")
    parser.add_argument("--out-root", default=None, help="Where to write the new dataset (default: --source-root).")
    parser.add_argument("--out-name", required=True, help="Name of the dataset directory to create.")
    parser.add_argument("--datasets", nargs="+", required=True, help="Source dataset directory names to merge.")
    parser.add_argument("--classes", nargs="+", required=True, help="Output classes, in id order.")
    parser.add_argument(
        "--alias", action="append", default=[], metavar="SOURCE=OUTPUT",
        help="Map a source class name onto an output class (repeatable), e.g. gun=handgun.",
    )
    parser.add_argument("--detector", default="/app/models/people/best_ckpt.engine", help="Person detector checkpoint/engine.")
    parser.add_argument("--person-class-id", type=int, default=0, help="Detector class id that means 'person'.")
    parser.add_argument("--person-score-threshold", type=float, default=0.5, help="Minimum person-detection confidence.")
    parser.add_argument("--detector-batch-size", type=int, default=8, help="Frames per detector forward pass.")
    parser.add_argument(
        "--expand-ratio", type=float, default=0.3,
        help="Pad each person box outward by this fraction of its side before cropping.",
    )
    parser.add_argument(
        "--min-inside-fraction", type=float, default=0.3,
        help="Fraction of a weapon box's area that must fall in a person box for that person to claim it.",
    )
    parser.add_argument(
        "--min-visible-fraction", type=float, default=0.1,
        help="Fraction of a box that must survive clipping for it to stay in a crop's labels.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=95, help="Quality for encoded crops (whole frames are copied).")
    parser.add_argument("--batch-size", type=int, default=8, help="Images per loader batch.")
    parser.add_argument("--num-workers", type=int, default=8, help="Loader workers decoding source images.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--person-box-cache", default=None, help="Directory for reusable per-dataset person-box JSON.")
    parser.add_argument(
        "--link-images-from", default=None,
        help="An already-built dataset dir whose matching images are hardlinked instead of re-encoded.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N images of each source (smoke test).")
    parser.add_argument("--preview", type=int, default=0, help="Write this many annotated samples for eyeballing.")
    parser.add_argument(
        "--preview-dir", default=None,
        help="Where --preview samples go (default: <out>_preview). Point it outside the source "
             "root, which is itself scanned for datasets.",
    )
    return parser.parse_args(argv)


def main() -> None:
    report = build_dataset(parse_args())
    totals = report["totals"]
    print(
        f"[build] {report['output']}: {totals['crops']} crop(s) + {totals['frames']} frame(s) "
        f"= {totals['crops'] + totals['frames']} image(s), {totals['boxes_written']} box(es), "
        f"{totals['boxes_dropped']} dropped, in {report['elapsed_seconds']}s"
    )


if __name__ == "__main__":
    main()
