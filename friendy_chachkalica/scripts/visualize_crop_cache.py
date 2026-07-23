#!/usr/bin/env python3
"""Draw a crop cache's boxes back onto their original full-frame images.

Reads a ``build_crop_cache`` output directory (``manifest.jsonl`` + ``images/``
+ ``labels/``, see ``preprocess.crop_cache``), remaps each crop's YOLO labels
back onto its source image via ``remap_crop_boxes_to_frame``, and saves one
annotated copy per distinct source image — a visual sanity check that person
crops and their labels line up with the original frame.

Example:
    python -m friendy_chachkalica.scripts.visualize_crop_cache \\
        /app/data/training/runs/PPE_v0.4-31/_person_crops_cache/train/train_prototype_0 \\
        --classes-file /app/data/source/train_prototype_0/classes.txt \\
        --output-dir /tmp/crop_cache_preview
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

try:
    from ..preprocess.crop_cache import read_crop_label_as_preds, read_manifest, remap_crop_boxes_to_frame
except ImportError:  # run as a flat script
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from preprocess.crop_cache import read_crop_label_as_preds, read_manifest, remap_crop_boxes_to_frame

# Cycled per crop within one source image, so overlapping people stay visually
# distinguishable; not tied to class (class name is drawn as text instead).
_BOX_COLORS = [
    (239, 83, 80), (66, 165, 245), (102, 187, 106), (255, 202, 40),
    (171, 71, 188), (38, 198, 218), (255, 112, 67), (118, 255, 122),
]


def load_classes(classes_file: Optional[Path]) -> Dict[int, str]:
    if classes_file is None:
        return {}
    names = [
        line.strip() for line in Path(classes_file).read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    return {index: name for index, name in enumerate(names)}


def draw_frame(
    source_image_path: Path,
    entries: List[dict],
    cache_dir: Path,
    classes: Dict[int, str],
) -> "Image.Image":
    from PIL import Image, ImageDraw

    image = Image.open(source_image_path).convert("RGB")
    frame_w, frame_h = image.size
    draw = ImageDraw.Draw(image)

    for crop_index, entry in enumerate(entries):
        color = _BOX_COLORS[crop_index % len(_BOX_COLORS)]
        x0, y0 = entry["offset"]
        crop_w, crop_h = entry["crop_size"]
        # The person-crop window itself, for context.
        draw.rectangle([x0, y0, x0 + crop_w, y0 + crop_h], outline=color, width=2)

        crop_boxes = read_crop_label_as_preds(cache_dir / entry["crop_label"])
        frame_boxes = remap_crop_boxes_to_frame(entry, crop_boxes)
        for cx, cy, w, h, _confidence, class_id in frame_boxes.tolist():
            x1, y1 = (cx - w / 2) * frame_w, (cy - h / 2) * frame_h
            x2, y2 = (cx + w / 2) * frame_w, (cy + h / 2) * frame_h
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            label = classes.get(int(class_id), str(int(class_id)))
            draw.text((x1 + 2, max(0, y1 - 12)), label, fill=color)

    return image


def visualize_crop_cache(
    cache_dir: Path,
    output_dir: Path,
    classes: Dict[int, str],
    limit: Optional[int] = None,
) -> int:
    cache_dir = Path(cache_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = read_manifest(cache_dir)
    by_source: Dict[str, List[dict]] = defaultdict(list)
    for entry in entries:
        by_source[entry["source_image_path"]].append(entry)

    source_paths = sorted(by_source)
    if limit is not None:
        source_paths = source_paths[:limit]

    written = 0
    for source_path in source_paths:
        image = draw_frame(Path(source_path), by_source[source_path], cache_dir, classes)
        out_path = output_dir / f"{Path(source_path).stem}.jpg"
        image.save(out_path, quality=95)
        written += 1

    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cache_dir", type=Path, help="A build_crop_cache output directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where to write annotated images")
    parser.add_argument("--classes-file", type=Path, default=None, help="classes.txt to label boxes by name")
    parser.add_argument("--limit", type=int, default=None, help="Only visualize the first N source images")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    classes = load_classes(args.classes_file)
    written = visualize_crop_cache(args.cache_dir, args.output_dir, classes, limit=args.limit)
    print(f"visualize_crop_cache: wrote {written} annotated image(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
