"""Assemble a COCO document from a directory of per-image `.txt` files.

This is the compile step run by `fleet.py sync`. It reads every `.txt` under
an annotator's label directory, converts the normalized geometry back to
absolute pixels (using the `W H` header stored in each file), and emits a
standard COCO detection/segmentation dict.
"""

from pathlib import Path

from . import txt_format


def _polygon_area(xs: list[float], ys: list[float]) -> float:
    """Shoelace area for a closed polygon given pixel coordinates."""
    n = len(xs)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += xs[i] * ys[j] - xs[j] * ys[i]
    return abs(area) / 2.0


def _coco_file_name(txt_path: Path) -> str:
    name = txt_path.name
    return name[:-4] if name.endswith(".txt") else name


def build_coco(labels_dir: Path, class_names: list[str]) -> dict:
    categories = [
        {"id": index + 1, "name": name, "supercategory": "object"}
        for index, name in enumerate(class_names)
    ]
    images: list[dict] = []
    annotations: list[dict] = []
    annotation_id = 1

    txt_files = sorted(labels_dir.glob("*.txt")) if Path(labels_dir).exists() else []
    for image_id, txt_path in enumerate(txt_files, start=1):
        width, height, objects = txt_format.parse_text(txt_path.read_text(encoding="utf-8"))
        images.append(
            {
                "id": image_id,
                "file_name": _coco_file_name(txt_path),
                "width": width,
                "height": height,
            }
        )
        for obj in objects:
            cx, cy, w, h = obj["bbox"]
            x = (cx - w / 2) * width
            y = (cy - h / 2) * height
            box_w = w * width
            box_h = h * height
            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": obj["class_id"] + 1,
                "bbox": [round(x, 2), round(y, 2), round(box_w, 2), round(box_h, 2)],
                "iscrowd": 0,
            }
            polygon = obj.get("polygon")
            if polygon:
                xs: list[float] = []
                ys: list[float] = []
                flat: list[float] = []
                for i in range(0, len(polygon), 2):
                    px = polygon[i] * width
                    py = polygon[i + 1] * height
                    xs.append(px)
                    ys.append(py)
                    flat.extend([round(px, 2), round(py, 2)])
                annotation["segmentation"] = [flat]
                annotation["area"] = round(_polygon_area(xs, ys), 2)
            else:
                annotation["segmentation"] = []
                annotation["area"] = round(box_w * box_h, 2)
            annotations.append(annotation)
            annotation_id += 1

    return {
        "info": {"description": "Label Studio annotator-fleet export"},
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
