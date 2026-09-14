"""Per-ground-truth-box measurements: how big the person is, and what they are doing.

Frame-level measures (brightness, contrast) are a decode and a histogram, so the
Django app computes those itself. These are different: they need a person
detector and the RTMO posture engine, so they need torch, TensorRT and a GPU —
and, more decisively, they need to number ground-truth boxes *exactly* the way
the match table does.

That numbering is why this lives here rather than app-side. ``gt.row`` in a match
table is the index into :func:`friendy_chachkalica.data._read_yolo_label_file`'s
**accepted** boxes, and that function skips short lines (which is also how it
skips the app's own ``W H`` header), odd-length polygons and degenerate boxes.
Re-implementing those rules on the other side of the repo would drift silently,
per image, on exactly the files that contain a bad line — and attribute a pose to
the box *next to* the one it was measured on. Here, it is the same function.

Driven by a request YAML so the trainer service can spawn it, mirroring
``eval_checkpoint.py``:

    .venv/bin/python ml/build_measures.py request.yaml
"""

import argparse
import json
import os
import signal
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageOps

try:
    from ..data import IMAGE_EXTENSIONS, _image_to_label_path, _read_yolo_label_file
    from ..device import resolve_device
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from data import IMAGE_EXTENSIONS, _image_to_label_path, _read_yolo_label_file
    from device import resolve_device

VERSION = 1

#: Share of a ground-truth box that must fall inside a person box for that
#: person to own it. Containment, not IoU: a helmet is a couple of percent of
#: its wearer's area, so its IoU with the *correct* person is ~0.02 and an
#: argmax over IoU is noise. Containment is ~1.0 for the wearer.
CONTAINMENT_MIN = 0.7

#: A ground-truth box that *is* a person is a different problem: the two boxes
#: describe the same object, so IoU is right and containment would let a large
#: foreground figure swallow someone standing behind them.
PERSON_IOU_MIN = 0.5

#: Names treated as "this box is itself a person", matched case-insensitively --
#: the same rule ``chachak.detector.load_detector`` uses to find its class.
PERSON_CLASS_NAMES = {"person", "people", "pedestrian"}

#: Emitted when a box falls inside no detected person. Distinct from "not
#: measured": the pass ran and found nobody, which is a different fact about the
#: image and often the interesting one.
NO_PERSON = "(no person)"


def _ensure_chachak_importable() -> None:
    """Put the repo root on sys.path so ``import chachak`` resolves in-process."""
    root = str(Path(__file__).resolve().parent.parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)


def _images(images_dir: Path) -> list[Path]:
    """Every frame, in the loader's own order (sorted, recursive)."""
    return sorted(
        path for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _load_frame(path: Path) -> tuple[torch.Tensor, int, int]:
    """An RGB CHW float tensor in [0, 1], plus the frame's size.

    Opened exactly the way ``data.YoloDetectionDataset`` opens it, EXIF
    transpose included: a rotated JPEG reports transposed dimensions otherwise,
    and every ratio computed from it would be wrong for those images alone.

    RGB, not BGR. The posture engine's meta says ``layout: "bgr"`` and its
    preprocessor does that flip itself (``onnx_infer/preprocess.py`` /
    ``trt_infer/preprocess_torch.py``), so handing it BGR double-flips. Measured:
    on the person_val_v2 frames, feeding RGB finds 7-8 people where pre-flipped
    BGR finds 4, at higher confidence -- and it never errors, it just quietly
    detects worse. (The standalone manual under ``pose_estimation_manual/`` says
    otherwise; the code is what is authoritative here.)
    """
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    width, height = image.size
    tensor = torch.from_numpy(np.asarray(image)).permute(2, 0, 1).float() / 255.0
    return tensor, width, height


def _xyxy(rows: torch.Tensor, width: int, height: int) -> np.ndarray:
    """Friendy ``(N, 6)`` normalized centre-xywh rows to absolute xyxy pixels."""
    if rows is None or rows.numel() == 0:
        return np.zeros((0, 4), dtype=np.float64)
    boxes = rows[:, :4].detach().cpu().numpy().astype(np.float64)
    cx, cy, bw, bh = boxes[:, 0] * width, boxes[:, 1] * height, boxes[:, 2] * width, boxes[:, 3] * height
    return np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)


def _areas(boxes: np.ndarray) -> np.ndarray:
    return np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)


def _intersections(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    if others.size == 0:
        return np.zeros((0,), dtype=np.float64)
    x0 = np.maximum(box[0], others[:, 0])
    y0 = np.maximum(box[1], others[:, 1])
    x1 = np.minimum(box[2], others[:, 2])
    y1 = np.minimum(box[3], others[:, 3])
    return np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)


def attribute(gt_box: np.ndarray, is_person: bool, people: np.ndarray) -> int | None:
    """Which person box owns ``gt_box``, or None.

    Two rules, chosen by what the ground-truth box *is*, because one rule gets
    the other case badly wrong -- see CONTAINMENT_MIN / PERSON_IOU_MIN.

    Ties go to the smallest qualifying person: where two overlapping people both
    contain a helmet, the tighter enclosure is the wearer more often than a large
    figure in the foreground.
    """
    if people.size == 0:
        return None
    intersection = _intersections(gt_box, people)
    person_areas = _areas(people)

    if is_person:
        own_area = float(_areas(gt_box[None, :])[0])
        union = own_area + person_areas - intersection
        score = np.divide(intersection, np.maximum(union, 1e-9))
        qualifying = np.nonzero(score >= PERSON_IOU_MIN)[0]
        if qualifying.size == 0:
            return None
        return int(qualifying[np.argmax(score[qualifying])])

    own_area = float(_areas(gt_box[None, :])[0])
    if own_area <= 0:
        return None
    contained = intersection / own_area
    qualifying = np.nonzero(contained >= CONTAINMENT_MIN)[0]
    if qualifying.size == 0:
        return None
    return int(qualifying[np.argmin(person_areas[qualifying])])


def _match_poses(people: np.ndarray, pose_boxes: np.ndarray, pose_labels: list[str]) -> list:
    """A posture per person box, by best IoU against the posture engine's own boxes.

    None where the engine saw nobody there -- the detector and the posture model
    are two models and are allowed to disagree, and inventing a default posture
    would put that disagreement into the data as though it were an observation.
    """
    poses: list = [None] * len(people)
    if people.size == 0 or pose_boxes.size == 0:
        return poses
    pose_areas = _areas(pose_boxes)
    for index, person in enumerate(people):
        intersection = _intersections(person, pose_boxes)
        union = float(_areas(person[None, :])[0]) + pose_areas - intersection
        iou = np.divide(intersection, np.maximum(union, 1e-9))
        best = int(np.argmax(iou))
        if iou[best] >= PERSON_IOU_MIN:
            poses[index] = pose_labels[best]
    return poses


def build_measures(request: dict, *, progress_path: Path | None = None) -> dict:
    """Measure every ground-truth box and write the sidecar."""
    _ensure_chachak_importable()
    from chachak.infer import load_checkpoint_adapter, predict_adapter

    images_dir = Path(request["images"])
    labels_dir = Path(request["labels"])
    classes = {int(k): str(v) for k, v in enumerate(request["classes"])} \
        if isinstance(request["classes"], list) else \
        {int(k): str(v) for k, v in request["classes"].items()}
    device = resolve_device(request.get("device", "auto"))
    source = dict(request.get("person_source") or {"kind": "rtmo"})
    pose_engine = Path(request["pose"]["engine"])
    pose_threshold = float(request["pose"].get("score_threshold", 0.4))

    print(f"[measures] frames from {images_dir}")
    print(f"[measures] labels from {labels_dir}")
    print(f"[measures] person source: {source.get('kind')}")

    pose_adapter, pose_info = load_checkpoint_adapter(pose_engine, device)
    pose_names = {int(k): str(v) for k, v in (pose_info.get("train_classes") or {}).items()}

    detector = None
    if source.get("kind") == "detector":
        from chachak.detector import load_detector
        detector = load_detector(
            Path(source["checkpoint"]), device,
            score_threshold=float(source.get("score_threshold", 0.5)))

    images = _images(images_dir)
    if not images:
        raise FileNotFoundError(f"No images found under {images_dir}.")

    entries: dict[str, dict] = {}
    totals = {"images": len(images), "images_done": 0, "boxes": 0,
              "attributed": 0, "without_person": 0, "people_found": 0}

    def _flush_progress(complete: bool) -> None:
        if progress_path is None:
            return
        payload = {**totals, "complete": complete}
        tmp = progress_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, progress_path)

    stopping = {"now": False}

    def _stop(_signum, _frame):
        # A partial pass is genuinely useful -- buckets are cut at read time over
        # whatever exists -- but a half-written file is not, so finish the frame
        # in flight and write cleanly.
        stopping["now"] = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    for position, path in enumerate(images, start=1):
        frame, width, height = _load_frame(path)
        frame_area = float(width * height) or 1.0
        label_path = _image_to_label_path(path, images_dir, labels_dir)
        boxes, labels = _read_yolo_label_file(label_path, width, height)
        if not boxes:
            totals["images_done"] = position
            continue

        moved = frame.to(device)
        pose_rows = predict_adapter(pose_adapter, [moved], score_threshold=pose_threshold)[0]
        pose_boxes = _xyxy(pose_rows, width, height)
        pose_labels = [pose_names.get(int(row[5]), str(int(row[5])))
                       for row in pose_rows.detach().cpu()] if pose_rows.numel() else []

        if detector is not None:
            person_rows = detector.predict([moved])[0]
            people = _xyxy(person_rows, width, height)
            poses = _match_poses(people, pose_boxes, pose_labels)
        elif source.get("kind") == "ground_truth":
            keep = [i for i, label in enumerate(labels)
                    if str(classes.get(int(label), "")).lower() in PERSON_CLASS_NAMES]
            people = np.asarray([boxes[i] for i in keep], dtype=np.float64) \
                if keep else np.zeros((0, 4))
            poses = _match_poses(people, pose_boxes, pose_labels)
        else:
            people, poses = pose_boxes, list(pose_labels)

        totals["people_found"] += int(len(people))
        rows = []
        person_areas = _areas(people) if len(people) else np.zeros((0,))
        for row, (box, label) in enumerate(zip(boxes, labels)):
            gt_box = np.asarray(box, dtype=np.float64)
            is_person = str(classes.get(int(label), "")).lower() in PERSON_CLASS_NAMES
            owner = attribute(gt_box, is_person, people)
            values: dict = {}
            if owner is None:
                values["pose"] = NO_PERSON
                totals["without_person"] += 1
            else:
                values["person size ratio"] = round(
                    float(person_areas[owner]) / frame_area, 6)
                values["pose"] = poses[owner] or NO_PERSON
                totals["attributed"] += 1
            rows.append({"row": row, "class_id": int(label), "values": values})
            totals["boxes"] += 1

        entries[path.name] = {"boxes": rows}
        totals["images_done"] = position
        if position % 50 == 0 or position == len(images):
            _flush_progress(False)
            print(f"[measures] {position}/{len(images)} frames", flush=True)
        if stopping["now"]:
            print("[measures] stop requested; writing what has been measured")
            break

    document = {
        "version": VERSION,
        "kind": "box",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "labels_dir": str(labels_dir),
        "images_dir": str(images_dir),
        "person_source": source.get("kind"),
        "person_checkpoint": str(source.get("checkpoint") or ""),
        "pose_engine": str(pose_engine),
        "complete": not stopping["now"],
        "totals": totals,
        "measures": [
            {"name": "pose", "kind": "categorical",
             "choices": [*sorted(set(pose_names.values())), NO_PERSON]},
            {"name": "person size ratio", "kind": "numeric"},
        ],
        "images": entries,
    }
    out = labels_dir / "auto_measures.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=out.parent, delete=False) as handle:
        json.dump(document, handle)
        temporary = Path(handle.name)
    os.replace(temporary, out)
    _flush_progress(True)
    print(f"[measures] wrote {out} ({totals['boxes']} boxes, "
          f"{totals['without_person']} with no person)")
    return {"path": str(out), **totals}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", help="Path to the request YAML")
    args = parser.parse_args()
    request = yaml.safe_load(Path(args.request).read_text(encoding="utf-8"))
    output_dir = Path(request.get("output_dir") or ".")
    output_dir.mkdir(parents=True, exist_ok=True)
    build_measures(request, progress_path=output_dir / "progress.json")


if __name__ == "__main__":
    main()
