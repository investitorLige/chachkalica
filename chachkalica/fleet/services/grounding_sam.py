"""Auto-label a dataset from its classes.txt using the Grounding SAM backend.

Talks to the standalone ML backend at chachkalica/ml_backends/sam over HTTP
(it keeps its own torch/lang-sam deps, so it can't be imported in-process),
mirroring training/services/runner.py's client style. Detected regions are
reduced to their bounding-box extent and written as bbox-only YOLO .txt files
into the dataset's source labels/ folder, alongside its existing annotations.
"""

import os
from pathlib import Path
from urllib.parse import quote

import cv2
import requests
from django.conf import settings

from fleet.models import Dataset, FleetSettings, GroundingSamRun
from fleet.reconcile import writer
from fleet.reconcile.labels import name_to_index
from fleet.reconcile.txt_format import objects_to_text
from fleet.services import lsapi
from fleet.services import datasets as datasets_svc
from fleet.services.paths import source_root

# Generous: the backend lazy-loads Lang-SAM (GroundingDINO + SAM) on its first
# call ever, which alone can take longer than a normal batch — a short timeout
# here would abort a job that's actually still working server-side.
PREDICT_TIMEOUT = 1800
BATCH_SIZE = 16


def base_url() -> str:
    return os.getenv("SAM_BACKEND_URL", "http://sam-backend:9090").rstrip("/")


def _data_root() -> Path:
    """The directory sam-backend sees as SAM_DATA_ROOT — same host mount as
    web/worker (see docker-compose.yml's SHARED FILESYSTEM note)."""
    return Path(settings.BASE_DIR) / "data"


def predict(tasks: list[dict], *, prompt: str, confidence: float) -> list[dict]:
    payload = {
        "tasks": tasks,
        "params": {"grounding_sam": prompt, "grounding_sam_confidence": confidence},
    }
    resp = requests.post(f"{base_url()}/predict", json=payload, timeout=PREDICT_TIMEOUT)
    if resp.status_code >= 400:
        raise RuntimeError(f"sam-backend /predict returned HTTP {resp.status_code}: {resp.text}")
    return resp.json()["results"]


def _has_existing_label(labels_dir: Path, image_path: Path) -> bool:
    return any(
        (labels_dir / candidate).exists()
        for candidate in (f"{image_path.name}.txt", f"{image_path.stem}.txt")
    )


def _task_for_image(image_path: Path, *, task_id: int, data_root: Path) -> dict:
    rel_path = lsapi.get_relative_path(image_path, data_root)
    return {
        "id": task_id,
        "data": {"image": f"/data/local-files/?d={quote(rel_path)}"},
        "meta": {"local_path": rel_path},
    }


def _image_size(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    height, width = image.shape[:2]
    return width, height


def _bbox_from_points(points: list[list[float]]) -> tuple[float, float, float, float]:
    """Points are LS-format percentages (0-100); return a fractional (cx, cy, w, h)."""
    xs = [p[0] / 100.0 for p in points]
    ys = [p[1] / 100.0 for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    return (min_x + max_x) / 2, (min_y + max_y) / 2, max_x - min_x, max_y - min_y


def _objects_from_prediction(prediction: dict, class_to_index: dict[str, int]) -> list[dict]:
    objects = []
    for item in prediction.get("result", []):
        if item.get("type") != "polygonlabels":
            continue
        value = item.get("value") or {}
        labels = value.get("polygonlabels") or []
        points = value.get("points") or []
        if not labels or labels[0] not in class_to_index or len(points) < 3:
            continue
        cx, cy, w, h = _bbox_from_points(points)
        objects.append({"class_id": class_to_index[labels[0]], "bbox": (cx, cy, w, h), "polygon": None})
    return objects


def generate_labels_for_dataset(
    dataset: Dataset, *, confidence: float, run: GroundingSamRun | None = None
) -> dict:
    """Populate a dataset's source labels/ folder from Grounding SAM.

    Images that already have a label file are left untouched. Only bounding
    boxes are written (SAM's polygon/mask result is reduced to its extent) —
    the training pipeline consumes bboxes only, so there's nothing to gain
    from carrying the full segmentation shape.

    When ``run`` is given (a GroundingSamRun row), its progress counters are
    updated after each batch so the admin list can be refreshed to watch the
    job move along.
    """
    fs = FleetSettings.load()
    dataset_dir = source_root(fs) / dataset.name
    names, _tools = lsapi.parse_classes_file(dataset_dir / "classes.txt")
    class_to_index = name_to_index(names)

    images = lsapi.list_dataset_images(lsapi.image_source_dir(dataset_dir))
    dest_dir = datasets_svc.labels_source_dir(dataset, fs)
    dest_dir.mkdir(parents=True, exist_ok=True)

    pending = [img for img in images if not _has_existing_label(dest_dir, img)]
    skipped = len(images) - len(pending)

    if run is not None:
        run.images_total = len(pending)
        run.save(update_fields=["images_total"])

    data_root = _data_root()
    prompt = ", ".join(names)
    images_labeled = 0
    detections_written = 0

    for start in range(0, len(pending), BATCH_SIZE):
        batch = pending[start : start + BATCH_SIZE]
        tasks = [
            _task_for_image(img, task_id=index, data_root=data_root)
            for index, img in enumerate(batch, start=start)
        ]
        results = predict(tasks, prompt=prompt, confidence=confidence)
        for image_path, prediction in zip(batch, results):
            objects = _objects_from_prediction(prediction, class_to_index)
            if not objects:
                continue
            width, height = _image_size(image_path)
            writer.write_atomic(
                dest_dir / writer.label_filename(image_path.name),
                objects_to_text(width, height, objects),
            )
            images_labeled += 1
            detections_written += len(objects)

        if run is not None:
            run.images_processed = start + len(batch)
            run.images_labeled = images_labeled
            run.detections_written = detections_written
            run.save(update_fields=["images_processed", "images_labeled", "detections_written"])

    datasets_svc.detect_labels(dataset)
    return {
        "dataset": dataset.name,
        "images_processed": len(pending),
        "images_skipped_labeled": skipped,
        "images_labeled": images_labeled,
        "detections_written": detections_written,
    }
