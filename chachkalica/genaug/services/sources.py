"""Where genaug keeps its files, and how it reads a source dataset.

Everything generated lives under ``genaug_root()`` (``data/genaug``), which the
web/worker containers and ``genaug-backend`` all mount at the same path — the
backend is handed paths, never bytes, like every other ML backend here.

* ``cache/<k[:2]>/<k>.png`` — every generated variant, content-addressed by
  what produced it (see :func:`genaug.services.generate.variant_key`). Studio
  previews and builds share it, so a variant previewed in the studio is not
  paid for again when a build picks the same image, prompt and seed.

A finished build's dataset is written under the *source root* instead, beside
every other dataset, so training, the dataset admin and the preview viewer see
it as an ordinary dataset.
"""

from __future__ import annotations

import os
from pathlib import Path

from django.conf import settings

from fleet.services import datasets as datasets_svc, lsapi
from fleet.services.paths import source_root


def genaug_root() -> Path:
    override = os.environ.get("GENAUG_DATA_DIR")
    return Path(override) if override else Path(settings.BASE_DIR) / "data" / "genaug"


def dataset_dir(dataset) -> Path:
    return source_root() / dataset.name


def image_dir(dataset) -> Path:
    return lsapi.image_source_dir(dataset_dir(dataset))


def list_images(dataset) -> list[Path]:
    return lsapi.list_dataset_images(dataset_dir(dataset))


def read_classes(dataset) -> list[str]:
    path = dataset_dir(dataset) / "classes.txt"
    if not path.is_file():
        return []
    classes, _tools = lsapi.parse_classes_file(path)
    return classes


def labels_dir(dataset, label_source, annotator=None, explicit_path="") -> Path:
    # Imported here: training imports fleet, and genaug must not make fleet
    # import training at module load.
    from training.services.config_gen import resolve_label_dir

    return resolve_label_dir(dataset, label_source, annotator, explicit_path)


def label_file(labels: Path, image_name: str) -> Path | None:
    return datasets_svc.find_label_file(labels, image_name)


def read_label_objects(labels: Path, image_name: str) -> list[dict] | None:
    """The image's label objects, or ``None`` when it has no label file.

    ``None`` (unlabeled) and ``[]`` (labeled empty — a real negative) are kept
    apart on purpose: an unlabeled image is never augmented, since there are
    no labels to preserve and a copied "empty" label would teach the model
    that whatever is in it is background.
    """
    from fleet.reconcile import txt_format

    path = label_file(labels, image_name)
    if path is None:
        return None
    _w, _h, objects = txt_format.parse_label_text(path.read_text(encoding="utf-8"))
    return objects


def pixel_boxes(objects: list[dict], width: int, height: int) -> list[tuple[float, float, float, float]]:
    """Normalized centre-xywh (polygons already reduced to their extent) -> pixel xyxy."""
    boxes = []
    for obj in objects:
        cx, cy, w, h = obj["bbox"]
        boxes.append(((cx - w / 2) * width, (cy - h / 2) * height,
                      (cx + w / 2) * width, (cy + h / 2) * height))
    return boxes
