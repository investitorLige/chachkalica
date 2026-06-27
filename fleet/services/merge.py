"""Merge several source datasets into one brand-new dataset.

A source dataset on disk is ``data/source/<name>/`` = images (flat or under
``images/``) + a ``classes.txt``, and *optionally* a ``labels/`` folder of YOLO
``.txt`` files (the "labeled" datasets — see ``datasets.detect_labels``). Merging
combines the images and writes a single ``classes.txt`` that keeps **only the
class names present in every selected dataset** (matched by name), in one unified
0-based index space.

The merged dataset uses the canonical YOLO layout: images under ``images/`` and,
for any source dataset that *has* labels, its annotations are carried over into
``labels/`` with each class index remapped from that dataset's own order into the
merged order. Annotations whose class did not survive the intersection are
dropped. Unlabeled source datasets contribute images only, so a merge of mixed
inputs is partially labeled. Nothing under ``target/`` is read or written here.
"""

import re
import shutil
from pathlib import Path

from fleet.models import Dataset
from fleet.reconcile import writer
from fleet.reconcile.txt_format import _is_int_token
from fleet.services import datasets as datasets_svc
from fleet.services import lsapi
from fleet.services.paths import source_root

# A bare directory name: letters/digits/_/-/. with no path separators or `..`.
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _dataset_classes(src: Path, dataset: Dataset) -> tuple[list[str], list[str]]:
    """Return (class names, tool keywords) for a dataset's classes.txt."""
    classes_file = src / dataset.name / "classes.txt"
    lsapi.require_path(classes_file, kind="Classes file")
    return lsapi.parse_classes_file(classes_file)


def _tool_identity(tool: str) -> str:
    """Return the Label Studio control identity for a tool keyword."""
    spec = lsapi.LABEL_TOOLS.get(tool)
    return spec[1] if spec else tool


def compute_intersection(datasets: list[Dataset]) -> tuple[list[str], list[str], list[str]]:
    """Compute (kept_names, dropped_names, tools) across datasets.

    ``kept`` = names present in *every* dataset, ordered by their position in the
    first dataset. ``dropped`` = every other name seen (for the preview), in
    first-appearance order. ``tools`` = tool keywords present in every dataset,
    ordered by their position in the first dataset.
    """
    src = source_root()
    per_dataset_names: list[list[str]] = []
    per_dataset_tools: list[list[str]] = []
    for dataset in datasets:
        names, ds_tools = _dataset_classes(src, dataset)
        per_dataset_names.append(names)
        per_dataset_tools.append(ds_tools)

    if not per_dataset_names:
        return [], [], []

    common = set(per_dataset_names[0])
    for names in per_dataset_names[1:]:
        common &= set(names)

    kept = [name for name in per_dataset_names[0] if name in common]

    common_tool_ids = {_tool_identity(tool) for tool in per_dataset_tools[0]}
    for ds_tools in per_dataset_tools[1:]:
        common_tool_ids &= {_tool_identity(tool) for tool in ds_tools}

    seen_tool_ids: set[str] = set()
    tools = []
    for tool in per_dataset_tools[0]:
        tool_id = _tool_identity(tool)
        if tool_id in common_tool_ids and tool_id not in seen_tool_ids:
            tools.append(tool)
            seen_tool_ids.add(tool_id)

    dropped: list[str] = []
    for names in per_dataset_names:
        for name in names:
            if name not in common and name not in dropped:
                dropped.append(name)

    return kept, dropped, tools


def merge_datasets(datasets: list[Dataset], new_name: str) -> dict:
    """Build a new merged dataset on disk + DB row. Returns a summary dict."""
    new_name = (new_name or "").strip()
    if len(datasets) < 2:
        raise RuntimeError("Select at least two datasets to merge.")
    if not _NAME_RE.match(new_name):
        raise RuntimeError(
            f"Invalid dataset name {new_name!r}: use letters, digits, '.', '_', '-' "
            "and no path separators."
        )
    if Dataset.objects.filter(name=new_name).exists():
        raise RuntimeError(f"A dataset named {new_name!r} already exists.")

    cloud = [d.name for d in datasets if d.storage_type != Dataset.LOCAL]
    if cloud:
        raise RuntimeError(
            "Merge supports local-storage datasets only; these are cloud: "
            + ", ".join(cloud)
        )

    src = source_root()
    new_dir = src / new_name
    if new_dir.exists():
        raise RuntimeError(f"Source directory already exists: {new_dir}")

    kept, dropped, tools = compute_intersection(datasets)
    if not kept:
        raise RuntimeError("The selected datasets share no common class names.")
    if not tools:
        raise RuntimeError("The selected datasets share no common labeling tools.")

    new_dir.mkdir(parents=True)
    try:
        images_dir = new_dir / "images"
        labels_dir = new_dir / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()

        kept_index = {name: i for i, name in enumerate(kept)}

        images = 0
        labels = 0
        for dataset in datasets:
            # Map this dataset's own class indices onto the merged order; classes
            # that didn't survive the intersection are simply absent from `remap`.
            names, _ = _dataset_classes(src, dataset)
            remap = {
                old: kept_index[name]
                for old, name in enumerate(names)
                if name in kept_index
            }
            labeled = datasets_svc.detect_labels(dataset, persist=False)
            labels_src = datasets_svc.labels_source_dir(dataset) if labeled else None

            image_dir = lsapi.image_source_dir(src / dataset.name)
            for img in sorted(image_dir.iterdir()):
                if img.suffix.lower() not in lsapi.IMAGE_EXTENSIONS:
                    continue
                merged_name = f"{dataset.name}__{img.name}"
                shutil.copy2(img, images_dir / merged_name)
                images += 1

                if labels_src is None:
                    continue
                label_file = _find_label_file(labels_src, img.name)
                if label_file is None:
                    continue
                remapped = _remap_label_text(label_file.read_text(encoding="utf-8"), remap)
                if remapped:
                    writer.write_atomic(labels_dir / f"{Path(merged_name).stem}.txt", remapped)
                    labels += 1

        header = f"# tools: {', '.join(tools)}\n" if tools else ""
        writer.write_atomic(new_dir / "classes.txt", header + "\n".join(kept) + "\n")

        dataset_row = Dataset.objects.create(name=new_name, storage_type=Dataset.LOCAL)
        # Flip has_labels from the labels we just wrote (False when none carried over).
        datasets_svc.detect_labels(dataset_row)
    except BaseException:
        shutil.rmtree(new_dir, ignore_errors=True)
        raise

    return {
        "new_name": new_name,
        "dataset_id": dataset_row.id,
        "images": images,
        "labels": labels,
        "kept": kept,
        "dropped": dropped,
        "sources": [d.name for d in datasets],
    }


def _find_label_file(labels_dir: Path, image_filename: str) -> Path | None:
    """Locate an image's YOLO label file, mirroring ``load_predictions_for_image``.

    Tries ``<image_filename>.txt`` (the app's own convention, e.g. ``img.jpg.txt``)
    then ``<stem>.txt`` (standard YOLO, e.g. ``img.txt``). Returns None if neither.
    """
    for candidate in (
        labels_dir / f"{image_filename}.txt",
        labels_dir / f"{Path(image_filename).stem}.txt",
    ):
        if candidate.exists():
            return candidate
    return None


def _remap_label_text(text: str, remap: dict[int, int]) -> str:
    """Rewrite a YOLO label file's class indices into the merged index space.

    Only the leading class-index token of each annotation line is changed; the
    coordinate/polygon tokens pass through verbatim. Lines whose class is not in
    ``remap`` (dropped from the intersection, or an out-of-range index) are
    removed. An optional app-format ``width height`` header (two integer tokens on
    the first non-empty line) is preserved. Returns ``""`` when no annotation
    lines survive, so the caller writes no file for that image.
    """
    out: list[str] = []
    header_checked = False
    kept_any = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        tokens = line.split()
        if not header_checked:
            header_checked = True
            if len(tokens) == 2 and all(_is_int_token(tok) for tok in tokens):
                out.append(line)  # width height header — keep as-is
                continue
        if len(tokens) < 5:
            continue
        old = int(float(tokens[0]))
        if old not in remap:
            continue
        tokens[0] = str(remap[old])
        out.append(" ".join(tokens))
        kept_any = True

    if not kept_any:
        return ""
    return "\n".join(out) + "\n"
