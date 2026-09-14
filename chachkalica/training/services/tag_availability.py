"""What a *planned* eval will be able to be sliced by, before it is queued.

Both evaluate forms ask for a dataset and a label source, and that pair is
exactly what decides which tag answers a later Tag analytics page can find: the
sidecar is looked up next to whichever labels the eval scored against (see
``eval_pipelines.admin._load_tag_analysis``). Until now neither form said a word
about it, so the choice that settles the question was made blind — and the
consequence only showed up much later, on a page that refused to render.

Read-only, cheap and eval-free: there is no match table yet, so nothing here
scores anything. It reads the dataset's declared tags out of the ORM and counts
answers in the sidecar, and it names the computed tags that need no annotation
at all.
"""

from fleet.services import datasets as datasets_svc
from training.services import config_gen, tag_analytics


def _answer_counts(document: dict) -> tuple[dict[str, int], dict[str, int]]:
    """Per tag name, how many images and how many boxes carry an answer.

    One walk of the sidecar rather than one per tag: a dataset with a dozen tags
    and 13k images would otherwise be a dozen passes over the same dict.
    """
    images: dict[str, int] = {}
    boxes: dict[str, int] = {}
    for entry in (document.get("images") or {}).values():
        for name in (entry.get("frame") or {}):
            images[name] = images.get(name, 0) + 1
        for box in (entry.get("boxes") or []):
            for name in (box.get("tags") or {}):
                boxes[name] = boxes.get(name, 0) + 1
    return images, boxes


def availability(dataset, label_source, annotator=None, explicit_labels_path="") -> dict:
    """Describe the slices an eval of ``dataset`` against these labels would have.

    Never raises for an unresolvable label source — that is an ordinary state of
    a half-filled form, and it comes back as ``labels_error`` for the panel to
    print.
    """
    labels_dir = None
    labels_error = ""
    try:
        labels_dir = config_gen.resolve_label_dir(
            dataset, label_source, annotator, explicit_labels_path or "")
    except ValueError as exc:
        labels_error = str(exc)

    document = tag_analytics.load_tag_document(labels_dir) if labels_dir else None
    answered_images, answered_boxes = _answer_counts(document or {})
    synced = {str(spec.get("name")) for spec in ((document or {}).get("tags") or [])}

    declared = []
    for spec in datasets_svc.dataset_tag_specs(dataset):
        name = str(spec.get("name") or "")
        is_frame = str(spec.get("scope")) != "region"
        count = (answered_images if is_frame else answered_boxes).get(name, 0)
        if document is None:
            state, detail = "unknown", "no tag answers have been synced to these labels yet"
        elif name not in synced:
            state, detail = "unsynced", "defined after the last sync, so it has no answers here"
        elif count:
            state = "answered"
            noun = ("image" if count == 1 else "images") if is_frame else (
                "box" if count == 1 else "boxes")
            detail = f"{count} {noun} answered"
        else:
            state, detail = "empty", "synced, but nothing has been answered yet"
        declared.append({
            "name": name,
            "scope": "box-wide" if not is_frame else "frame-wide",
            "widget": spec.get("widget") or "",
            "choices": spec.get("choices") or [],
            "state": state,
            "detail": detail,
        })

    return {
        "dataset_name": dataset.name,
        "labels_dir": str(labels_dir) if labels_dir else "",
        "labels_error": labels_error,
        "sidecar_present": document is not None,
        "declared": declared,
        "auto": [
            {**spec, "scope": "box-wide" if spec["scope"] == "region" else "frame-wide"}
            for spec in tag_analytics.AUTO_TAG_CATALOGUE
        ],
    }
