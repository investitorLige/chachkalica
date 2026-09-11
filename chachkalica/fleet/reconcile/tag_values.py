"""Turn Label Studio tag answers into the dataset's ``annotation_tags.json``.

The geometry export has nowhere to put an attribute: a ``.txt`` line is
``class_id cx cy w h`` and any extra number on it is read back as a polygon
coordinate (see :func:`txt_format.parse_text`), so appending tag values to the
label files would silently corrupt every consumer's boxes. Tag answers
therefore travel as one JSON sidecar written *into the label directory*, next
to the ``.txt`` files it describes:

    target/<dataset>/<username>/annotation_tags.json     # written by sync
    source/<dataset>/labels/annotation_tags.json         # after promote

Nothing globs a label directory for anything but ``*.txt``, so the sidecar is
inert to the COCO build, the prune sweep, and the trainer's dataloader.

Shape (``version`` 1)::

    {
      "version": 1,
      "dataset": "ppe", "annotator": "alice",
      "generated_at": "2026-09-11T09:00:00+00:00",
      "tags": [{"name": "weather", "scope": "frame", "widget": "radio",
                "choices": ["sun", "rain"], "max_rating": 5}],
      "images": {
        "img01.jpg": {
          "frame": {"weather": ["rain"]},
          "boxes": [{"row": 0, "class_id": 2, "tags": {"occluded": ["heavy"]}}]
        }
      }
    }

``row`` is the box's line index in that image's ``.txt`` (0-based, after the
``W H`` header) — the only identity a geometry-only format leaves to join on.
``class_id`` is stored beside it purely so a consumer can *detect* a drift
between the two files instead of silently attributing a tag to the wrong box.

Region answers are matched to their box by Label Studio's region ``id``, which
a ``perRegion`` result repeats from the region it annotates. That id never
reaches the ``.txt``, which is why the join has to happen here, in the same
pass that assigns the rows.
"""

from datetime import datetime, timezone
from pathlib import Path

#: The sidecar's filename inside a label directory.
TAGS_FILENAME = "annotation_tags.json"

#: Document format version, bumped when the shape below changes incompatibly.
VERSION = 1

# Scope values, mirroring ``fleet.models.AnnotationTag`` without importing it
# (this package stays ORM-free).
FRAME = "frame"
REGION = "region"

# Result `value` keys per widget family. A Choices control answers under
# "choices", a Rating under "rating", a TextArea under "text" (as a list).
_CHOICE_WIDGETS = frozenset({"checkbox", "radio", "dropdown", "multiselect"})


def tags_path(labels_dir) -> Path:
    """Path of the sidecar for a label directory (may not exist)."""
    return Path(labels_dir) / TAGS_FILENAME


def normalize_value(widget: str, value: dict):
    """Pull one control's answer out of an LS result ``value``, or None.

    Returns the widget's natural type — a list of strings for the four list
    widgets, an int for a rating, a string for free text — rather than
    flattening everything to strings here. Grouping is the analytics layer's
    job and it has one normalizer for it; this file stays a faithful record of
    what the annotator answered.

    None means *unanswered*, which is not the same as an empty answer: an
    optional checkbox tag nobody ticked should not count as its own group.
    """
    if not isinstance(value, dict):
        return None

    if widget in _CHOICE_WIDGETS:
        choices = value.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        return [str(choice) for choice in choices]

    if widget == "rating":
        rating = value.get("rating")
        if rating in (None, 0, ""):
            return None
        try:
            return int(rating)
        except (TypeError, ValueError):
            return None

    if widget == "text":
        text = value.get("text")
        if isinstance(text, list):
            text = "\n".join(str(part) for part in text)
        text = (str(text) if text is not None else "").strip()
        return text or None

    return None


def collect_answers(result: list[dict], specs: list[dict]) -> tuple[dict, dict]:
    """Split one annotation's tag answers into (frame answers, per-region answers).

    ``result`` is a Label Studio annotation ``result`` list — the same one
    :func:`txt_format.result_to_image` reads the geometry out of. Scope comes
    from ``specs`` (the dataset's own tag definitions) rather than from the
    payload: a frame answer and a region answer are the same kind of result
    entry, and only the dataset knows which control was declared ``perRegion``.

    Returns ``({name: value}, {region_id: {name: value}})``.
    """
    widgets = {spec["name"]: spec.get("widget", "radio") for spec in specs}
    scopes = {spec["name"]: spec.get("scope", FRAME) for spec in specs}

    frame: dict = {}
    regions: dict = {}
    for item in result or []:
        name = item.get("from_name")
        if name not in widgets:
            continue  # geometry, the SAM keypoint, or a control we don't own
        value = normalize_value(widgets[name], item.get("value") or {})
        if value is None:
            continue
        if scopes[name] == REGION:
            region_id = item.get("id")
            if region_id is None:
                continue
            regions.setdefault(str(region_id), {})[name] = value
        else:
            frame[name] = value
    return frame, regions


def image_entry(objects: list[dict], frame: dict, regions: dict) -> dict | None:
    """Build one image's entry, or None when it carries no answers at all.

    ``objects`` are the cleaned objects about to be serialized into the image's
    ``.txt``, in that order — so an object's position here *is* its ``row``
    there. Region answers are attached through each object's ``region_id``,
    which :func:`txt_format.result_item_to_object` carries over from the LS
    region and :func:`validate.clean_objects` preserves.

    Boxes with no answers are still listed when any box in the image has one,
    so the rows stay contiguous and a reader can tell "not annotated" from
    "row missing".
    """
    boxes = []
    any_box_answer = False
    for row, obj in enumerate(objects):
        answers = regions.get(str(obj.get("region_id"))) or {}
        if answers:
            any_box_answer = True
        boxes.append({"row": row, "class_id": obj["class_id"], "tags": answers})

    if not frame and not any_box_answer:
        return None
    entry: dict = {}
    if frame:
        entry["frame"] = frame
    if any_box_answer:
        entry["boxes"] = boxes
    return entry


def build_document(*, dataset: str, annotator: str | None, specs: list[dict],
                   images: dict) -> dict:
    """Assemble the sidecar document. ``images`` maps filename -> image_entry."""
    return {
        "version": VERSION,
        "dataset": dataset,
        "annotator": annotator,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tags": [
            {
                "name": spec["name"],
                "scope": spec.get("scope", FRAME),
                "widget": spec.get("widget", "radio"),
                "choices": list(spec.get("choices") or []),
                "max_rating": int(spec.get("max_rating") or 5),
            }
            for spec in specs
        ],
        "images": dict(sorted(images.items())),
    }
