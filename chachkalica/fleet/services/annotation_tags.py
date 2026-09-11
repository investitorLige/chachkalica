"""Read the tag editor's form back into ``AnnotationTag`` rows.

The editor posts a variable number of rows, so the field names carry a row
index (``tag-3-name``) rather than repeating a bare name — a checkbox that is
simply absent when unchecked would otherwise slide every later row's values up
by one. The indices only establish identity within one POST; the stored
``order`` is the row's position in its section, so a deleted row leaves no gap.

Nothing here talks to Label Studio. Turning the saved rows into a labeling
interface is ``datasets.dataset_label_config``; pushing that onto projects that
already exist is ``datasets.apply_tags_to_projects``.
"""

import re

from django.core.exceptions import ValidationError
from django.db import transaction

from fleet.models import AnnotationTag, Dataset

_FIELD = re.compile(r"^tag-(\d+)-name$")

# Accepts "none, low, high" and one-per-line alike — annotators' option lists
# get pasted from both shapes.
_CHOICE_SPLIT = re.compile(r"[,\n\r]+")


def split_choices(raw: str) -> list[str]:
    """Split a typed option list into trimmed, de-duplicated values, in order."""
    out: list[str] = []
    for piece in _CHOICE_SPLIT.split(raw or ""):
        value = piece.strip()
        if value and value not in out:
            out.append(value)
    return out


def posted_rows(post) -> list[dict]:
    """Pull the editor's rows out of a POST, in section then screen order.

    A row whose name is blank is dropped rather than rejected: that is what an
    added-then-abandoned row looks like, and refusing to save the whole form
    over one is worse than ignoring it.
    """
    rows = []
    for key in post:
        match = _FIELD.match(key)
        if not match:
            continue
        index = int(match.group(1))
        name = (post.get(f"tag-{index}-name") or "").strip()
        if not name:
            continue
        try:
            max_rating = int(post.get(f"tag-{index}-max_rating") or 5)
        except ValueError:
            max_rating = 0  # out of range; the model's clean() reports it
        rows.append({
            "index": index,
            "scope": (post.get(f"tag-{index}-scope") or AnnotationTag.FRAME).strip(),
            "name": name,
            "widget": (post.get(f"tag-{index}-widget") or AnnotationTag.RADIO).strip(),
            "choices": split_choices(post.get(f"tag-{index}-choices") or ""),
            "max_rating": max_rating,
            "required": bool(post.get(f"tag-{index}-required")),
        })

    rows.sort(key=lambda row: row["index"])
    counters = {AnnotationTag.FRAME: 0, AnnotationTag.REGION: 0}
    for row in rows:
        scope = row["scope"] if row["scope"] in counters else AnnotationTag.FRAME
        row["order"] = counters[scope]
        counters[scope] += 1
    return rows


def build_tags(dataset: Dataset, rows: list[dict]) -> list[AnnotationTag]:
    """Turn posted rows into validated (unsaved) tags, or raise.

    Every row is validated before any of them is saved, so a typo in the last
    one never leaves the dataset half-edited. Duplicate names are caught here
    rather than by the unique constraint, because within one POST the old rows
    are about to be deleted and the constraint would fire against them instead
    of against the real collision.
    """
    tags: list[AnnotationTag] = []
    errors: list[str] = []
    seen: set[str] = set()

    for row in rows:
        tag = AnnotationTag(
            dataset=dataset,
            scope=row["scope"],
            name=row["name"],
            widget=row["widget"],
            choices=row["choices"],
            max_rating=row["max_rating"],
            required=row["required"],
            order=row["order"],
        )
        try:
            tag.full_clean(exclude=["dataset"], validate_unique=False)
        except ValidationError as exc:
            for messages in exc.message_dict.values():
                errors.extend(f"{row['name']}: {message}" for message in messages)
            continue
        if tag.name in seen:
            errors.append(f"{tag.name}: two tags cannot share a name.")
            continue
        seen.add(tag.name)
        tags.append(tag)

    if errors:
        raise ValidationError(errors)
    return tags


@transaction.atomic
def save_tags(dataset: Dataset, rows: list[dict]) -> list[AnnotationTag]:
    """Replace the dataset's tags with the posted set.

    Replace rather than merge: the editor shows the complete list, so a row the
    operator removed from it is a row they meant to delete.
    """
    tags = build_tags(dataset, rows)
    dataset.tags.all().delete()
    AnnotationTag.objects.bulk_create(tags)
    return tags
