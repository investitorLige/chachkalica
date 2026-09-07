"""The pipeline-config document: shared shape for ``pipeline.json`` and a
per-camera override file.

A bundle's ``pipeline.json`` and a ``per-camera-overrides/<name>.json`` file are
the same document — which chachak pipeline to run, and the parameters that go
with it — just at different levels of completeness. ``pipeline.json`` is always
complete (every field the bundle was exported with); an override file is sparse,
carrying only the fields it changes. This module validates and builds both, and
keeps that distinction correct: an *absent* key means "inherit from the bundle",
while an *explicit* ``null`` on a nullable field (``tile_size_px``,
``person_class_id``) is a real value, not an absence.
"""

from __future__ import annotations

from typing import Any, Optional

from vendor.chachak.config import (
    ALL_PIPELINES,
    PIPELINE_NAMES,
    RAW,
    DetectorConfig,
    InferenceConfig,
    TilingConfig,
)

SCHEMA_VERSION = 1

VALID_PIPELINES = ALL_PIPELINES

_TILING_KEYS = ("tile_size_px", "tile_width_pct", "tile_height_pct", "overlap", "nms_iou")
_DETECTOR_KEYS = (
    "score_threshold", "person_class_name", "person_class_id",
    "expand_ratio", "nms_iou", "min_box_size", "batch_size",
)
_TOP_LEVEL_KEYS = (
    "pipeline", "chain", "score_threshold", "infer_batch_size", "merge_nms_iou", "notes",
)


class PipelineConfigError(ValueError):
    """A pipeline-config document exists but cannot be trusted."""


def parse(raw: dict[str, Any], source: str = "document", *, require_pipeline: bool = False) -> dict[str, Any]:
    """Validate and normalize a pipeline-config mapping.

    ``schema_version`` is guarded the way ``ModelMeta.from_dict`` guards Contract
    B's: an older backend refuses a newer artifact loudly rather than silently
    mis-reading fields it doesn't know about.

    ``require_pipeline`` is set for a bundle's own ``pipeline.json`` (which must
    always declare how it runs) and left off for an override file (which may only
    touch, say, ``tiling.overlap`` and leave everything else to the bundle).

    Keys absent from ``raw`` stay absent from the result rather than being filled
    with defaults. That distinction is what makes the layering in
    ``Camera.build_inference_config`` work — "the bundle says 0.2" and "the bundle
    says nothing, so chachak's 0.2 applies" have to stay distinguishable, or an
    override could never leave a knob alone.
    """
    version = raw.get("schema_version")
    if version != SCHEMA_VERSION:
        raise PipelineConfigError(
            f"{source}: schema_version {version!r} != supported {SCHEMA_VERSION}. "
            "Re-export the bundle, or upgrade the backend."
        )

    pipeline = raw.get("pipeline")
    if require_pipeline and not pipeline:
        raise PipelineConfigError(f"{source}: must declare a top-level 'pipeline' field.")
    if pipeline is not None and pipeline not in VALID_PIPELINES:
        raise PipelineConfigError(
            f"{source}: pipeline {pipeline!r} is not one of {', '.join(VALID_PIPELINES)}."
        )

    parsed: dict[str, Any] = {
        key: raw[key] for key in _TOP_LEVEL_KEYS if key in raw and raw[key] is not None
    }
    # `chain` is the exception: an explicit empty list is meaningful ("this
    # pipeline is not a chain"), so it survives the None filter above.
    if isinstance(raw.get("chain"), list):
        parsed["chain"] = [str(item) for item in raw["chain"]]

    if pipeline == "chain" and not parsed.get("chain"):
        raise PipelineConfigError(f"{source}: pipeline 'chain' requires a non-empty 'chain' list.")
    for child in parsed.get("chain", []):
        if child not in PIPELINE_NAMES or child == "chain":
            raise PipelineConfigError(f"{source}: invalid chain member {child!r}.")

    for group, keys in (("tiling", _TILING_KEYS), ("detector", _DETECTOR_KEYS)):
        block = raw.get(group)
        if block is None:
            continue
        if not isinstance(block, dict):
            raise PipelineConfigError(f"{source}: '{group}' must be a JSON object.")
        # tile_size_px and person_class_id are legitimately null ("no fixed tile
        # size", "resolve the person class by name"), so they are kept even when
        # None — unlike the top-level keys above.
        nullable = {"tile_size_px", "person_class_id"}
        kept = {k: block[k] for k in keys if k in block and (block[k] is not None or k in nullable)}
        if kept:
            parsed[group] = kept

    return parsed


def build(
    pipeline: Optional[str] = None,
    *,
    chain: Optional[list[str]] = None,
    score_threshold: Optional[float] = None,
    infer_batch_size: Optional[int] = None,
    merge_nms_iou: Optional[float] = None,
    tiling: Optional[dict[str, Any]] = None,
    detector: Optional[dict[str, Any]] = None,
    notes: str = "",
) -> dict[str, Any]:
    """Assemble a pipeline-config document, ready to serialize.

    ``pipeline`` is optional here — a pure override may only touch, e.g.,
    ``tiling.overlap`` and leave the pipeline choice to the bundle. Only what the
    caller actually specified is written — see :func:`parse` on why an absent key
    must not become a default.
    """
    if pipeline is not None and pipeline not in VALID_PIPELINES:
        raise PipelineConfigError(
            f"pipeline {pipeline!r} is not one of {', '.join(VALID_PIPELINES)}."
        )

    document: dict[str, Any] = {"schema_version": SCHEMA_VERSION}
    if pipeline is not None:
        document["pipeline"] = pipeline
    if chain is not None:
        document["chain"] = list(chain)
    if score_threshold is not None:
        document["score_threshold"] = float(score_threshold)
    if infer_batch_size is not None:
        document["infer_batch_size"] = int(infer_batch_size)
    if merge_nms_iou is not None:
        document["merge_nms_iou"] = float(merge_nms_iou)
    if tiling:
        document["tiling"] = {k: v for k, v in tiling.items() if k in _TILING_KEYS}
    if detector:
        document["detector"] = {k: v for k, v in detector.items() if k in _DETECTOR_KEYS}
    if notes:
        document["notes"] = notes

    # Round-trip through parse() so a document this function builds can never be
    # one that a stricter reader would later reject.
    parse(document, source="generated document")
    return document


def merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Layer a sparse override document onto a (normally complete) base one.

    Top-level keys are replaced outright; ``tiling``/``detector`` are merged
    field-by-field, since those are the sub-documents an override typically
    touches only part of. Both ``base`` and ``override`` are expected to already
    be :func:`parse`-shaped (only explicitly-set keys present).
    """
    merged = dict(base)
    for key, value in override.items():
        if key in ("tiling", "detector"):
            merged[key] = {**(base.get(key) or {}), **value}
        else:
            merged[key] = value
    return merged


def to_inference_config(doc: dict[str, Any]) -> InferenceConfig:
    """Build a real :class:`InferenceConfig` from a parsed (or merged) document.

    Any field the document doesn't carry falls back to chachak's own dataclass
    default — the same "absent means chachak's default" rule :func:`parse` is
    built around.
    """
    return InferenceConfig(
        pipeline=doc.get("pipeline") or RAW,
        chain=list(doc.get("chain") or []),
        infer_batch_size=int(doc.get("infer_batch_size", 4)),
        score_threshold=float(doc.get("score_threshold", 0.5)),
        merge_nms_iou=float(doc.get("merge_nms_iou", 0.5)),
        tiling=TilingConfig(**(doc.get("tiling") or {})),
        detector=DetectorConfig(**(doc.get("detector") or {})),
    )
