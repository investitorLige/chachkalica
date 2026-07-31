"""The one definition of a model's pipeline metadata.

A model is only meaningful together with the pipeline it was trained through:
serve a person-crop model whole frames, or tile it at a size it never saw, and
the detections are quietly wrong. So every catalogued model carries a frozen
record of that pipeline — :attr:`training.models.TrainedModel.pipeline_metadata`
— and every export carries a copy of the same blob beside the artifact
(``<artifact>.pipeline.json``, written by
:func:`training.services.exports.export_pipeline_sidecar`).

The rule this module exists to enforce: **one schema, every format, every
consumer.** A ``.pt`` checkpoint, an ``.onnx`` graph and a ``.engine`` all
answer the same question with the same keys, so anything that runs a model
(video inference, camera live inference, a chachak bundle) prefills itself from
the metadata instead of asking an operator to retype geometry they already
chose at training time.

Before this module the blob was hand-copied in three places — the video admin,
the camera admin and the export sidecar writer — each carrying the subset of
fields its own form happened to have, so a knob added to one was silently
dropped by the others. :data:`FIELDS` is now the whole vocabulary; consumers may
*use* a subset, but they all *read* the same record.

Resolution order, and why: a frozen blob wins over the source
:class:`~training.models.Experiment` (a promoted model must keep describing the
run it came from even after the experiment is retuned or deleted), and a model
with neither falls back to :func:`raw` — an explicit "full frames", not a hole
in the record.
"""

from __future__ import annotations

from typing import Any, Dict

SCHEMA_VERSION = 1

# `Experiment.pipeline` is blank for plain full-frame training; the serving side
# spells the same thing "raw" (InferenceJob.RAW / CameraInference.RAW), which is
# what the metadata carries — a value the pipeline dropdowns can select.
RAW = "raw"

# Every pipeline knob the metadata carries, with the value meaning "chachak's
# own default". Adding a field here makes it flow into new metadata, new
# sidecars and every consumer's prefill at once, which is the point.
FIELDS: Dict[str, Any] = {
    "pipeline": RAW,
    "detector_checkpoint": "",
    "detector_expand_ratio": None,
    "detector_min_box_size": None,
    "tile_size_px": None,
    "tile_width_pct": None,
    "tile_height_pct": None,
    "overlap": None,
    "merge_nms_iou": None,
    "chain": [],
    # The operating point the model was evaluated at — a starting threshold for
    # serving, not geometry, so overriding it changes nothing about how frames
    # reach the model.
    "score_threshold": None,
}


def raw() -> Dict[str, Any]:
    """Metadata for a model served plain full frames."""
    return normalize({})


def normalize(blob: Any) -> Dict[str, Any]:
    """A complete, JSON-safe metadata dict from whatever ``blob`` holds.

    Tolerates a partial or absent record on purpose: sidecars written before a
    field existed, and blobs from an older ``SCHEMA_VERSION``, are filled out
    with :data:`FIELDS` defaults rather than rejected — a missing key means
    "chachak's default", which is exactly what the older writer meant by not
    writing it. Anything unparseable degrades to :func:`raw` instead of raising,
    since no consumer of this is worth a 500.
    """
    if not isinstance(blob, dict):
        blob = {}

    out: Dict[str, Any] = {"schema_version": SCHEMA_VERSION}
    for key, default in FIELDS.items():
        out[key] = blob.get(key, default)

    out["pipeline"] = (out["pipeline"] or RAW)
    out["detector_checkpoint"] = (out["detector_checkpoint"] or "")
    chain = out["chain"]
    out["chain"] = [str(item) for item in chain] if isinstance(chain, (list, tuple)) else []
    return out


def from_experiment(experiment) -> Dict[str, Any]:
    """Freeze ``experiment``'s train/eval pipeline into a metadata blob.

    Called at promotion time (see :func:`training.services.promote
    .promote_run_result`), so a model records the geometry it was actually
    trained with. A blank ``experiment.pipeline`` yields :func:`raw`.
    """
    if experiment is None or not experiment.pipeline:
        return raw()
    return normalize({
        "pipeline": experiment.pipeline,
        "detector_checkpoint": experiment.detector_checkpoint or "",
        "detector_expand_ratio": experiment.detector_expand_ratio,
        "detector_min_box_size": experiment.detector_min_box_size,
        "tile_size_px": experiment.tile_size_px,
        "tile_width_pct": experiment.tile_width_pct,
        "tile_height_pct": experiment.tile_height_pct,
        "overlap": experiment.overlap,
        "merge_nms_iou": experiment.merge_nms_iou,
        "chain": list(experiment.chain or []),
        "score_threshold": experiment.eval_score_threshold,
    })


def source_experiment(trained_model):
    """The :class:`~training.models.Experiment` ``trained_model`` was trained in,
    or ``None`` — it may have been promoted from an untracked checkpoint, or its
    run may since have been deleted."""
    run = getattr(getattr(trained_model, "source_run_result", None), "run", None)
    return getattr(run, "experiment", None)


def for_trained_model(trained_model) -> Dict[str, Any]:
    """Pipeline metadata for a catalogued model, always a complete blob.

    Prefers the frozen ``pipeline_metadata`` field; falls back to deriving it
    from the source experiment (which covers a model promoted before the field
    existed, should the backfill migration have missed it — e.g. the row was
    created by a fixture), then to :func:`raw`.
    """
    frozen = getattr(trained_model, "pipeline_metadata", None)
    if isinstance(frozen, dict) and frozen:
        return normalize(frozen)
    return from_experiment(source_experiment(trained_model))
