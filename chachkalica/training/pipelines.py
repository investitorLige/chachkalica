"""Canonical chachak pipeline vocabulary for the Django app.

Both :class:`training.models.Experiment` (the train/eval pipeline attached to an
experiment) and :class:`eval_pipelines.models.PipelineEvalRun` (a standalone
pipeline eval) reference these constants, so the choices live in exactly one
place.

Keep in sync with ``chachak.config.PIPELINE_NAMES`` — we can't import chachak
here (its package pulls in torch, absent in the app env), so this list is a
hand-maintained mirror.
"""

BATCH_DETECT = "batch_detect"
PEOPLE_DETECT_FIRST = "people_detect_first"
BATCH_PEOPLE = "batch_people"
CHAIN = "chain"

PIPELINE_CHOICES = [
    (BATCH_DETECT, "batch_detect — tile, detect per tile, merge"),
    (PEOPLE_DETECT_FIRST, "people_detect_first — detect people, crop, detect per crop"),
    (BATCH_PEOPLE, "batch_people — tile, detect people, crop, detect per crop"),
    (CHAIN, "chain — run several pipelines and merge"),
]

# Pipelines that require a person-detector checkpoint.
DETECTOR_PIPELINES = {PEOPLE_DETECT_FIRST, BATCH_PEOPLE}


def needs_detector(pipeline: str, chain=None) -> bool:
    """Does ``pipeline`` (with its ``chain``, for :data:`CHAIN`) run a person detector?

    The one definition. This predicate decides whether a detector checkpoint is
    required, whether the crop knobs mean anything, and whether an export must
    carry a detector alongside the model — and it was previously re-spelled at
    each of those call sites, which is how a caller ends up asking "is the field
    non-blank" instead (and leaking a detector block into a tiling pipeline, or
    dropping one a chain member needs).
    """
    if pipeline in DETECTOR_PIPELINES:
        return True
    return pipeline == CHAIN and any(c in DETECTOR_PIPELINES for c in (chain or []))

# Pipelines that support being trained *through*: the training loop applies the
# same frame transform (tiling, or person-cropping via the detector) it uses at
# val/test, so the model trains on the exact sub-frames it is served on. Keep in
# sync with friendy_chachkalica.ml.train._TRAINABLE_PIPELINES. Only ``chain`` is
# excluded — merging several pipelines has no single train-time transform.
TRAINABLE_PIPELINES = {BATCH_DETECT, PEOPLE_DETECT_FIRST, BATCH_PEOPLE}

# Pipelines offered in the Experiment UI: only those supported end-to-end
# (train + val + test) today. The rest are hidden until train-loop support
# lands, so an operator can't pick a pipeline that would train on full frames.
EXPERIMENT_PIPELINE_CHOICES = [c for c in PIPELINE_CHOICES if c[0] in TRAINABLE_PIPELINES]
