"""The model + pipeline half of every "run this model" form, in one place.

Three admin surfaces now ask the same two questions before running a model —
*which* model, and *through what geometry* — over videos
(``videos.admin.VideoAdmin._inference_wizard``), over a dataset
(``fleet.admin.DatasetAdmin.run_model_inference``) and, in a bundle-only
variant, in the Marketing Studio. The answer is the same in all of them: the
three model sources (catalogued ``.pt``, exported ``.onnx``/``.engine``,
self-contained bundle) and the pipeline knobs
:data:`training.services.pipeline_meta.FIELDS` names.

This module owns that shared half: it builds the context the
``admin/videos/_inference_fields.html`` partial renders, and parses that
partial's POST back into validated values. Both halves derive their field list
from ``pipeline_meta.FIELDS`` rather than restating it, which is the
standing rule for this vocabulary (see ``docs/pipeline-metadata.md``): a knob
added to the metadata must reach every serving surface's prefill at once, and
the way that broke before was a new call site hand-listing the subset its own
form happened to have.

What a caller still owns is everything *around* the shared half: its own row
(an ``InferenceJob``, a ``DatasetInferenceRun``), the run knobs that are
properties of the run rather than of the model (a video's ``frame_stride``, a
dataset run's ``image_limit``), and the marketing Look section.
"""

from __future__ import annotations

from training import pipelines
from training.services import bundles, exports, pipeline_meta

# ---------------------------------------------------------------- model sources
# The canonical spelling of "where does this model come from". Persisted as a
# column by every row that runs a model (``videos.InferenceJob``,
# ``cameras.CameraInference``, ``fleet.DatasetInferenceRun``), which each declare
# the same three values against their own field — this is the copy the *forms*
# read, so a wizard never spells them itself.
TRAINED = "trained"
EXPORTED = "exported"
BUNDLE = "bundle"
MODEL_SOURCE_CHOICES = [
    (TRAINED, "trained model (.pt checkpoint)"),
    (EXPORTED, "exported artifact (ONNX / TensorRT)"),
    (BUNDLE, "infer bundle (self-contained pipeline directory)"),
]

#: "raw" is not a chachak pipeline — it means "no pipeline, feed the model the
#: whole frame". Same value ``pipeline_meta`` carries for plain full-frame models.
RAW = pipeline_meta.RAW
PIPELINE_CHOICES = [
    (RAW, "raw — whole frame straight to the model"),
    *pipelines.PIPELINE_CHOICES,
]

#: Threshold a model with no recorded operating point is offered. Unlike a blank
#: geometry field (which means "chachak's default"), a blank threshold has no
#: sensible meaning at serving time, so the form always carries a number.
DEFAULT_SCORE_THRESHOLD = 0.5

# Sentinel for the optional parsers: they must tell "left blank" (a valid None,
# meaning chachak's default) apart from "typed nonsense".
_INVALID = object()

#: ``(field, low, high, message)`` for every optional float knob, in the order a
#: form reports problems. Ranges match what the widgets allow.
_OPTIONAL_FLOATS = [
    ("detector_expand_ratio", 0.0, 10.0,
     "Person-box expand ratio must be between 0 and 10."),
    ("tile_width_pct", 0.0001, 100.0, "Tile width % must be in (0, 100]."),
    ("tile_height_pct", 0.0001, 100.0, "Tile height % must be in (0, 100]."),
    ("overlap", 0.0, 0.99, "Tile overlap must be in [0, 1)."),
    ("merge_nms_iou", 0.0, 1.0, "Merge NMS IoU must be between 0 and 1."),
    ("detector_min_box_size", 0.0, 100000.0,
     "Person-crop minimum size must be 0 or more pixels."),
]


# --------------------------------------------------------------------- parsing
def parse_float_in_range(raw, default, low: float, high: float):
    """Parse a float within ``[low, high]``; blank -> ``default``, bad -> ``None``."""
    raw = (raw or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if low <= value <= high else None


def parse_positive_int(raw, default):
    """Parse an int >= 1; blank -> ``default``, bad -> ``None``."""
    raw = (raw or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value >= 1 else None


def parse_optional_float(raw, low: float, high: float):
    """Parse an optional float in ``[low, high]``; blank -> ``None``, bad -> ``_INVALID``."""
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return _INVALID
    return value if low <= value <= high else _INVALID


def parse_optional_int(raw, low: int, high: int):
    """Parse an optional int in ``[low, high]``; blank -> ``None``, bad -> ``_INVALID``."""
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return _INVALID
    return value if low <= value <= high else _INVALID


def parse_pipeline_fields(posted) -> tuple[dict, str | None]:
    """Validate the partial's pipeline half into ``pipeline_meta``-shaped values.

    Returns ``(values, None)`` or ``({}, message)`` on the first problem found,
    so a caller can re-render its form with a warning. ``values`` carries every
    key in :data:`pipeline_meta.FIELDS`, with the knobs the chosen pipeline
    cannot use set back to their "not applicable" value — the form hides
    irrelevant rows with CSS, which still submits them, and a saved row should be
    an honest record of what actually ran rather than of what the page happened
    to be carrying.
    """
    pipeline = posted.get("pipeline") or RAW
    if pipeline not in dict(PIPELINE_CHOICES):
        return {}, f"Unknown pipeline {pipeline!r}."

    score_threshold = parse_float_in_range(
        posted.get("score_threshold"), default=DEFAULT_SCORE_THRESHOLD,
        low=0.0, high=1.0)
    if score_threshold is None:
        return {}, "Score threshold must be between 0 and 1."

    values = {}
    for field, low, high, message in _OPTIONAL_FLOATS:
        value = parse_optional_float(posted.get(field), low, high)
        if value is _INVALID:
            return {}, message
        values[field] = value

    tile_size_px = parse_optional_int(posted.get("tile_size_px"), 1, 100000)
    if tile_size_px is _INVALID:
        return {}, "Tile size must be a positive number of pixels."

    chain = [part.strip() for part in (posted.get("chain") or "").split(",")
             if part.strip()]
    detector_checkpoint = (posted.get("detector_checkpoint") or "").strip()

    # Which knobs this pipeline can actually use. 'chain' keeps all of them: what
    # it can use depends on its members, and it may include any of them.
    uses_detector = pipeline in pipelines.DETECTOR_PIPELINES or pipeline == pipelines.CHAIN
    uses_tiling = pipeline in (
        pipelines.BATCH_DETECT, pipelines.BATCH_PEOPLE, pipelines.CHAIN)
    if not uses_detector:
        detector_checkpoint = ""
        values["detector_expand_ratio"] = None
        values["detector_min_box_size"] = None
    if not uses_tiling:
        tile_size_px = None
        values["tile_width_pct"] = None
        values["tile_height_pct"] = None
        values["overlap"] = None
    # merge_nms_iou applies to every pipeline that makes several predictions per
    # frame to reconcile — that is, all of them except "raw", which runs the model
    # once on the whole frame and has nothing to merge.
    if pipeline == RAW:
        values["merge_nms_iou"] = None
    if pipeline != pipelines.CHAIN:
        chain = []

    return {
        **values,
        "pipeline": pipeline,
        "detector_checkpoint": detector_checkpoint,
        "tile_size_px": tile_size_px,
        "chain": chain,
        "score_threshold": score_threshold,
    }, None


def parse_model_source(posted, model_source: str) -> tuple[dict, str | None]:
    """Validate which model the form picked, for ``model_source``.

    Returns ``({"trained_model", "artifact_path", "bundle_path"}, None)`` — the
    three columns every model-running row carries, exactly one of them filled —
    or ``({}, message)``. A bundle is checked structurally only: the load test
    belongs to the "Sync bundle" button, where the operator asked for it, because
    it takes the trainer's GPU lock.
    """
    from training.models import TrainedModel

    empty = {"trained_model": None, "artifact_path": "", "bundle_path": ""}
    if model_source == EXPORTED:
        artifact_path = (posted.get("artifact_path") or "").strip()
        if not artifact_path:
            return {}, "Choose an exported artifact."
        return {**empty, "artifact_path": artifact_path}, None

    if model_source == BUNDLE:
        bundle_path = (posted.get("bundle_path") or "").strip()
        if not bundle_path:
            return {}, "Choose a bundle."
        result = bundles.validate(bundle_path)
        if not result["ok"]:
            failures = "; ".join(
                f"{c['label']}: {c['detail']}"
                for c in result["checks"] if c["status"] == "fail"
            )
            return {}, f"{bundle_path} is not usable — {failures}"
        return {**empty, "bundle_path": bundle_path}, None

    trained_model = TrainedModel.objects.filter(
        pk=posted.get("trained_model") or None).first()
    if trained_model is None:
        return {}, "Choose a trained model."
    return {**empty, "trained_model": trained_model}, None


# --------------------------------------------------------------------- context
def form_values(defaults: dict, posted, *, extra: dict | None = None) -> dict:
    """The values to render the pipeline half with.

    A re-render after a validation warning keeps whatever the operator typed
    (``posted`` wins); a first render uses the selected model's own pipeline
    metadata, so the page *arrives* configured the way the model was trained
    instead of blank. ``extra`` adds the caller's own run knobs (a video's
    ``frame_stride``) with their first-render defaults.

    ``chain`` is a list in the metadata and a comma-separated string in the form.
    """
    extra = extra or {}
    fields = (*pipeline_meta.FIELDS, *extra)
    if posted.get("apply"):
        return {key: posted.get(key, "") for key in fields}

    values = {key: defaults.get(key) for key in pipeline_meta.FIELDS}
    values["chain"] = ", ".join(defaults.get("chain") or [])
    if values.get("score_threshold") is None:
        values["score_threshold"] = DEFAULT_SCORE_THRESHOLD
    values.update(extra)
    return {key: "" if value is None else value for key, value in values.items()}


def pipeline_context() -> dict:
    """The vocabulary half of the partial's context: the pipeline dropdown, and
    which rows each pipeline uses.

    Rows opt in by naming pipelines in ``data-pipelines``; the script in
    ``_inference_scripts.html`` hides the rest. Same convention as the
    model-preview page.
    """
    return {
        "pipeline_choices": PIPELINE_CHOICES,
        "detector_pipelines": " ".join(sorted(pipelines.DETECTOR_PIPELINES)),
        "tiling_pipelines": " ".join([
            pipelines.BATCH_DETECT, pipelines.BATCH_PEOPLE, pipelines.CHAIN]),
        "chain_pipelines": pipelines.CHAIN,
        # Every pipeline merges several per-frame predictions back together; only
        # "raw" has nothing to merge.
        "merging_pipelines": " ".join(
            value for value, _label in pipelines.PIPELINE_CHOICES),
        "default_score_threshold": DEFAULT_SCORE_THRESHOLD,
    }


def model_context(model_source: str, posted) -> dict:
    """The model half of the partial's context, for the chosen ``model_source``.

    Also returns ``defaults`` — the selected model's pipeline metadata, which the
    caller feeds to :func:`form_values` so the prefill happens **server-side on
    page load** rather than only on a JS change event.

    Both model lists are newest-first and the first entry is preselected, so the
    page arrives with a model chosen *and* its pipeline filled in: the common case
    ("run this model the way it was trained") needs no selection at all.

    The bundle branch is deliberately different — no preselection and no
    prefill. A bundle's geometry arrives via the explicit "Sync bundle" press,
    which is also the only thing that reports whether the bundle loads *here*
    (``bundles.validate``); landing on a preselected bundle with its fields
    already filled would imply that check had happened.
    """
    from django.urls import reverse

    from training.models import TrainedModel

    artifacts = exports.list_artifacts()
    is_exported = model_source == EXPORTED
    is_bundle = model_source == BUNDLE
    bundle_list = bundles.list_bundles() if is_bundle else []
    trained_models = list(
        TrainedModel.objects
        .select_related("source_run_result__run__experiment")
        .order_by("-created_at", "name")
    )
    trained_defaults = {
        str(tm.pk): pipeline_meta.for_trained_model(tm) for tm in trained_models
    }
    # An artifact with neither a ".pipeline.json" sidecar nor a catalogued model
    # behind it has nothing on record (see exports.read_pipeline_defaults) and is
    # simply left out of the map — selecting it leaves the form as it stands.
    artifact_defaults = {
        a["relpath"]: d
        for a in artifacts
        for d in [exports.read_pipeline_defaults(a["relpath"])] if d
    }

    if is_bundle:
        selected = posted.get("bundle_path") or ""
        defaults = pipeline_meta.raw()
    elif is_exported:
        selected = posted.get("artifact_path") or (
            artifacts[0]["relpath"] if artifacts else "")
        defaults = artifact_defaults.get(selected) or pipeline_meta.raw()
    else:
        selected = posted.get("trained_model") or (
            str(trained_models[0].pk) if trained_models else "")
        defaults = trained_defaults.get(selected) or pipeline_meta.raw()

    return {
        "model_source": model_source,
        "is_exported": is_exported,
        "is_bundle": is_bundle,
        "trained_models": trained_models,
        # Passed as dicts, not pre-serialized: the template renders them with
        # ``json_script``, which escapes the HTML-significant characters
        # ``json.dumps`` does not. Model names, artifact filenames and detector
        # paths all reach these maps, and a ``</script>`` in any of them would
        # otherwise close the script element early.
        "trained_defaults": trained_defaults,
        "artifacts": artifacts,
        "artifact_defaults": artifact_defaults,
        "bundles": bundle_list,
        "bundles_root": str(bundles.bundles_root()) if is_bundle else "",
        "bundle_sync_url": reverse("bundle-sync"),
        "selected_model": selected,
        "exports_root": str(exports.exports_root()),
        "defaults": defaults,
    }


def source_step_context() -> dict:
    """What step 1 (pick a model *kind*) needs to describe the three choices."""
    return {
        "model_source_choices": MODEL_SOURCE_CHOICES,
        "default_model_source": TRAINED,
        "artifact_count": len(exports.list_artifacts()),
        "exports_root": str(exports.exports_root()),
        "bundle_count": len(bundles.list_bundles()),
        "bundles_root": str(bundles.bundles_root()),
    }
