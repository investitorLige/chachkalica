"""Discovery and validation of infer bundles under the bundle root.

A *bundle* is what ``chachak.bundle_export`` writes: one directory carrying a
complete pipeline — ``pipeline.json`` (the manifest, "Contract C"), the exported
``models/`` it runs, and a vendored ``runtime/`` plus ``infer.py`` that run it
with no chachak checkout at all. The export jobs drop one beside every artifact
(see ``training.jobs._bundle_after_export``), and because it is self-contained it
*travels*: scp a bundle onto another machine and it is still a runnable pipeline.

This module is what makes a bundle a first-class **model source** for the serving
side, the way :mod:`training.services.exports` is for loose artifacts:

* :func:`list_bundles` scans :func:`bundles_root`, so every inference form can
  offer bundles in a dropdown instead of asking for a path;
* :func:`pipeline_defaults` translates a manifest into the one metadata schema
  every consumer already reads (:mod:`training.services.pipeline_meta`), so a
  bundle prefills a form exactly the way a ``.pt`` or a ``.pipeline.json``
  sidecar does;
* :func:`apply_defaults` writes that translation onto an inference row. This is
  why bundle-sourced pipeline fields are read-only in the admin and re-derived on
  save: a bundle's geometry was tuned together with its weights, so the *bundle*
  is the record of what runs — not whatever a form happened to post.
* :func:`validate` backs the "Sync bundle" button: every structural check, plus
  an optional real load through the trainer.

Two things differ from an exported artifact. A bundle is addressed by its
**directory**, not by the model file inside it — so the same relpath answers for
the model, the detector and the geometry at once. And it needs no
``.pipeline.json`` sidecar: its own manifest already holds strictly more than the
sidecar does, which is what lets a bundle arrive from another machine and still
describe itself completely.
"""

import json
from pathlib import Path

from training import pipelines
from training.models import TrainingSettings
from training.services import config_gen, pipeline_meta

# The manifest file that *makes* a directory a bundle: found here, this is a
# bundle; absent, the directory is just a directory.
MANIFEST_NAME = "pipeline.json"

# Highest ``pipeline.json`` schema this reader understands. Mirrors
# ``chachak.bundle_export.manifest.SCHEMA_VERSION`` — we can't import chachak in
# the app env (it pulls in torch), so this is a hand-maintained mirror, same
# convention as :mod:`training.pipelines`.
SUPPORTED_SCHEMA_VERSION = 1

# Model/detector suffixes a bundle can carry; chachak's ``load_checkpoint_adapter``
# dispatches on these (kept the same as ``exports.ARTIFACT_SUFFIXES``, since the
# artifacts inside a bundle are the very files that module scans for loose).
ARTIFACT_SUFFIXES = (".onnx", ".engine")

# Where :func:`validate`'s load test writes its synthetic frame. Under the bundle
# root because that is already on the filesystem the trainer shares (it has to
# be: ``/predict_image`` takes a path and opens it itself), and dot-prefixed
# because it is machinery, not a bundle — :func:`list_bundles` ignores it either
# way, having no manifest.
LOAD_CHECK_DIRNAME = ".load_check"

_KIND_LABELS = {".onnx": "ONNX", ".engine": "TensorRT"}


class BundleError(ValueError):
    """A bundle is missing, outside the root, or not a bundle at all."""


def bundles_root(ts: TrainingSettings | None = None) -> Path:
    """Absolute bundle directory (relative settings resolve to the project root)."""
    ts = ts or TrainingSettings.load()
    return config_gen._resolve(ts.bundles_root)


def list_bundles(ts: TrainingSettings | None = None) -> list[dict]:
    """Bundles under :func:`bundles_root`, newest first.

    A bundle is any directory holding a :data:`MANIFEST_NAME`; the scan is
    recursive so bundles may be grouped in subdirectories (and so a whole tree of
    them can be copied in at once). Each entry is
    ``{relpath, name, pipeline, kind, classes, size_mb, mtime, absolute, ok}``,
    where ``relpath`` is the form value and ``ok`` is a cheap "the manifest parses
    and its model file is there" flag — enough to label a broken bundle in a
    dropdown without running the full :func:`validate`.

    Never raises: a directory whose manifest is unreadable is still listed (with
    ``ok`` false), because hiding it would leave the operator with a bundle that
    is visibly on disk and inexplicably absent from the form.
    """
    root = bundles_root(ts)
    if not root.is_dir():
        return []

    entries = []
    for manifest_path in root.rglob(MANIFEST_NAME):
        if not manifest_path.is_file():
            continue
        bundle_dir = manifest_path.parent
        # "." when the root itself is a bundle (an operator who pointed the
        # setting straight at one). Left as-is rather than prettified to the
        # directory's name, because this string has to round-trip through
        # :func:`resolve` — and "." does.
        relpath = bundle_dir.relative_to(root).as_posix()
        manifest = _read_manifest_file(manifest_path)
        model_rel = ((manifest or {}).get("artifacts") or {}).get("model") or ""
        model_path = bundle_dir / model_rel if model_rel else None
        entries.append({
            "relpath": relpath,
            "name": (manifest or {}).get("name") or bundle_dir.name,
            "pipeline": (manifest or {}).get("pipeline") or "",
            "kind": _KIND_LABELS.get(
                Path(model_rel).suffix.lower() if model_rel else "", "unknown"),
            "classes": list(((manifest or {}).get("classes") or {}).values()),
            "size_mb": (
                round(model_path.stat().st_size / (1024 * 1024), 1)
                if model_path is not None and model_path.is_file() else 0.0
            ),
            "mtime": manifest_path.stat().st_mtime,
            "absolute": str(bundle_dir),
            "ok": bool(manifest) and model_path is not None and model_path.is_file(),
        })
    entries.sort(key=lambda e: (-e["mtime"], e["relpath"]))
    return entries


def resolve(relpath: str, ts: TrainingSettings | None = None) -> Path:
    """Absolute bundle directory for a relpath under :func:`bundles_root`.

    Raises :class:`BundleError` for a blank value, a path escaping the root, a
    missing directory, or a directory with no manifest — the same
    "reject anything a crafted form value could point at" contract as
    :func:`training.services.exports.resolve`.
    """
    relpath = (relpath or "").strip()
    if not relpath:
        raise BundleError("No bundle selected.")

    root = bundles_root(ts)
    candidate = (root / relpath).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        raise BundleError(
            f"{relpath!r} is outside the bundle directory ({root})."
        ) from None
    if not candidate.is_dir():
        raise BundleError(f"Bundle not found: {candidate}")
    if not (candidate / MANIFEST_NAME).is_file():
        raise BundleError(
            f"{relpath!r} is not a bundle — no {MANIFEST_NAME} in {candidate}."
        )
    return candidate


def _read_manifest_file(path: Path) -> dict | None:
    try:
        manifest = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return manifest if isinstance(manifest, dict) else None


def read_manifest(relpath: str, ts: TrainingSettings | None = None) -> dict:
    """A bundle's parsed ``pipeline.json``.

    Raises :class:`BundleError` when it can't be read, isn't an object, or
    declares a ``schema_version`` newer than :data:`SUPPORTED_SCHEMA_VERSION` —
    a newer bundle is refused loudly rather than read with fields this code
    doesn't know about silently dropped.
    """
    bundle_dir = resolve(relpath, ts)
    manifest = _read_manifest_file(bundle_dir / MANIFEST_NAME)
    if manifest is None:
        raise BundleError(f"{relpath}: {MANIFEST_NAME} is missing or not valid JSON.")

    version = manifest.get("schema_version", 1)
    try:
        version = int(version)
    except (TypeError, ValueError):
        raise BundleError(f"{relpath}: schema_version {version!r} is not a number.") from None
    if version > SUPPORTED_SCHEMA_VERSION:
        raise BundleError(
            f"{relpath}: bundle schema_version {version} is newer than this app "
            f"supports ({SUPPORTED_SCHEMA_VERSION}). Update the app to run it."
        )
    return manifest


def model_artifact(relpath: str, ts: TrainingSettings | None = None) -> Path:
    """Absolute path of the model a bundle runs (``artifacts.model``).

    Bundle-relative by contract, and checked for escaping the bundle for the same
    reason :func:`resolve` checks the root: the manifest is a file that arrived
    from somewhere else.
    """
    bundle_dir = resolve(relpath, ts)
    manifest = read_manifest(relpath, ts)
    return _artifact_in(bundle_dir, (manifest.get("artifacts") or {}).get("model"),
                        relpath, "model")


def _artifact_in(bundle_dir: Path, rel: str | None, relpath: str, role: str) -> Path:
    if not rel:
        raise BundleError(f"{relpath}: manifest names no {role} artifact.")
    candidate = (bundle_dir / rel).resolve()
    try:
        candidate.relative_to(bundle_dir.resolve())
    except ValueError:
        raise BundleError(
            f"{relpath}: {role} path {rel!r} points outside the bundle."
        ) from None
    if candidate.suffix.lower() not in ARTIFACT_SUFFIXES:
        raise BundleError(
            f"{relpath}: {role} {rel!r} is not a runnable artifact "
            f"({' / '.join(ARTIFACT_SUFFIXES)})."
        )
    if not candidate.is_file():
        raise BundleError(f"{relpath}: {role} artifact missing at {candidate}.")
    return candidate


def _manifest_defaults(manifest: dict, bundle_dir: Path) -> dict:
    """Translate a manifest into :mod:`training.services.pipeline_meta`'s schema.

    The manifest is chachak's flattened ``PipelineConfig`` (nested ``tiling`` /
    ``detector`` objects, thresholds promoted into ``defaults``); the metadata
    schema is the flat vocabulary every Django-side consumer prefills from. This
    function is the only place the two vocabularies meet.

    The detector comes out as the **bundled** detector's absolute path, not the
    host path under ``provenance`` — the whole point of a bundle is that its
    detector travels with it, so a bundle copied from another machine resolves
    against files that are actually here.
    """
    tiling = manifest.get("tiling") or {}
    detector = manifest.get("detector") or {}
    shipped = manifest.get("defaults") or {}
    detector_rel = (manifest.get("artifacts") or {}).get("detector")
    return pipeline_meta.normalize({
        "pipeline": manifest.get("pipeline") or pipeline_meta.RAW,
        "detector_checkpoint": str(bundle_dir / detector_rel) if detector_rel else "",
        "detector_expand_ratio": detector.get("expand_ratio"),
        "detector_min_box_size": detector.get("min_box_size"),
        "tile_size_px": tiling.get("tile_size_px"),
        "tile_width_pct": tiling.get("tile_width_pct"),
        "tile_height_pct": tiling.get("tile_height_pct"),
        "overlap": tiling.get("overlap"),
        "merge_nms_iou": manifest.get("merge_nms_iou"),
        "chain": manifest.get("chain") or [],
        # `defaults.conf` is the operating point the bundle ships with — what its
        # own infer.py runs at when given no --conf. `score_threshold` is the
        # pre-`defaults` spelling, kept as a fallback for older manifests.
        "score_threshold": shipped.get("conf", manifest.get("score_threshold")),
    })


def pipeline_defaults(relpath: str, ts: TrainingSettings | None = None) -> dict | None:
    """Pipeline metadata for a bundle, or ``None`` if it can't be read.

    The ``None`` mirrors :func:`training.services.exports.read_pipeline_defaults`:
    consumers building a "pick a model, get its pipeline" map leave unreadable
    entries out rather than failing the page they are rendering.
    """
    try:
        bundle_dir = resolve(relpath, ts)
        manifest = read_manifest(relpath, ts)
    except BundleError:
        return None
    return _manifest_defaults(manifest, bundle_dir)


# The fields a bundle owns. `score_threshold` is deliberately absent: the bundle
# ships one, but confidence is the one knob its own README treats as tunable, so
# an operator's value wins. Kept in sync with GEOMETRY_FIELDS in
# training/static/training/bundle_sync.js, which locks exactly these inputs.
GEOMETRY_FIELDS = (
    "pipeline", "detector_checkpoint", "detector_expand_ratio",
    "detector_min_box_size", "tile_size_px", "tile_width_pct",
    "tile_height_pct", "overlap", "merge_nms_iou", "chain",
)


def geometry(defaults: dict) -> dict:
    """:data:`GEOMETRY_FIELDS` out of ``defaults``, typed for the model fields.

    ``tile_size_px`` is coerced to ``int`` because the manifest carries chachak's
    float-tolerant value while the field is a ``PositiveIntegerField``, and
    ``chain`` is copied so callers can't mutate the metadata dict through it.
    """
    out = {field: defaults[field] for field in GEOMETRY_FIELDS}
    if out["tile_size_px"] is not None:
        out["tile_size_px"] = int(out["tile_size_px"])
    out["chain"] = list(out["chain"] or [])
    return out


def apply_defaults(obj, relpath: str, ts: TrainingSettings | None = None) -> dict:
    """Overwrite ``obj``'s pipeline fields from the bundle at ``relpath``.

    Called on save by every bundle-sourced inference row
    (:class:`videos.models.InferenceJob`,
    :class:`cameras.models.CameraInference`), which is what makes the admin's
    read-only bundle fields *true* rather than cosmetic: whatever a form posted
    for the geometry is replaced by what the bundle actually carries, so the
    saved row can't drift from the pipeline that ran.

    Returns the full metadata it applied (a superset of what it wrote — see
    :func:`geometry`). Raises :class:`BundleError` if the bundle is unreadable.
    """
    bundle_dir = resolve(relpath, ts)
    defaults = _manifest_defaults(read_manifest(relpath, ts), bundle_dir)
    for field, value in geometry(defaults).items():
        setattr(obj, field, value)
    return defaults


def _needs_detector(defaults: dict) -> bool:
    return defaults["pipeline"] in pipelines.DETECTOR_PIPELINES or (
        defaults["pipeline"] == pipelines.CHAIN
        and any(c in pipelines.DETECTOR_PIPELINES for c in defaults["chain"])
    )


def probe_image(ts: TrainingSettings | None = None) -> Path:
    """A synthetic frame for :func:`validate`'s load test, created on first use.

    Uniform mid-grey rather than anything from a dataset: the point is to make
    the trainer *build* the pipeline (both adapters, see the trainer's
    ``_build_predict_runtime``), not to detect anything, and a synthesised frame
    needs no dataset to exist and no images-root to be configured.
    """
    import cv2
    import numpy as np

    directory = bundles_root(ts) / LOAD_CHECK_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "probe.jpg"
    if not path.is_file():
        cv2.imwrite(str(path), np.full((480, 640, 3), 127, dtype=np.uint8))
    return path


def _check(label, status, detail):
    return {"label": label, "status": status, "detail": detail}


def validate(relpath: str, *, load_test: bool = False,
             ts: TrainingSettings | None = None) -> dict:
    """Everything the "Sync bundle" button reports about one bundle.

    Returns ``{"ok", "name", "defaults", "checks": [{label, status, detail}]}``
    where ``status`` is ``"ok"``, ``"fail"`` or ``"info"`` (a note or a skipped
    check, neither of which makes the bundle unusable). ``ok`` is "no check
    failed"; ``defaults`` is the metadata to fill the form with, or ``None`` when
    the bundle couldn't be read far enough to have any.

    The structural half is pure filesystem work — instant, and safe to run
    against a bundle while a camera is inferring. ``load_test`` adds the half
    that proves the artifacts actually *load*: the pipeline is built and run on
    :func:`probe_image` through the trainer's ``/predict_image``. That is the only
    check that can catch the failure a copied bundle is most likely to hit — a
    TensorRT engine built for a different GPU or TensorRT version — but it takes
    the trainer's GPU lock and evicts its single-entry model cache, so it is opt
    in.

    Never raises: every failure becomes a failing check, because this is rendered
    into a form the operator is in the middle of filling out.
    """
    checks: list[dict] = []

    # ---------------------------------------------------------------- manifest
    try:
        bundle_dir = resolve(relpath, ts)
        manifest = read_manifest(relpath, ts)
    except BundleError as exc:
        checks.append(_check("Manifest", "fail", str(exc)))
        return {"ok": False, "name": relpath, "defaults": None, "checks": checks}

    checks.append(_check(
        "Manifest", "ok",
        f"{MANIFEST_NAME} v{manifest.get('schema_version', 1)}, exported "
        f"{manifest.get('bundle', {}).get('created_utc') or 'at an unrecorded time'}"))

    defaults = _manifest_defaults(manifest, bundle_dir)
    name = manifest.get("name") or bundle_dir.name

    # ---------------------------------------------------------------- pipeline
    known = {pipeline_meta.RAW, *(v for v, _ in pipelines.PIPELINE_CHOICES)}
    if defaults["pipeline"] in known:
        chain = ", ".join(defaults["chain"])
        checks.append(_check(
            "Pipeline", "ok",
            defaults["pipeline"] + (f" ({chain})" if chain else "")))
    else:
        checks.append(_check(
            "Pipeline", "fail",
            f"{defaults['pipeline']!r} is not a pipeline this app can serve "
            f"({', '.join(sorted(known))})."))

    unknown_chain = [c for c in defaults["chain"] if c not in known or c == pipeline_meta.RAW]
    if unknown_chain:
        checks.append(_check("Chain", "fail",
                             f"unknown chain member(s): {', '.join(unknown_chain)}"))

    # ------------------------------------------------------------------- model
    model_path = None
    try:
        model_path = _artifact_in(
            bundle_dir, (manifest.get("artifacts") or {}).get("model"), relpath, "model")
        size_mb = round(model_path.stat().st_size / (1024 * 1024), 1)
        checks.append(_check(
            "Model artifact", "ok",
            f"{model_path.relative_to(bundle_dir).as_posix()} "
            f"({_KIND_LABELS[model_path.suffix.lower()]}, {size_mb} MB)"))
        checks.append(_model_meta_check(model_path, "Model metadata"))
    except BundleError as exc:
        checks.append(_check("Model artifact", "fail", str(exc)))

    # ---------------------------------------------------------------- detector
    detector_rel = (manifest.get("artifacts") or {}).get("detector")
    detector_path = None
    if _needs_detector(defaults):
        try:
            detector_path = _artifact_in(bundle_dir, detector_rel, relpath, "detector")
            checks.append(_check(
                "Detector artifact", "ok",
                f"{detector_path.relative_to(bundle_dir).as_posix()} "
                f"({_KIND_LABELS[detector_path.suffix.lower()]})"))
            checks.append(_model_meta_check(detector_path, "Detector metadata"))
        except BundleError as exc:
            checks.append(_check(
                "Detector artifact", "fail",
                f"pipeline {defaults['pipeline']!r} crops around detected people, "
                f"so the bundle must carry a detector — {exc}"))
    elif detector_rel:
        checks.append(_check("Detector artifact", "info",
                             f"{detector_rel} carried but unused by "
                             f"{defaults['pipeline']!r}"))
    else:
        checks.append(_check("Detector artifact", "info",
                             f"not required by {defaults['pipeline']!r}"))

    # ----------------------------------------------------------------- classes
    classes = manifest.get("classes") or {}
    if classes:
        checks.append(_check("Classes", "ok",
                             f"{len(classes)} — {', '.join(map(str, classes.values()))}"))
    else:
        checks.append(_check("Classes", "fail",
                             "the manifest carries no class names, so detections "
                             "would be unlabelled"))

    # ------------------------------------------------------------------ format
    declared = (manifest.get("bundle") or {}).get("default_format") or ""
    if model_path is not None:
        actual = "engine" if model_path.suffix.lower() == ".engine" else "onnx"
        if declared and declared != actual:
            checks.append(_check(
                "Format", "fail",
                f"manifest declares default_format {declared!r} but the model is "
                f"a .{actual} file"))
        elif actual == "engine":
            checks.append(_check(
                "Format", "info",
                "TensorRT — a .engine only loads on the GPU model and TensorRT "
                "version it was built on. Run the load test below to prove this "
                "one loads here."))
        else:
            checks.append(_check("Format", "ok", "ONNX — portable across machines"))

    # -------------------------------------------------- self-contained runtime
    missing = [n for n in ("runtime", "infer.py") if not (bundle_dir / n).exists()]
    if missing:
        checks.append(_check(
            "Standalone runtime", "info",
            f"no {' or '.join(missing)} — this app can still serve the bundle, but "
            f"it won't run on its own with `python infer.py`"))
    else:
        checks.append(_check("Standalone runtime", "ok",
                             "runtime/ + infer.py present"))

    # --------------------------------------------------------------- load test
    foreign = _foreign_build(model_path) if model_path is not None else None
    if foreign:
        # Running it would fail, correctly and uselessly. The engine was compiled
        # somewhere else, so it cannot load here by design — reporting that as a
        # failure would read as "this bundle is broken" when it is exactly what was
        # asked for. Say where it belongs instead.
        checks.append(_check("Load test", "info", f"skipped — {foreign}"))
    elif not load_test:
        checks.append(_check("Load test", "info", "not run"))
    elif model_path is None:
        checks.append(_check("Load test", "info",
                             "skipped — no model artifact to load"))
    else:
        checks.append(_load_test_check(model_path, detector_path, defaults, ts))

    return {
        "ok": not any(c["status"] == "fail" for c in checks),
        "name": name,
        "defaults": defaults,
        "checks": checks,
    }


def _foreign_build(model_path: Path) -> str | None:
    """Why this engine can't be load-tested on this machine, or ``None``.

    A remote build node writes ``built_by: buildnode`` into the engine's
    ``.engine.json``, along with the GPU and TensorRT version it compiled against.
    A plan only deserializes on the pair that produced it, so a bundle from a node
    is not loadable here and never will be — that is the point of having built it
    there. Only engines are affected; an ONNX bundle runs anywhere.
    """
    if model_path.suffix.lower() != ".engine":
        return None
    provenance_path = Path(str(model_path) + ".json")
    if not provenance_path.is_file():
        return None
    try:
        provenance = json.loads(provenance_path.read_text())
    except (OSError, ValueError):
        return None
    if provenance.get("built_by") != "buildnode":
        return None
    gpu = provenance.get("gpu_name") or "another machine's GPU"
    version = provenance.get("tensorrt_version")
    where = f"built on a build node for {gpu}"
    if version:
        where += f" with TensorRT {version}"
    return f"{where}; a TensorRT engine only loads where it was built"


def _model_meta_check(artifact: Path, label: str) -> dict:
    """Report an artifact's ``.meta.json`` (chachak's "Contract B" sidecar).

    Informational, not fatal: the runtime reads this to preprocess correctly, so
    its absence is a real problem for the *bundle's own* ``infer.py`` — but this
    app serves the artifact through the trainer, which re-reads the sidecar
    itself and will say so far more precisely than a guess from here.
    """
    meta_path = artifact.with_suffix(".meta.json")
    if not meta_path.is_file():
        return _check(label, "info", f"no {meta_path.name} beside the artifact")
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, ValueError) as exc:
        return _check(label, "fail", f"{meta_path.name} is unreadable: {exc}")
    arch = meta.get("arch") or "unknown arch"
    class_map = meta.get("class_map") or {}
    size = (meta.get("input") or {}).get("size")
    detail = f"{arch}, {len(class_map)} classes"
    if size:
        detail += f", input {size}"
    return _check(label, "ok", detail)


def _load_test_check(model_path: Path, detector_path: Path | None,
                     defaults: dict, ts: TrainingSettings | None) -> dict:
    """Build and run the bundle's pipeline once, on a synthetic frame.

    Zero boxes is a pass: mid-grey pixels contain nothing to detect, and what is
    being proven is that the artifacts load and the pipeline assembles — the one
    thing no amount of reading files can establish about a TensorRT engine that
    arrived from another machine.
    """
    try:
        from training.services import runner

        image = probe_image(ts)
        payload = config_gen.build_predict_request(
            str(model_path),
            defaults["pipeline"],
            str(image),
            detector_checkpoint=str(detector_path) if detector_path else "",
            detector_expand_ratio=defaults["detector_expand_ratio"],
            detector_min_box_size=defaults["detector_min_box_size"],
            tile_size_px=defaults["tile_size_px"],
            tile_width_pct=defaults["tile_width_pct"],
            tile_height_pct=defaults["tile_height_pct"],
            overlap=defaults["overlap"],
            merge_nms_iou=defaults["merge_nms_iou"],
            chain=defaults["chain"],
            score_threshold=defaults["score_threshold"] or 0.5,
            ts=ts,
        )
        result = runner.predict_image(payload, ts=ts)
    except ValueError as exc:  # an inconsistent pipeline — caught before the call
        return _check("Load test", "fail", str(exc))
    except Exception as exc:  # noqa: BLE001 - trainer/network failure is the answer
        return _check("Load test", "fail", str(exc))

    boxes = result.get("boxes") or []
    names = ", ".join(str(n) for n in (result.get("classes") or {}).values())
    return _check(
        "Load test", "ok",
        f"pipeline built and ran on a synthetic 640×480 frame — {len(boxes)} "
        f"detections (none expected on flat grey)"
        + (f"; model reports classes: {names}" if names else ""))
