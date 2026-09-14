"""Compare a model's class space against a dataset's, and translate between them.

The service behind the "Check classes" button on the Datasets tab's *Evaluate a
model on this dataset…* form, and the validator for the class map that button
produces.

Why it exists
-------------
An eval scores predictions against ground truth **by class name**
(``friendy_chachkalica.metrics.evaluate_detection``): both spaces are remapped
onto their intersection, and anything outside it is dropped. That is the right
default — a class the model was never trained on would otherwise score a hard
AP of 0 and drag the mean down — but it means two taxonomies that share no name
produce an eval of nothing at all: every prediction and every ground-truth box
dropped, and 0.0 reported for mAP, precision and recall. A single-class ``gun``
detector scored against a ``handgun / rifle / knife / bat`` test set is exactly
that, and the zeros look identical to a model that detects nothing.

So the mismatch is worth showing *before* the run rather than after: this module
reads both class lists, says which names line up, and proposes a map for the
ones that don't. The operator confirms or edits it, and the eval scores the
dataset through it. Leaving the map empty runs the eval exactly as it ran before
this existed.

The map itself is ``{dataset class name: model class name}``, with ``None``
meaning "drop this class from the eval". It is stored on the eval row
(``EvaluatedModelSource.class_map``), sent verbatim in the request YAML, and
applied by ``friendy_chachkalica.metrics.apply_class_map`` in the trainer — see
that function for what it does to the three class spaces.
"""

from __future__ import annotations

from training.services import bundles, exports, inference_form

#: A model class name meaning "this dataset class is not scored at all". The
#: form posts an empty string for it; everything below normalizes to ``None`` so
#: the trainer sees one spelling (mirrors ``chachak.config._parse_class_map``).
DROP = None


def model_class_names(model_source: str, posted) -> tuple[list[str], str]:
    """``(class names, model label)`` for whichever model the form has picked.

    Each source records its classes somewhere different: a catalogued model on
    its own row, an exported artifact in the ``.meta.json`` written beside it, a
    bundle in the sidecar of the model inside it. All three are read without
    loading any weights.

    An empty list means "not on record" — an artifact exported before the
    sidecar existed, or a bundle whose manifest can't be read. That is reported
    as its own outcome rather than treated as "no classes": a map cannot be
    proposed against an unknown space, and guessing one would silently relabel
    every box.
    """
    from training.models import TrainedModel

    if model_source == inference_form.EXPORTED:
        relpath = (posted.get("artifact_path") or "").strip()
        return list(exports.read_class_names(relpath)) if relpath else [], relpath

    if model_source == inference_form.BUNDLE:
        relpath = (posted.get("bundle_path") or "").strip()
        return list(bundles.read_class_names(relpath)) if relpath else [], relpath

    # int() first: filter(pk=...) raises on a non-numeric pk instead of returning
    # nothing, and this reads a value straight off a POST.
    try:
        model = TrainedModel.objects.filter(pk=int(posted.get("trained_model"))).first()
    except (TypeError, ValueError):
        model = None
    if model is None:
        return [], ""
    return [str(name) for name in (model.classes or [])], model.name


def suggest(dataset_classes: list[str], model_classes: list[str]) -> dict:
    """A starting class map for ``dataset_classes`` → ``model_classes``.

    Deliberately unclever, because a wrong guess here silently mislabels every
    box in the run:

    * a dataset class the model already names (ignoring case) maps to itself;
    * otherwise, if the model has exactly **one** class, it maps onto that one —
      a coarse detector scored against a fine-grained test set is the whole case
      this feature exists for, and "everything is a gun" is the only reading a
      one-class model admits;
    * otherwise it is dropped, which scores the rest of the dataset rather than
      inventing a correspondence between two names nobody has claimed are equal.

    Only ever a prefill: the operator sees every row and edits it before
    anything runs.
    """
    by_lower = {name.lower(): name for name in model_classes}
    single = model_classes[0] if len(model_classes) == 1 else DROP
    # The model's own spelling on the right-hand side, always: the eval matches
    # the two spaces by exact name, so "Gun" and "gun" are different classes to
    # it and a case-only difference has to be mapped away like any other.
    return {name: by_lower.get(name.lower(), single) for name in dataset_classes}


def report(model_source: str, posted, dataset_classes: list[str]) -> dict:
    """The checklist the button renders: how the two class spaces line up.

    Shape mirrors :func:`training.services.bundles.validate` — ``{ok, name,
    checks, ...}`` with ``checks`` a list of ``{status, label, detail}`` — so the
    page renders it the same way the "Sync bundle" report is rendered.

    ``ok`` means "this eval would score something as it stands": at least one
    dataset class name the model also predicts. It is false for the disjoint
    case, which is precisely when the map is needed.
    """
    model_classes, label = model_class_names(model_source, posted)
    if not model_classes:
        return {
            "ok": False,
            "unknown": True,
            "name": label or "(no model selected)",
            "model_classes": [],
            "dataset_classes": dataset_classes,
            "suggested": {},
            "checks": [{
                "status": "fail",
                "label": "Model classes",
                "detail": (
                    "not on record — an artifact exported before the .meta.json "
                    "sidecar carried a class map, or a bundle whose manifest cannot "
                    "be read. Re-export it from a checkpoint that records its class "
                    "names; a map cannot be proposed against an unknown class space."
                ),
            }],
        }

    # Exact, case-sensitive membership — the same test evaluate_detection makes.
    # Reporting a case-insensitive match as "lines up" would promise a score the
    # run would not produce; `suggest` maps the case difference away instead.
    model_names = set(model_classes)
    matched = [name for name in dataset_classes if name in model_names]
    unmatched = [name for name in dataset_classes if name not in model_names]
    dataset_names = set(dataset_classes)
    extra = [name for name in model_classes if name not in dataset_names]

    checks = [
        {"status": "info", "label": "Model predicts",
         "detail": f"{len(model_classes)} class(es): {', '.join(model_classes)}"},
        {"status": "info", "label": "Dataset labels",
         "detail": f"{len(dataset_classes)} class(es): {', '.join(dataset_classes)}"},
    ]

    if matched and not unmatched:
        checks.append({
            "status": "ok", "label": "Names line up",
            "detail": "every dataset class is one the model predicts — no map needed.",
        })
    elif matched:
        checks.append({
            "status": "ok", "label": "Scored as-is",
            "detail": f"{', '.join(matched)} — the model predicts these by name.",
        })
        checks.append({
            "status": "fail", "label": "Dropped as-is",
            "detail": (
                f"{', '.join(unmatched)} — the model names none of these, so without a "
                "map their boxes are silently excluded from the eval and never count "
                "as misses."
            ),
        })
    else:
        checks.append({
            "status": "fail", "label": "No shared names",
            "detail": (
                "the two class spaces have nothing in common, so as it stands this "
                "eval drops every prediction and every ground-truth box and reports "
                "0.0 for every metric. Map the dataset's classes onto the model's "
                "below."
            ),
        })

    if extra:
        checks.append({
            "status": "info", "label": "Never labelled",
            "detail": (
                f"{', '.join(extra)} — the model predicts these but the dataset has no "
                "such class, so they are excluded from mAP rather than scoring 0."
            ),
        })

    return {
        "ok": bool(matched),
        "unknown": False,
        "name": label,
        "model_classes": model_classes,
        "dataset_classes": dataset_classes,
        "matched": matched,
        "unmatched": unmatched,
        "extra": extra,
        "suggested": suggest(dataset_classes, model_classes),
        "checks": checks,
    }


def parse_posted(posted, dataset_classes: list[str]) -> tuple[dict, str | None]:
    """Read the class map off the submitted form.

    The page posts one ``class_map__<dataset class name>`` field per dataset
    class, holding the model class name to score it as, or ``""`` to drop it.
    Returns ``({}, None)`` when the operator never opened the mapping section —
    an eval with no map, which is what every eval was before this existed.

    Returns ``({}, message)`` on a map that would score nothing, which is worth
    catching here rather than as a trainer traceback half an hour later. Entries
    that merely restate a dataset class as itself are dropped from the stored map
    so a no-op map is recorded as no map at all.
    """
    prefix = "class_map__"
    posted_names = [key[len(prefix):] for key in posted if key.startswith(prefix)]
    if not posted_names:
        return {}, None

    class_map: dict[str, str | None] = {}
    for name in dataset_classes:
        if prefix + name not in posted:
            continue
        mapped = (posted.get(prefix + name) or "").strip()
        class_map[name] = mapped or DROP

    if all(mapped == name for name, mapped in class_map.items()):
        return {}, None
    if class_map and all(mapped is DROP for mapped in class_map.values()):
        return {}, ("That class map drops every class in the dataset, which would "
                    "score nothing. Map at least one class onto the model's.")
    return class_map, None


def describe(class_map: dict) -> str:
    """One-line human rendering of a stored map, for admin columns and logs."""
    if not class_map:
        return ""
    return ", ".join(
        f"{name} → {mapped}" if mapped else f"{name} → (dropped)"
        for name, mapped in sorted(class_map.items())
    )
