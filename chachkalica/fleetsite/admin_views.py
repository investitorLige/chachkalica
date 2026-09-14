"""Standalone admin pages and endpoints that aren't tied to a single model.

Four of them:

- the TRT/ONNX/PT benchmark console: a read-only dark "instrument console"
  rendering the results of ``inferlica/benchmark`` (run in the trainer
  container). Data is NOT computed here -- each sweep drops one JSON file per
  image size into ``data/benchmarks/`` (which is bind-mounted into the web
  container at ``/app/data``), and this view just reads whichever sizes are
  present and hands the selected one to the template. Adding a new image-size run
  is therefore a pure data drop -- no code change and no redeploy.
- :func:`bundle_sync_view`, the JSON endpoint behind every "Sync bundle" button.
  It lives here rather than on a ModelAdmin because both inference surfaces (the
  video "Run model inference..." wizard and the camera live-inference inline) call
  the same one, and it belongs to neither.
- :func:`class_sync_view`, the same idea for the "Check classes" button on the
  dataset-eval form: it compares the selected model's class space against the
  dataset's and proposes a translation between them.
- :func:`dataset_tags_view`, which renders the "what can this eval be sliced by"
  panel for both evaluate forms. Same reason it is here: the model-side form
  starts from a checkpoint and the dataset-side one from a dataset, and neither
  owns the panel.
"""

from __future__ import annotations

import json
from pathlib import Path

from django.conf import settings
from django.contrib import admin
from django.http import Http404, JsonResponse
from django.template.response import TemplateResponse
from django.template.loader import render_to_string
from django.views.decorators.http import require_GET, require_POST

BENCH_DIR = Path(settings.BASE_DIR) / "data" / "benchmarks"


def _load_sizes():
    """Every readable ``<image_size>.json`` in BENCH_DIR, ascending by size.

    Each file is ``{"image_size": int, "label": str, "data": {...}, ...}``.
    A malformed file is skipped rather than 500-ing the whole page.
    """
    if not BENCH_DIR.is_dir():
        return []
    entries = []
    for path in BENCH_DIR.glob("*.json"):
        try:
            meta = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(meta, dict) or "data" not in meta:
            continue
        meta["_path"] = path
        entries.append(meta)
    entries.sort(key=lambda m: m.get("image_size") or 0)
    return entries


def benchmark_console_view(request):
    sizes = _load_sizes()
    if not sizes:
        raise Http404(
            "No benchmark data found. Drop a <image_size>.json into data/benchmarks/ "
            "(produced by inferlica/benchmark)."
        )

    requested = request.GET.get("size")
    chosen = next((m for m in sizes if str(m.get("image_size")) == str(requested)), None)
    if chosen is None:
        # Default to 640 (the reference size) when present, else the smallest.
        chosen = next((m for m in sizes if m.get("image_size") == 640), sizes[0])

    context = {
        **admin.site.each_context(request),
        "title": "Benchmark console",
        "bench": chosen,
        "bench_data": chosen["data"],
        "sizes": [
            {
                "image_size": m.get("image_size"),
                "label": m.get("label") or f"{m.get('image_size')}",
                "current": m is chosen,
            }
            for m in sizes
        ],
    }
    return TemplateResponse(request, "admin/benchmarks/console.html", context)


@require_POST
def bundle_sync_view(request):
    """Validate one infer bundle and return the form values it dictates.

    The endpoint behind the "Sync bundle" button on both inference forms. POST
    ``bundle`` (a path relative to the bundle root) and optionally
    ``load_test=1``; the response is
    :func:`training.services.bundles.validate` verbatim —
    ``{ok, name, defaults, checks}`` — which the page renders as a checklist and
    applies to its pipeline fields.

    POST-only because the load test is expensive and side-effecting (it takes the
    trainer's GPU lock and evicts its warm model), which is not something a URL
    someone pasted should be able to trigger. Wrapped in ``admin_view`` at the
    URLconf, so it is staff-only like the rest of the admin.
    """
    from training.services import bundles

    bundle = (request.POST.get("bundle") or "").strip()
    if not bundle:
        return JsonResponse({"error": "No bundle selected."}, status=400)
    load_test = request.POST.get("load_test") in ("1", "true", "on")

    # bundles.validate never raises for a bad bundle -- it reports. Anything that
    # escapes it is a bug or a broken trainer, and the button should say so rather
    # than the fetch failing with an opaque 500.
    try:
        return JsonResponse(bundles.validate(bundle, load_test=load_test))
    except Exception as exc:  # noqa: BLE001 - surfaced in the checklist
        return JsonResponse({"error": f"{type(exc).__name__}: {exc}"}, status=500)


@require_POST
def class_sync_view(request):
    """Compare the selected model's class space against a dataset's.

    The endpoint behind the "Check classes" button on the Datasets tab's
    *Evaluate a model on this dataset…* form. POST the ``model_source``, the
    matching model field (``trained_model`` / ``artifact_path`` /
    ``bundle_path``) and the ``dataset`` pk; the response is
    :func:`training.services.class_sync.report` verbatim — ``{ok, name, checks,
    model_classes, dataset_classes, suggested, ...}`` — which the page renders as
    a checklist plus a prefilled mapping table.

    The dataset arrives as a **pk**, not a name: the name is a path segment under
    the dataset source root, and a name posted by the client would be one.

    POST-only and staff-only for the same reasons as :func:`bundle_sync_view`,
    though this one only reads sidecars and a classes.txt — no weights are
    loaded and no GPU is taken.
    """
    from fleet.models import Dataset
    from training.services import class_sync, config_gen, inference_form

    model_source = (request.POST.get("model_source") or "").strip()
    if model_source not in dict(inference_form.MODEL_SOURCE_CHOICES):
        return JsonResponse({"error": "Unknown model source."}, status=400)

    # filter(pk=...) *raises* on a non-numeric pk rather than coming back empty,
    # so a hand-rolled POST would 500 instead of being told no.
    try:
        dataset = Dataset.objects.filter(pk=int(request.POST.get("dataset"))).first()
    except (TypeError, ValueError):
        dataset = None
    if dataset is None:
        return JsonResponse({"error": "No such dataset."}, status=400)

    try:
        dataset_classes = config_gen.dataset_classes(dataset)
    except OSError as exc:
        return JsonResponse(
            {"error": f"{dataset.name}: cannot read its classes.txt ({exc})."}, status=400)
    if not dataset_classes:
        return JsonResponse(
            {"error": f"{dataset.name} has no classes.txt, so there is nothing to "
                      "score against or to map."}, status=400)

    # Mirrors bundle_sync_view: report() is written not to raise for a model it
    # cannot read, so anything escaping it is a bug and the button should say so
    # rather than the fetch failing opaquely.
    try:
        return JsonResponse(class_sync.report(model_source, request.POST, dataset_classes))
    except Exception as exc:  # noqa: BLE001 - surfaced in the checklist
        return JsonResponse({"error": f"{type(exc).__name__}: {exc}"}, status=500)


@require_GET
def dataset_tags_view(request):
    """Render the tag-availability panel for one dataset and label source.

    ``?dataset=<pk>&label_source=&annotator=&explicit_labels_path=`` returns
    ``{"html": ...}``. It hands back **rendered HTML, not data**, on purpose: the
    dataset-side form includes the same partial directly, so returning JSON here
    would mean one panel built by a Django template and an identical-looking one
    built by a JS string-concatenator, free to drift apart. One renderer, two
    callers.

    GET because it only reads, which also keeps it out of CSRF's way on a form
    that has not been submitted yet.
    """
    from fleet.models import Annotator, Dataset
    from training.services import tag_availability

    dataset = Dataset.objects.filter(pk=request.GET.get("dataset") or None).first()
    if dataset is None:
        return JsonResponse({"error": "Choose a dataset to see its tags."}, status=400)
    annotator = Annotator.objects.filter(pk=request.GET.get("annotator") or None).first()

    try:
        availability = tag_availability.availability(
            dataset,
            request.GET.get("label_source") or "",
            annotator,
            request.GET.get("explicit_labels_path") or "",
        )
    except Exception as exc:  # noqa: BLE001 - surfaced in the panel, not a 500
        return JsonResponse({"error": f"{type(exc).__name__}: {exc}"}, status=500)

    return JsonResponse({"html": render_to_string(
        "admin/training/_tag_availability.html", {"availability": availability})})
