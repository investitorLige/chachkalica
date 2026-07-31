"""Standalone admin pages and endpoints that aren't tied to a single model.

Two of them:

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
"""

from __future__ import annotations

import json
from pathlib import Path

from django.conf import settings
from django.contrib import admin
from django.http import Http404, JsonResponse
from django.template.response import TemplateResponse
from django.views.decorators.http import require_POST

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
