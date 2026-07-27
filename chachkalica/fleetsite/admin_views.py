"""Standalone admin pages that aren't tied to a single model.

Currently just the TRT/ONNX/PT benchmark console: a read-only dark "instrument
console" rendering the results of ``inferlica/benchmark`` (run in the trainer
container). Data is NOT computed here -- each sweep drops one JSON file per
image size into ``data/benchmarks/`` (which is bind-mounted into the web
container at ``/app/data``), and this view just reads whichever sizes are
present and hands the selected one to the template. Adding a new image-size run
is therefore a pure data drop -- no code change and no redeploy.
"""

from __future__ import annotations

import json
from pathlib import Path

from django.conf import settings
from django.contrib import admin
from django.http import Http404
from django.template.response import TemplateResponse

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
