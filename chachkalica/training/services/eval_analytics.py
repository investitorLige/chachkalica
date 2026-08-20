"""Compare the metrics of one or more finished :class:`~training.models.EvalRun`.

Read-only: it takes EvalRuns that already carry an ingested ``metrics`` dict
(see ``ingest.ingest_eval``) and reshapes them into a side-by-side comparison
the admin renders synchronously — one column per eval, one row per metric, with
the winning cell flagged per row. Mirrors fleet's "analyze dataset" action, but
across evaluated models rather than within one dataset.
"""

# (metric key, human label, higher_is_better). The count rows are context, not
# quality scores, so they get no winner highlight; eval time is lower-is-better.
_METRICS = [
    ("map50", "mAP@50", True),
    ("map50_95", "mAP@50-95", True),
    ("precision", "Precision", True),
    ("recall", "Recall", True),
    ("f1", "F1", True),
    ("num_predictions", "# predictions", None),
    ("num_targets", "# targets", None),
    ("num_images", "# images", None),
    ("eval_seconds", "Eval time (s)", False),
    ("eval_seconds_per_frame", "Eval per frame time (s)", False),
]


def _num(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _eval_kind_and_id(eval_run) -> tuple[str, int]:
    """Return ("base"/"pipeline", real-row-pk) for any of the three eval shapes.

    A ``CombinedEval`` union row already carries these (``kind``/``orig_id``);
    a real ``EvalRun``/``PipelineEvalRun`` doesn't, so it's told apart by
    whether it has a ``pipeline`` field at all — only ``PipelineEvalRun`` does.
    Kept duck-typed (no ``eval_pipelines`` import) so this stays the
    training-side, decoupled module its docstring promises.
    """
    kind = getattr(eval_run, "kind", None)
    if kind is not None:
        return kind, eval_run.orig_id
    return ("pipeline" if hasattr(eval_run, "pipeline") else "base"), eval_run.pk


def _column(eval_run) -> dict:
    metrics = eval_run.metrics if isinstance(eval_run.metrics, dict) else {}
    kind, orig_id = _eval_kind_and_id(eval_run)
    return {
        "id": eval_run.pk,
        "kind": kind,
        "orig_id": orig_id,
        "output_dir": getattr(eval_run, "output_dir", "") or "",
        "model": eval_run.trained_model.name,
        "arch": eval_run.trained_model.arch,
        "dataset": eval_run.dataset.name,
        "metrics": metrics,
        # When the eval ran (stamped by the trainer). Fall back to the row's
        # finished timestamp for evals predating the metric.
        "evaluated_at": metrics.get("evaluated_at") or (
            eval_run.finished_at.isoformat(timespec="seconds") if eval_run.finished_at else None
        ),
    }


def _metric_value(metrics, key):
    if key == "eval_seconds_per_frame":
        eval_seconds = _num(metrics.get("eval_seconds"))
        num_images = _num(metrics.get("num_images"))
        if eval_seconds is None or not num_images:
            return None
        return eval_seconds / num_images
    return _num(metrics.get(key))


def _winner(values, higher_is_better) -> float | None:
    """The best numeric value in ``values``, or None if no contest applies."""
    nums = [v for v in values if v is not None]
    if higher_is_better is None or len(nums) < 2 or len(set(nums)) < 2:
        return None  # counts, single column, or an all-tie row → nothing to flag
    return max(nums) if higher_is_better else min(nums)


def _overall_rows(columns) -> list[dict]:
    rows = []
    for key, label, higher_is_better in _METRICS:
        values = [_metric_value(col["metrics"], key) for col in columns]
        best = _winner(values, higher_is_better)
        rows.append({
            "label": label,
            "cells": [
                {"value": v, "is_best": best is not None and v == best}
                for v in values
            ],
        })
    return rows


def _class_rows(columns) -> list[dict]:
    """Per-class AP comparison, unioning the classes each eval reported."""
    names: dict = {}  # class_id -> class_name (first non-empty wins)
    for col in columns:
        for class_id, stats in (col["metrics"].get("per_class") or {}).items():
            names.setdefault(class_id, (stats or {}).get("class_name") or str(class_id))

    rows = []
    for class_id, class_name in names.items():
        ap50 = [_num((col["metrics"].get("per_class") or {}).get(class_id, {}).get("ap50")) for col in columns]
        ap = [_num((col["metrics"].get("per_class") or {}).get(class_id, {}).get("ap50_95")) for col in columns]
        best50, best = _winner(ap50, True), _winner(ap, True)
        rows.append({
            "class_name": class_name,
            "ap50": [{"value": v, "is_best": best50 is not None and v == best50} for v in ap50],
            "ap50_95": [{"value": v, "is_best": best is not None and v == best} for v in ap],
        })
    return rows


_HEADLINE_TILES = [
    ("map50", "mAP@50", "pct"),
    ("map50_95", "mAP@50-95", "pct"),
    ("precision", "Precision", "pct"),
    ("recall", "Recall", "pct"),
    ("f1", "F1", "pct"),
    ("num_images", "Images", "int"),
    ("num_predictions", "Predictions", "int"),
    ("num_targets", "Targets", "int"),
    ("eval_seconds", "Eval time", "time"),
]


def _headline_tiles(metrics: dict) -> list[dict]:
    return [
        {"label": label, "value": _num(metrics.get(key)), "kind": kind}
        for key, label, kind in _HEADLINE_TILES
    ]


def _per_class_rows(metrics: dict) -> list[dict]:
    """Full per-class metric rows for a single eval's summary display.

    Unlike :func:`_class_rows` (AP only, one column per compared eval), this is
    for one eval at a time, so it surfaces everything ``per_class`` carries.
    """
    per_class = metrics.get("per_class")
    if not isinstance(per_class, dict):
        return []
    rows = [
        {
            "class_name": (stats or {}).get("class_name") or str(class_id),
            "ap50": _num((stats or {}).get("ap50")),
            "ap50_95": _num((stats or {}).get("ap50_95")),
            "precision": _num((stats or {}).get("precision")),
            "recall": _num((stats or {}).get("recall")),
            "f1": _num((stats or {}).get("f1")),
            "gt": (stats or {}).get("ground_truth_count"),
            "pred": (stats or {}).get("prediction_count"),
        }
        for class_id, stats in per_class.items()
    ]
    rows.sort(key=lambda r: r["class_name"])
    return rows


def summary(eval_run) -> dict:
    """Single-eval metrics summary for the change_form "metrics" field display.

    Reshapes the raw ``metrics`` JSON into headline tiles, a per-class table, and
    a confusion matrix (when stored) for
    ``admin/eval_pipelines/metrics_summary.html`` — the prettified replacement for
    Django's default one-line ``json.dumps`` readonly rendering.
    """
    metrics = eval_run.metrics if isinstance(eval_run.metrics, dict) else {}
    if not metrics:
        return {"has_metrics": False}
    column = _column(eval_run)
    return {
        "has_metrics": True,
        "evaluated_at": column["evaluated_at"],
        "tiles": _headline_tiles(metrics),
        "class_rows": _per_class_rows(metrics),
        "confusion_matrix": _confusion_matrix(column),
    }


def _confusion_matrix(column) -> dict | None:
    """Reshape a column's stored ``confusion_matrix`` metric for the heatmap table.

    Returns None for evals whose metrics predate the confusion matrix (they simply
    won't render one). Rows are ground-truth classes, columns predictions, both with
    a trailing "background" axis label. Each cell carries its raw count plus a
    row-normalised ``intensity`` (share of that truth class) so the template can shade
    the heatmap without re-deriving totals; the diagonal and background cells are
    flagged so correct detections and errors can be coloured differently.
    """
    cm = column["metrics"].get("confusion_matrix")
    if not isinstance(cm, dict):
        return None
    matrix = cm.get("matrix")
    labels = cm.get("labels")
    if not isinstance(matrix, list) or not isinstance(labels, list) or not matrix:
        return None

    background = cm.get("background_index")
    if not isinstance(background, int):
        background = len(labels)
    axis = [str(label) for label in labels] + ["background"]

    rows = []
    for i, raw_row in enumerate(matrix):
        if not isinstance(raw_row, list):
            return None
        values = [v if isinstance(v, (int, float)) and not isinstance(v, bool) else 0 for v in raw_row]
        total = sum(values)
        rows.append({
            "label": axis[i] if i < len(axis) else str(i),
            "is_background": i == background,
            "cells": [
                {
                    "value": v,
                    "is_diagonal": i == j and i != background,
                    "is_background": i == background or j == background,
                    "intensity": round(v / total, 4) if total else 0.0,
                }
                for j, v in enumerate(values)
            ],
        })

    # When the eval stored the re-thresholdable event histogram, hand it to the
    # template so the page can rebuild the matrix client-side as the user drags a
    # confidence slider. Older evals lack it and simply render the static matrix.
    data = column["metrics"].get("confusion_matrix_data")
    interactive = _confusion_data_for_slider(data) if isinstance(data, dict) else None

    return {
        "id": column["id"],
        "model": column["model"],
        "arch": column["arch"],
        "dataset": column["dataset"],
        "axis": axis,
        "iou_threshold": _num(cm.get("iou_threshold")),
        "conf_threshold": _num(cm.get("conf_threshold")),
        "rows": rows,
        "interactive": interactive,
        # Pre-built DOM id for the json_script block; Django's ``add`` filter can't
        # concatenate a string with the integer pk in the template.
        "json_id": f"cmdata-{column['id']}",
    }


def _confusion_data_for_slider(data) -> dict | None:
    """Validate and normalise a stored ``confusion_matrix_data`` event histogram.

    Returns a JSON-serialisable dict the template embeds via ``json_script`` for the
    client-side re-thresholding code, or None if the stored shape is unusable (so the
    page falls back to the static matrix).
    """
    labels = data.get("labels")
    matched = data.get("matched")
    false_positives = data.get("false_positives")
    misses = data.get("misses")
    background = data.get("background_index")
    if not isinstance(labels, list) or not isinstance(matched, list):
        return None
    if not isinstance(false_positives, list) or not isinstance(misses, list):
        return None
    if not isinstance(background, int):
        background = len(labels)

    floor = _num(data.get("floor"))
    return {
        "labels": [str(label) for label in labels],
        "background_index": background,
        "floor": floor if floor is not None else 0.01,
        "matched": matched,
        "false_positives": false_positives,
        "misses": misses,
    }


def compare(eval_runs) -> dict:
    """Build the comparison context for a list of EvalRuns (metrics required).

    Columns are ordered best-mAP@50 first so the strongest model reads left-to-right.
    """
    columns = sorted(
        (_column(e) for e in eval_runs),
        key=lambda c: (_num(c["metrics"].get("map50")) is not None, _num(c["metrics"].get("map50")) or 0),
        reverse=True,
    )
    return {
        "columns": columns,
        "overall_rows": _overall_rows(columns),
        "class_rows": _class_rows(columns),
        "confusion_matrices": [cm for cm in (_confusion_matrix(c) for c in columns) if cm],
    }
