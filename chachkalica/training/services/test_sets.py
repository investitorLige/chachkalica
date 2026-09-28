"""Several test datasets for one eval request, and the page section comparing them.

Every entry point that queues an eval -- Models → "Evaluate…", Datasets →
"Evaluate a model on these datasets…", and the automatic test eval after a
training run -- accepts more than one dataset. Each dataset still becomes its
own :class:`~training.models.EvalRun` / :class:`~eval_pipelines.models.PipelineEvalRun`,
so nothing downstream (tag analytics, hard images, compare, promote-to-labels)
has to learn about groups; the rows just share a ``test_group`` UUID. The one
place that reads the group is the tag analytics page, which renders
:func:`section` -- per-set metrics side by side, plus the sets pooled -- via
``tag_analytics.test_set_breakdown``.
"""

from __future__ import annotations

import uuid

from django.urls import reverse
from django.utils.http import urlencode

from training.services import tag_analytics


def new_group(dataset_count: int):
    """A fresh group id when a request covers 2+ datasets, else None.

    A single-dataset eval stays ungrouped, so the page shows no one-row section.
    """
    return uuid.uuid4() if dataset_count > 1 else None


def siblings(eval_obj) -> list[tuple[str, object]]:
    """Every eval in ``eval_obj``'s test group as ``(kind, eval)``, itself included.

    Both tables are searched: all rows one request queues are the same kind
    today, but nothing about the group requires it.
    """
    from eval_pipelines.models import PipelineEvalRun
    from training.models import EvalRun

    group = getattr(eval_obj, "test_group", None)
    if group is None:
        return []
    rows = [("base", row) for row in
            EvalRun.objects.filter(test_group=group).select_related("dataset")]
    rows += [("pipeline", row) for row in
             PipelineEvalRun.objects.filter(test_group=group).select_related("dataset")]
    rows.sort(key=lambda pair: (pair[1].dataset.name, pair[1].pk))
    return rows


def _url(kind: str, eval_obj) -> str:
    return reverse("admin:eval_pipelines_combinedeval_tag_analytics") + "?" + urlencode(
        {"kind": kind, "eval": eval_obj.pk})


def section(eval_obj, kind: str) -> dict | None:
    """The "Test sets" section for ``eval_obj``'s page, or None when it has no group.

    Reads each sibling's match table only (cached by ``load_table``) -- none of
    the tag sidecars -- so a group of five costs five parses the first time and
    nothing after.
    """
    group = siblings(eval_obj)
    if len(group) < 2:
        return None

    entries = []
    for sibling_kind, sibling in group:
        table, problem = None, ""
        if sibling.status != sibling.OK:
            problem = f"eval is {sibling.status}"
        else:
            artifact = tag_analytics.match_table_artifact(sibling.output_dir)
            if artifact is None:
                problem = "no match table — run “Build tag analytics data”"
            else:
                try:
                    table = tag_analytics.load_table(artifact)
                except (OSError, ValueError) as exc:
                    problem = f"could not read {artifact.name}: {exc}"
        entries.append({
            "label": sibling.dataset.name,
            "eval_label": f"{sibling_kind} #{sibling.pk}",
            "label_source": sibling.get_label_source_display(),
            "url": _url(sibling_kind, sibling),
            "current": sibling_kind == kind and sibling.pk == eval_obj.pk,
            "status": sibling.status,
            "problem": problem,
            "table": table,
        })
    breakdown = tag_analytics.test_set_breakdown(entries)
    breakdown["model"] = eval_obj.model_label()
    return breakdown
