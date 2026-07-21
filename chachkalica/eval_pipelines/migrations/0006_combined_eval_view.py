"""Create the read-only ``eval_pipelines_combined_eval`` view.

Unions ``eval_pipelines_pipelineevalrun`` (chachak pipeline evals) and
``training_evalrun`` (base / raw-model evals) into one PipelineEvalRun-shaped
result so the :class:`~eval_pipelines.models.CombinedEval` proxy can list both in
the single "All pipeline evals" changelist. Base rows report ``pipeline='base'``;
each row keeps a text ``id`` (``"pe-<pk>"`` / ``"be-<pk>"``) so the two integer
pk spaces never collide, plus ``kind`` + ``orig_id`` to route back to the source
row. The model is ``managed=False``, so only this view (not a table) is created.
"""

from django.db import migrations

VIEW_NAME = "eval_pipelines_combined_eval"

CREATE_VIEW = f"""
CREATE VIEW {VIEW_NAME} AS
SELECT
    'pe-' || id::text        AS id,
    id                       AS orig_id,
    'pipeline'               AS kind,
    trained_model_id,
    dataset_id,
    pipeline,
    status,
    metrics,
    score_threshold,
    output_dir,
    request_yaml_path,
    created_at,
    finished_at
FROM eval_pipelines_pipelineevalrun
UNION ALL
SELECT
    'be-' || id::text        AS id,
    id                       AS orig_id,
    'base'                   AS kind,
    trained_model_id,
    dataset_id,
    'base'                   AS pipeline,
    status,
    metrics,
    score_threshold,
    output_dir,
    request_yaml_path,
    created_at,
    finished_at
FROM training_evalrun;
"""

DROP_VIEW = f"DROP VIEW IF EXISTS {VIEW_NAME};"


class Migration(migrations.Migration):

    dependencies = [
        ("eval_pipelines", "0005_pipelineevalrun_score_threshold"),
        ("training", "0017_experiment_tile_size_px"),
    ]

    operations = [
        migrations.RunSQL(sql=CREATE_VIEW, reverse_sql=DROP_VIEW),
    ]
