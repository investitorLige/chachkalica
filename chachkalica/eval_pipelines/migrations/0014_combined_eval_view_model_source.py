"""Re-create ``eval_pipelines_combined_eval`` carrying what each eval scored.

The union was written when every eval was an eval *of a catalogued model*, so
``trained_model_id`` was enough to name one. An eval of an exported artifact or
of a bundle has no catalogue entry behind it, so the view now also carries
``model_source`` and the ``model_label_snapshot`` the row stamped on itself —
which is what the changelist renders instead of a join that would come back
null. A view cannot be altered column-wise in Postgres; it is dropped and
re-created.
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
    model_source,
    model_label_snapshot,
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
    model_source,
    model_label_snapshot,
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

# The shape this replaces (eval_pipelines/0006_combined_eval_view.py), so a
# reverse leaves the view working rather than merely absent.
PREVIOUS_VIEW = f"""
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
        ("eval_pipelines", "0013_pipelineevalrun_artifact_path_and_more"),
        ("training", "0039_evalrun_artifact_path_evalrun_bundle_path_and_more"),
    ]

    operations = [
        migrations.RunSQL(sql=DROP_VIEW + CREATE_VIEW,
                          reverse_sql=DROP_VIEW + PREVIOUS_VIEW),
    ]
