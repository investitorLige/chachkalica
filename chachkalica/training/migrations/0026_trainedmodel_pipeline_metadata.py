"""Freeze each model's training pipeline onto the model itself.

Adds ``TrainedModel.pipeline_metadata`` and backfills it for models already in
the catalogue by walking back to the experiment they were promoted from — the
same walk every consumer used to do live, done once here so the record survives
the experiment being retuned or deleted.

The backfill reuses :func:`training.services.pipeline_meta.from_experiment`
rather than restating the field mapping: the function only reads concrete fields
that exist on the historical models, and a second copy of the mapping in a
migration is exactly the drift this change exists to remove.
"""

from django.db import migrations, models

from training.services import pipeline_meta


def freeze_existing(apps, schema_editor):
    TrainedModel = apps.get_model("training", "TrainedModel")
    rows = TrainedModel.objects.select_related(
        "source_run_result__run__experiment"
    ).all()

    updated = []
    for model in rows:
        if model.pipeline_metadata:
            continue
        model.pipeline_metadata = pipeline_meta.from_experiment(
            pipeline_meta.source_experiment(model)
        )
        updated.append(model)
    if updated:
        TrainedModel.objects.bulk_update(updated, ["pipeline_metadata"])


def clear(apps, schema_editor):
    """Reverse of the backfill: the column is dropped by the schema operation, so
    there is nothing to undo — declared only to keep the migration reversible."""


class Migration(migrations.Migration):

    dependencies = [
        ("training", "0025_trainingsettings_exports_root_and_more"),
    ]

    operations = [
        migrations.AddField(
            model_name="trainedmodel",
            name="pipeline_metadata",
            field=models.JSONField(
                blank=True,
                default=dict,
                help_text="Frozen record of the chachak pipeline this model was "
                          "trained through — the geometry it must be served with "
                          "(see training.services.pipeline_meta). Written at "
                          "promotion time and copied beside every ONNX/TensorRT "
                          "export, so every action that runs this model prefills "
                          "itself from one record instead of asking for the "
                          "parameters again.",
            ),
        ),
        migrations.RunPython(freeze_existing, clear),
    ]
