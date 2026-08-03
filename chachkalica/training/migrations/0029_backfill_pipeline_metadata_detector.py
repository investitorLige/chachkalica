"""Fill in the detector a person-crop model was actually trained with.

``pipeline_meta.from_experiment`` used to freeze ``Experiment.detector_checkpoint``
verbatim, so an experiment that never overrode the default (the field only gained
``default=DEFAULT_PERSON_DETECTOR_CHECKPOINT`` later) left a blank on the record —
even though ``config_gen.pipeline_block`` had substituted the default when it
wrote the training YAML. Models promoted from those experiments therefore
describe a pipeline they were never trained through, their admin prefills arrive
with an empty detector field, and ``config_gen.build_predict_request`` refuses to
launch video/camera inference for them at all.

Patches the one key in place rather than re-deriving the blob from the source
experiment: a model whose run has since been deleted must keep the record it
froze, and only this field was ever wrong.
"""

from django.db import migrations

from training import pipelines

DEFAULT_PERSON_DETECTOR_CHECKPOINT = "models/people/best_ckpt.engine"


def fill_detector(apps, schema_editor):
    TrainedModel = apps.get_model("training", "TrainedModel")

    updated = []
    for model in TrainedModel.objects.all():
        metadata = model.pipeline_metadata
        if not isinstance(metadata, dict) or not metadata:
            continue
        if metadata.get("detector_checkpoint"):
            continue
        if not pipelines.needs_detector(
            metadata.get("pipeline") or "", metadata.get("chain") or []
        ):
            continue
        metadata["detector_checkpoint"] = DEFAULT_PERSON_DETECTOR_CHECKPOINT
        model.pipeline_metadata = metadata
        updated.append(model)
    if updated:
        TrainedModel.objects.bulk_update(updated, ["pipeline_metadata"])


def blank_detector(apps, schema_editor):
    """Reverse: put the blank back, so the migration is reversible.

    Only touches rows still carrying exactly the default path — an operator who
    has since pointed a model at its own detector keeps that.
    """
    TrainedModel = apps.get_model("training", "TrainedModel")

    updated = []
    for model in TrainedModel.objects.all():
        metadata = model.pipeline_metadata
        if not isinstance(metadata, dict) or not metadata:
            continue
        if metadata.get("detector_checkpoint") != DEFAULT_PERSON_DETECTOR_CHECKPOINT:
            continue
        metadata["detector_checkpoint"] = ""
        model.pipeline_metadata = metadata
        updated.append(model)
    if updated:
        TrainedModel.objects.bulk_update(updated, ["pipeline_metadata"])


class Migration(migrations.Migration):

    dependencies = [
        ("training", "0028_trainingsettings_bundles_root"),
    ]

    operations = [
        migrations.RunPython(fill_detector, blank_detector),
    ]
