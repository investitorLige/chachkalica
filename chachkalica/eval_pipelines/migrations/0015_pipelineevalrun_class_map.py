"""Add ``class_map`` to PipelineEvalRun.

An eval scores predictions against ground truth by class *name*, dropping
whatever the two class spaces don't share — so a model and a dataset with
different taxonomies score nothing at all, silently. ``class_map`` translates
the dataset's names into the model's before scoring (and drops the classes the
model cannot predict), which is what makes such a pair scorable. Empty for
every existing row, which is exactly the behaviour those rows already had.

See ``training.services.class_sync`` for the form half and
``friendy_chachkalica.metrics.apply_class_map`` for what the trainer does with it.
"""

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("eval_pipelines", "0014_combined_eval_view_model_source"),
    ]

    operations = [
        migrations.AddField(
            model_name="pipelineevalrun",
            name="class_map",
            field=models.JSONField(
                blank=True,
                default=dict,
                help_text='Optional {dataset class name: model class name} translation applied to the ground truth before scoring. A name mapped to null/"" is dropped from the eval entirely. Empty (the default) scores the dataset\'s classes exactly as they are — which only produces a number when the model shares their names.',
            ),
        ),
    ]
