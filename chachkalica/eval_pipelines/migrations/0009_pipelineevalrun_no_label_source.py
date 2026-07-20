from django.db import migrations, models


LABEL_CHOICES = [
    ("source", "source labels"),
    ("annotator", "annotator output"),
    ("explicit", "explicit path"),
    ("none", "no label source (prediction only)"),
]


class Migration(migrations.Migration):
    dependencies = [("eval_pipelines", "0008_pipelineevalrun_combined_models")]

    operations = [
        migrations.AlterField(
            model_name="pipelineevalrun",
            name="label_source",
            field=models.CharField(choices=LABEL_CHOICES, default="source", max_length=16),
        ),
    ]
