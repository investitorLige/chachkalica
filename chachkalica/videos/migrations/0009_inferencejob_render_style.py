from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('videos', '0008_alter_inferencejob_detector_min_box_size'),
    ]

    operations = [
        migrations.AddField(
            model_name='inferencejob',
            name='render_style',
            field=models.JSONField(blank=True, default=dict, help_text='Look of the burned-in overlay, as written by the "Run model inference for marketing…" action (see videos.services.render_style). Empty = the plain overlay the ordinary inference action draws.'),
        ),
    ]
