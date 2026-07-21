from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('fleet', '0003_dataset_has_labels'),
    ]

    operations = [
        migrations.AddField(
            model_name='fleetsettings',
            name='videos_dir',
            field=models.CharField(default='data/videos', help_text='Directory holding raw video files (imported or downloaded).', max_length=512),
        ),
    ]
