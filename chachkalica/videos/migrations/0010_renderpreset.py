from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('videos', '0009_inferencejob_render_style'),
    ]

    operations = [
        migrations.CreateModel(
            name='RenderPreset',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True,
                                           serialize=False, verbose_name='ID')),
                ('name', models.CharField(help_text='What this look is called in the preset dropdown.', max_length=120, unique=True)),
                ('style', models.JSONField(default=dict, help_text='The style dict (see videos.services.render_style).')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
            ],
            options={
                'verbose_name': 'Render preset',
                'verbose_name_plural': 'Render presets',
                'ordering': ['name'],
            },
        ),
    ]
