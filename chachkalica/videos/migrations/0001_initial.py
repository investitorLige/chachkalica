from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name='Video',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(help_text='Display name (defaults to the file name without extension).', max_length=255, unique=True)),
                ('filename', models.CharField(blank=True, help_text='File name under the videos root. Filled in by the download job.', max_length=512)),
                ('source_url', models.URLField(blank=True, help_text='Link the video was downloaded from (empty for imported files).', max_length=1024)),
                ('quality', models.CharField(blank=True, help_text='Requested max height for downloads (e.g. 1080), or blank for best.', max_length=16)),
                ('status', models.CharField(choices=[('pending', 'pending'), ('downloading', 'downloading'), ('ready', 'ready'), ('error', 'error')], default='ready', max_length=16)),
                ('last_error', models.TextField(blank=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
            ],
            options={
                'ordering': ['-created_at', 'name'],
            },
        ),
    ]
