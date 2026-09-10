from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('videos', '0010_renderpreset'),
    ]

    operations = [
        migrations.CreateModel(
            name='MarketingVideo',
            fields=[
            ],
            options={
                'verbose_name': 'Marketing video',
                'verbose_name_plural': 'Marketing videos',
                'proxy': True,
                'indexes': [],
                'constraints': [],
            },
            bases=('videos.inferencejob',),
        ),
    ]
