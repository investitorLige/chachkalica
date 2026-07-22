import django.core.validators
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('training', '0022_experiment_default_person_detector'),
    ]

    operations = [
        migrations.AddField(
            model_name='experiment',
            name='detector_min_box_size',
            field=models.FloatField(default=224.0, help_text="Drop a detected person box if its crop (after expand ratio) is narrower or shorter than this many pixels, so degenerate tiny person detections never become a training or eval crop. 0 disables the floor. Only applied for people_detect_first — batch_people already starts from fixed-size tiles so its person crops don't shrink to the same degenerate sizes. Introduced after run PPE_v0.4_ppl_first-29's RT-DETR run crashed on epoch 1 batch 1 with 'selected index k out of range': RT-DETR's encoder does topk(num_queries) over its last feature map, which has (padded_size / 32)^2 tokens, so a crop must be large enough to out-token whatever num_queries the model uses. The default 224 gives >= (224/32)^2 = 49 tokens (a conservative square-crop floor; real person crops are usually taller than wide, so the true count is typically higher) — comfortably above the 25-query default this app now injects for rtdetr models on this pipeline (see config_gen.model_entry). Raise both together if you deliberately increase num_queries in a model's params.", validators=[django.core.validators.MinValueValidator(0.0)], verbose_name='Person-crop minimum size (px)'),
        ),
    ]
