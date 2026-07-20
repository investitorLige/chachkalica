from django.core.validators import MinValueValidator
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("training", "0016_alter_trainingrun_status"),
    ]

    operations = [
        migrations.AddField(
            model_name="experiment",
            name="tile_size_px",
            field=models.PositiveIntegerField(
                blank=True,
                help_text=(
                    "Fixed square source-image tile size used by batch_detect. Set this "
                    "to the 'trained @' resolution shown beside the selected pretrained "
                    "weights when you want to preserve that model's native pixel scale. "
                    "For example, 560 produces 560×560 tiles. When set, this overrides "
                    "Tile width % and Tile height %. Full-size windows are shifted flush "
                    "to the right/bottom edge where possible; only an image smaller than "
                    "this size is zero-padded on the right/bottom, with no stretching or "
                    "upscaling. The saved value is used consistently for training, "
                    "validation, and pipeline inference. For RF-DETR, keep the model "
                    "Input resolution equal to this value. If an experiment contains "
                    "models with different native resolutions, choose one common tile "
                    "size deliberately or use separate experiments. Blank keeps the "
                    "legacy percentage-based tiling behavior."
                ),
                null=True,
                validators=[MinValueValidator(1)],
                verbose_name="Tile size (pixels)",
            ),
        ),
    ]
