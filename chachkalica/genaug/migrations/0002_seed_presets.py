from django.db import migrations


def seed(apps, schema_editor):
    from genaug.presets import seed_presets

    seed_presets(apps.get_model("genaug", "AugPrompt"))


class Migration(migrations.Migration):
    dependencies = [("genaug", "0001_initial")]

    operations = [migrations.RunPython(seed, migrations.RunPython.noop)]
