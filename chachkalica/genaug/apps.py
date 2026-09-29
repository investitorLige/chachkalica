from django.apps import AppConfig


class GenaugConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "genaug"
    # The admin section header — a new tab group is a new app with a verbose_name.
    verbose_name = "Generative augmentation"
