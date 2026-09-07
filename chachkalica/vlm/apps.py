from django.apps import AppConfig


class VlmConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "vlm"
    # This string is the admin section header — there is no custom AdminSite in
    # this project, so a new "tab group" is just a new app with a verbose_name.
    verbose_name = "VLM"

    def ready(self):
        # Register post_delete handlers that clean up on-disk video files.
        from vlm import signals  # noqa: F401
