from django.apps import AppConfig


class MarketingStudioConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "marketing_studio"
    # This string is the admin section header — there is no custom AdminSite in
    # this project, so a new "tab group" is just a new app with a verbose_name
    # (see vlm/apps.py). Sections sort alphabetically by it.
    verbose_name = "Marketing Studio"

    def ready(self):
        # Register post_delete handlers that clean up on-disk video artifacts.
        from marketing_studio import signals  # noqa: F401
