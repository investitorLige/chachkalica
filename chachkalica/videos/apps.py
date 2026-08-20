from django.apps import AppConfig


class VideosConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "videos"
    verbose_name = "Videos"

    def ready(self):
        # Register post_delete handlers that clean up on-disk video artifacts.
        from videos import signals  # noqa: F401
