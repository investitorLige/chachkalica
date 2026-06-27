from django.apps import AppConfig


class FleetConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'fleet'

    def ready(self):
        # Register signal receivers (dataset teardown, etc.).
        from fleet import signals  # noqa: F401
