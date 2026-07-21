from django.apps import AppConfig


class EvalPipelinesConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "eval_pipelines"
    verbose_name = "Eval Pipelines"

    def ready(self):
        # Register the post_delete handler that cleans up on-disk pipeline
        # eval artifacts (output dir + generated request YAML).
        from eval_pipelines import signals  # noqa: F401
