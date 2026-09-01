from django.apps import AppConfig


class BenchmarksConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "benchmarks"
    # The admin index section header. "Bundle" is load-bearing: the site already has
    # a Benchmarks console (fleetsite/admin_views.py) that measures architecture
    # variants from random-init weights on synthetic tensors. This section measures
    # exported bundles on real frames -- a different question with different numbers,
    # and conflating the two would be worse than a clumsy name.
    verbose_name = "Bundle Benchmarks"
