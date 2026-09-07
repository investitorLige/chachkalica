from django.apps import AppConfig


class AdminSectionsConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "admin_sections"
    verbose_name = "Admin Sections"

    def ready(self):
        # Every other app's admin.py has already run its admin.site.register()
        # calls by the time ANY app's ready() fires — django.contrib.admin's own
        # ready() (earlier in INSTALLED_APPS) is what calls autodiscover(), and
        # that imports every app's admin.py synchronously, this app's included.
        # So the registry below is already complete regardless of where
        # "admin_sections" sits among the custom apps in INSTALLED_APPS.
        from django.contrib import admin
        from django.contrib.contenttypes.models import ContentType

        from admin_sections.actions import TRANSFER_ACTION_NAME, transfer_to_section
        from admin_sections.models import AdminSection, SectionAssignment
        from admin_sections.sites import section_site

        # A section can't sensibly contain itself.
        excluded = {AdminSection, SectionAssignment}

        class SectionScopedAdmin:
            """Mixed into every registered ModelAdmin's class below: filters
            each changelist down to the current admin section (if any) and
            adds the universal "Transfer to admin section…" action."""

            def get_queryset(self, request):
                qs = super().get_queryset(request)
                slug = getattr(request, "admin_section_slug", None)
                if not slug:
                    return qs  # main /admin/ — unfiltered, unchanged behaviour
                ct = ContentType.objects.get_for_model(self.model)
                ids = SectionAssignment.objects.filter(
                    content_type=ct, section__slug=slug
                ).values_list("object_id", flat=True)
                return qs.filter(pk__in=list(ids))

            def get_actions(self, request):
                actions = super().get_actions(request)
                actions[TRANSFER_ACTION_NAME] = (
                    transfer_to_section,
                    TRANSFER_ACTION_NAME,
                    transfer_to_section.short_description,
                )
                return actions

        # One dynamic subclass per distinct ModelAdmin class, not per model —
        # several apps register the same ModelAdmin class against more than one
        # model (e.g. eval_pipelines' PipelineEvalRunAdmin), and they can share it.
        scoped_cache = {}

        def scoped_class_for(base_cls):
            if base_cls not in scoped_cache:
                scoped_cache[base_cls] = type(
                    base_cls.__name__, (SectionScopedAdmin, base_cls), {}
                )
            return scoped_cache[base_cls]

        for model, model_admin in list(admin.site._registry.items()):
            if model in excluded:
                continue
            scoped_cls = scoped_class_for(model_admin.__class__)
            model_admin.__class__ = scoped_cls
            # Mirror the (now scoped) registration onto the section front, so a
            # brand-new section starts with every model available but zero rows
            # — genuinely blank until something is transferred in.
            section_site.register(model, scoped_cls)
