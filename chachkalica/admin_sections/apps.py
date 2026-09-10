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

        from admin_sections.actions import (
            ADD_ACTION_NAME,
            REMOVE_ACTION_NAME,
            add_to_section,
            remove_from_this_section,
        )
        from admin_sections.models import AdminSection, SectionAssignment
        from admin_sections.scoping import scope_queryset
        from admin_sections.sites import section_site

        # A section can't sensibly contain itself — these two are mirrored
        # below unscoped (no filtering, no add/remove actions), never with
        # SectionScopedAdmin mixed in.
        unscoped = {AdminSection, SectionAssignment}

        class SectionScopedAdmin:
            """Mixed into every registered ModelAdmin's class below: filters
            each changelist down to the current admin section (if any), adds
            the universal add/remove-from-section actions, and auto-tags
            anything newly added from within a section into that section (or
            it would vanish from that same changelist right after creation)."""

            def get_queryset(self, request):
                return scope_queryset(super().get_queryset(request), self.model, request)

            def save_model(self, request, obj, form, change):
                super().save_model(request, obj, form, change)
                slug = getattr(request, "admin_section_slug", None)
                if slug and not change:
                    section = AdminSection.objects.filter(slug=slug).first()
                    if section:
                        ct = ContentType.objects.get_for_model(self.model)
                        SectionAssignment.objects.get_or_create(
                            content_type=ct, object_id=str(obj.pk), section=section
                        )

            def get_actions(self, request):
                actions = super().get_actions(request)
                actions[ADD_ACTION_NAME] = (
                    add_to_section, ADD_ACTION_NAME, add_to_section.short_description,
                )
                # Only offered while viewing a section — "remove from *this*
                # section" is meaningless on the (always-unfiltered) main admin.
                if getattr(request, "admin_section_slug", None):
                    actions[REMOVE_ACTION_NAME] = (
                        remove_from_this_section, REMOVE_ACTION_NAME, remove_from_this_section.short_description,
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
            if model in unscoped:
                # Mirrored as-is (no filtering, no actions) so this app_label
                # still resolves under section_urlconf too — third-party code
                # that calls the global admin.site directly regardless of which
                # site actually served the page (django-rq's dashboard does
                # exactly this in its own each_context() call) needs every app
                # in admin.site's registry to also exist here, or its own
                # reverse() calls 500 while a section is active.
                section_site.register(model, model_admin.__class__)
                continue
            scoped_cls = scoped_class_for(model_admin.__class__)
            model_admin.__class__ = scoped_cls
            # Mirror the (now scoped) registration onto the section front, so a
            # brand-new section starts with every model available but zero rows
            # — genuinely blank until something is transferred in.
            section_site.register(model, scoped_cls)
