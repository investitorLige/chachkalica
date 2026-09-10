"""The universal "Add to admin section…" / "Remove from this admin section"
actions.

Mixed into every registered ModelAdmin's ``get_actions`` by
``AdminSectionsConfig.ready()`` (see apps.py) — not declared as a method on
any one ModelAdmin, since they need to work identically for all of them.

Membership, not a move: adding a row to a section does not touch any other
section it's already in (main admin included — main is unfiltered and never
tracked here at all). "Remove from this admin section" only detaches the one
section currently being viewed.
"""

from django.contrib import messages
from django.contrib.admin.helpers import ACTION_CHECKBOX_NAME
from django.contrib.contenttypes.models import ContentType
from django.template.response import TemplateResponse

from admin_sections.models import AdminSection, SectionAssignment

ADD_ACTION_NAME = "add_to_section"
REMOVE_ACTION_NAME = "remove_from_this_section"


def add_to_section(modeladmin, request, queryset):
    """Add the selected rows to another admin section, in addition to
    wherever they already show up.

    Same "intermediate confirmation page" shape as ``fleet``'s
    ``merge_selected``: the first pass (no ``apply`` in POST) renders a page
    to pick the destination; submitting that page re-posts the same
    selection with ``apply`` set, which is when the rows actually get added.
    """
    opts = modeladmin.model._meta

    if request.POST.get("apply"):
        section = AdminSection.objects.filter(pk=request.POST.get("section")).first()
        if section is None:
            modeladmin.message_user(request, "Pick a destination admin section.", level=messages.WARNING)
            return None

        ct = ContentType.objects.get_for_model(modeladmin.model)
        for obj in queryset:
            SectionAssignment.objects.get_or_create(content_type=ct, object_id=str(obj.pk), section=section)

        modeladmin.message_user(request, f"Added {queryset.count()} {opts.verbose_name_plural} to “{section.name}”.")
        return None

    context = {
        **modeladmin.admin_site.each_context(request),
        "title": "Add to admin section",
        "opts": opts,
        "objects": queryset,
        "sections": AdminSection.objects.all(),
        "action": ADD_ACTION_NAME,
        "selected": [str(obj.pk) for obj in queryset],
        "action_checkbox_name": ACTION_CHECKBOX_NAME,
    }
    return TemplateResponse(request, "admin/admin_sections/add_to_section_confirmation.html", context)


add_to_section.short_description = "Add to admin section…"


def remove_from_this_section(modeladmin, request, queryset):
    """Detach the selected rows from *this* section only — the one currently
    being viewed. Only offered while viewing a section (see apps.py); doesn't
    touch any other section the rows are also in.
    """
    opts = modeladmin.model._meta
    slug = getattr(request, "admin_section_slug", None)
    if not slug:
        modeladmin.message_user(request, "Not viewing a specific admin section.", level=messages.WARNING)
        return None

    ct = ContentType.objects.get_for_model(modeladmin.model)
    pks = [str(obj.pk) for obj in queryset]
    deleted, _ = SectionAssignment.objects.filter(
        content_type=ct, object_id__in=pks, section__slug=slug
    ).delete()

    modeladmin.message_user(request, f"Removed {len(pks)} {opts.verbose_name_plural} from this admin section.")
    return None


remove_from_this_section.short_description = "Remove from this admin section"
