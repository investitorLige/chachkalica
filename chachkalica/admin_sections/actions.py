"""The universal "Transfer to admin section…" action.

Mixed into every registered ModelAdmin's ``get_actions`` by
``AdminSectionsConfig.ready()`` (see apps.py) — not declared as a method on
any one ModelAdmin, since it needs to work identically for all of them.
"""

from django.contrib.admin.helpers import ACTION_CHECKBOX_NAME
from django.contrib.contenttypes.models import ContentType
from django.template.response import TemplateResponse

from admin_sections.models import AdminSection, SectionAssignment

TRANSFER_ACTION_NAME = "transfer_to_section"


def transfer_to_section(modeladmin, request, queryset):
    """Move the selected rows to another admin section (or back to Main admin).

    Same "intermediate confirmation page" shape as ``fleet``'s ``merge_selected``:
    the first pass (no ``apply`` in POST) renders a page to pick the
    destination; submitting that page re-posts the same selection with
    ``apply`` set, which is when the reassignment actually happens.
    """
    opts = modeladmin.model._meta

    if request.POST.get("apply"):
        section_id = request.POST.get("section") or None
        section = AdminSection.objects.filter(pk=section_id).first() if section_id else None
        ct = ContentType.objects.get_for_model(modeladmin.model)
        pks = [str(obj.pk) for obj in queryset]

        if section is None:
            SectionAssignment.objects.filter(content_type=ct, object_id__in=pks).delete()
        else:
            for pk in pks:
                SectionAssignment.objects.update_or_create(
                    content_type=ct, object_id=pk, defaults={"section": section},
                )

        label = section.name if section else "Main admin"
        modeladmin.message_user(request, f"Transferred {len(pks)} {opts.verbose_name_plural} to “{label}”.")
        return None

    context = {
        **modeladmin.admin_site.each_context(request),
        "title": "Transfer to admin section",
        "opts": opts,
        "objects": queryset,
        "sections": AdminSection.objects.all(),
        "action": TRANSFER_ACTION_NAME,
        "selected": [str(obj.pk) for obj in queryset],
        "action_checkbox_name": ACTION_CHECKBOX_NAME,
    }
    return TemplateResponse(request, "admin/admin_sections/transfer_confirmation.html", context)


transfer_to_section.short_description = "Transfer to admin section…"
