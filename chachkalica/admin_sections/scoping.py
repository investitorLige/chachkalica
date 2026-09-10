"""The "what does the current admin front show" logic, factored out of
``SectionScopedAdmin.get_queryset`` (apps.py) so other code can scope itself
the same way — e.g. a model's own add-form deciding what already exists
*on this front* rather than globally (see ``fleet.admin.DatasetAdminForm``).
"""

from django.contrib.contenttypes.models import ContentType

from admin_sections.models import SectionAssignment


def scope_queryset(queryset, model, request):
    """Restrict ``queryset`` (over ``model``) to what ``request``'s admin
    front would show: unfiltered on main admin (or when ``request`` carries
    no section at all), or only rows tagged to the current section.
    """
    slug = getattr(request, "admin_section_slug", None)
    if not slug:
        return queryset
    ct = ContentType.objects.get_for_model(model)
    ids = SectionAssignment.objects.filter(
        content_type=ct, section__slug=slug
    ).values_list("object_id", flat=True)
    return queryset.filter(pk__in=list(ids))
