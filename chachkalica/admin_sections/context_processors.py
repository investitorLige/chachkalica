"""Feeds the admin-section nav (list + current) to templates/admin/base_site.html.

Unconditional — it's one small table, and every other template simply ignores
these variables.
"""

from admin_sections.models import AdminSection


def admin_sections_nav(request):
    sections = list(AdminSection.objects.all())
    slug = getattr(request, "admin_section_slug", None)
    current = next((s for s in sections if s.slug == slug), None) if slug else None
    return {
        "nav_admin_sections": sections,
        "nav_current_section": current,
    }
