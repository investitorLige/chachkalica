"""Admin for AdminSection itself — the "list of admins" + "make a new one" UI.

Lives on the main /admin/ only (excluded from the section-mirroring in
apps.py, since a section can't sensibly contain itself). Creating one here
*is* "making another blank admin under a different URL": response_add below
sends you straight to the new section's own front instead of back to this
changelist.
"""

from django.contrib import admin
from django.http import HttpResponseRedirect
from django.utils.html import format_html

from admin_sections.models import AdminSection


@admin.register(AdminSection)
class AdminSectionAdmin(admin.ModelAdmin):
    list_display = ("name", "slug", "row_count", "created_at", "open_link")
    prepopulated_fields = {"slug": ("name",)}
    readonly_fields = ("created_at",)

    @admin.display(description="Rows assigned")
    def row_count(self, obj):
        return obj.assignments.count()

    @admin.display(description="")
    def open_link(self, obj):
        return format_html('<a href="/admin/s/{}/">Open →</a>', obj.slug)

    def response_add(self, request, obj, post_url_continue=None):
        if "_addanother" in request.POST or "_continue" in request.POST:
            return super().response_add(request, obj, post_url_continue)
        self.message_user(request, f"Created admin section “{obj.name}”.")
        return HttpResponseRedirect(f"/admin/s/{obj.slug}/")
