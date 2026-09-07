"""The URLconf ``SectionAdminMiddleware`` swaps in for any ``/admin/s/<slug>/``
request. ``section_site`` is mounted at this module's own root ("") — the
slug prefix has already been stripped from ``request.path_info`` by the time
Django resolves against this — so nothing here needs to know about it.
"""

from django.urls import path

from admin_sections.sites import section_site
from fleetsite.admin_views import bundle_sync_view

urlpatterns = [
    # A handful of ModelAdmins (videos, cameras) call reverse("bundle-sync")
    # directly rather than through admin_site.name — since they're mirrored
    # onto section_site like everything else, that name needs to resolve here
    # too. Must precede the catch-all site.urls below for the same
    # prefix-shadowing reason fleetsite/urls.py documents for the main site.
    path("bundles/sync/", section_site.admin_view(bundle_sync_view), name="bundle-sync"),
    path("", section_site.urls),
]
