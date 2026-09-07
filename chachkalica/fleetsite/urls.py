"""Root URL configuration for fleetsite.

- the Django admin is the operator UI (provision / setup-dataset / sync / remove)
- `django_rq` exposes the queue + failed-job dashboards under the admin
- the fleet app contributes the Label Studio webhook receiver at /hook
"""

from django.contrib import admin
from django.urls import include, path

from fleetsite.admin_views import benchmark_console_view, bundle_sync_view

admin.site.site_header = "Chachkalica Fleet"
admin.site.site_title = "Chachkalica Fleet"
admin.site.index_title = "Label Studio annotator fleet"

urlpatterns = [
    # Standalone admin page (must precede admin.site.urls so it isn't shadowed);
    # admin_view enforces the same staff-only auth as the rest of the admin.
    path(
        "admin/benchmarks/",
        admin.site.admin_view(benchmark_console_view),
        name="benchmark-console",
    ),
    # JSON endpoint shared by the "Sync bundle" button on every inference form;
    # same admin_view auth, same must-precede-admin.site.urls rule as above.
    path(
        "admin/bundles/sync/",
        admin.site.admin_view(bundle_sync_view),
        name="bundle-sync",
    ),
    # Every project's own blank admin front — /admin/s/<slug>/, one row in
    # AdminSection per front, no code change or redeploy to add one — is NOT
    # routed here. SectionAdminMiddleware intercepts it earlier (swapping in
    # admin_sections.section_urlconf) before URL resolution ever sees this
    # urlconf; see that middleware's docstring for why a urlpattern with a
    # captured slug doesn't work.
    path("admin/", admin.site.urls),
    path("django-rq/", include("django_rq.urls")),
    path("", include("fleet.urls")),
]
