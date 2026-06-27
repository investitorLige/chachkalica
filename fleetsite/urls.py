"""Root URL configuration for fleetsite.

- the Django admin is the operator UI (provision / setup-dataset / sync / remove)
- `django_rq` exposes the queue + failed-job dashboards under the admin
- the fleet app contributes the Label Studio webhook receiver at /hook
"""

from django.contrib import admin
from django.urls import include, path

admin.site.site_header = "Mercury Fleet"
admin.site.site_title = "Mercury Fleet"
admin.site.index_title = "Label Studio annotator fleet"

urlpatterns = [
    path("admin/", admin.site.urls),
    path("django-rq/", include("django_rq.urls")),
    path("", include("fleet.urls")),
]
