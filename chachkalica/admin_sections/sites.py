"""The dynamic per-section admin front.

One ``AdminSite`` instance — not one per section — reached at
``/admin/s/<slug>/...``. Unlike the rest of this project's URLs, that prefix
is *not* a captured kwarg in any urlpattern: ``SectionAdminMiddleware``
(admin_sections/middleware.py) recognizes the prefix, strips it from
``request.path_info``, and swaps in ``admin_sections.section_urlconf`` (which
mounts this site at its root) as ``request.urlconf`` for the rest of that
request — Django's own documented per-request URLconf override. That's what
lets every one of Django admin's internal ``reverse()`` calls (breadcrumbs,
redirects, the index app-list, …) keep working unmodified instead of needing
the slug threaded through dozens of call sites; the alternative — capturing
the slug directly in a shared urlpattern — breaks exactly those calls (see the
commit that replaced it).

Every ModelAdmin registered on the default ``admin.site`` gets mirrored here
(see ``AdminSectionsConfig.ready()``), so a brand-new ``AdminSection`` starts
genuinely blank: same models, same forms, just zero rows until something is
transferred in via the "Transfer to admin section…" action.
"""

from django.contrib.admin.sites import AdminSite


class SectionAdminSite(AdminSite):
    pass


section_site = SectionAdminSite(name="section_admin")
