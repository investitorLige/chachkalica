"""Routes ``/admin/s/<slug>/...`` to the section admin front without a static
urlpattern per section (that's the whole point — a new ``AdminSection`` row
is reachable with no urls.py edit or redeploy).

Mounting ``section_site`` under a *captured* URL kwarg instead (e.g.
``path("admin/s/<slug:section_slug>/", section_site.urls)``) was the first
approach tried here, and it breaks: Django admin's own internal ``reverse()``
calls (the index app-list, breadcrumbs, every changeform/changelist redirect)
don't know to supply that extra kwarg, so they all raise ``NoReverseMatch``.
This middleware instead uses Django's own supported "URLconf per request"
mechanism (``request.urlconf`` — see ``BaseHandler.resolve_request``) plus
``set_script_prefix`` (the same pair Django itself uses for an app mounted
under a WSGI ``SCRIPT_NAME``), so every ``reverse()`` call made against
``admin_sections.section_urlconf`` resolves — and builds links — correctly,
with zero changes to Django admin's own code.
"""

import re

from django.urls import set_script_prefix

_SECTION_PREFIX_RE = re.compile(r"^/admin/s/(?P<slug>[-a-zA-Z0-9_]+)/")


class SectionAdminMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        match = _SECTION_PREFIX_RE.match(request.path_info)
        if not match:
            return self.get_response(request)

        prefix = match.group(0)
        request.admin_section_slug = match.group("slug")
        request.path_info = "/" + request.path_info[len(prefix):]
        request.urlconf = "admin_sections.section_urlconf"

        # set_script_prefix is a thread-local, not per-request state — Django's
        # real WSGI handler resets it before every call (get_script_name(environ)
        # is always "" here), but nothing resets it *between* two calls that
        # don't both go through that handler: Django's own test Client doesn't,
        # and neither does a bare reverse() call in test code run right after a
        # section request (both bit this exactly). try/finally scopes the
        # prefix to just this one request, however it exits, instead of
        # trusting whatever runs next to clean up after it.
        set_script_prefix(prefix)
        try:
            return self.get_response(request)
        finally:
            set_script_prefix("/")
