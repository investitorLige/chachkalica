"""Cameras admin — register an RTSP camera by URL.

Saving (add or edit) synchronously opens the stream for a moment to confirm
it's reachable; the row saves either way, the test result just sets
``status``/``last_error``. Preview/snapshot-style actions come later.
"""

import re
from django.contrib import admin, messages
from django.utils import timezone
from django.utils.html import format_html

from cameras.models import Camera
from cameras.services import rtsp

_STATUS_COLORS = {
    "online": "#22c55e",
    "unknown": "#9ca3af",
    "offline": "#ef4444",
}

_CRED_RE = re.compile(r"^(rtsps?://)([^:@/]+):([^@/]+)@")


def mask_rtsp_url(url: str) -> str:
    """Replace the password in ``rtsp://user:pass@host/...`` with dots."""
    return _CRED_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}:••••@", url or "")


def _status_badge(value: str):
    if not value:
        return "—"
    color = _STATUS_COLORS.get(value, "#9ca3af")
    return format_html('<b style="color:{};">●</b> {}', color, value)


@admin.register(Camera)
class CameraAdmin(admin.ModelAdmin):
    list_display = ["name", "masked_url", "status_badge", "last_checked_at", "created_at"]
    list_filter = ["status"]
    search_fields = ["name", "rtsp_url"]
    readonly_fields = ["status", "last_error", "last_checked_at", "created_at", "updated_at"]
    fields = ["name", "rtsp_url", "status", "last_error", "last_checked_at",
              "created_at", "updated_at"]

    @admin.display(description="rtsp url")
    def masked_url(self, obj):
        return mask_rtsp_url(obj.rtsp_url)

    @admin.display(description="status", ordering="status")
    def status_badge(self, obj):
        return _status_badge(obj.status)

    def save_model(self, request, obj, form, change):
        super().save_model(request, obj, form, change)
        ok, error = rtsp.test_connection(obj.rtsp_url)
        obj.status = Camera.ONLINE if ok else Camera.OFFLINE
        obj.last_error = error
        obj.last_checked_at = timezone.now()
        obj.save(update_fields=["status", "last_error", "last_checked_at"])
        if ok:
            self.message_user(request, f"{obj.name}: connected successfully.")
        else:
            self.message_user(
                request, f"{obj.name}: saved, but connection test failed — {error}",
                level=messages.WARNING,
            )
