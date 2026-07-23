"""Cameras admin — register an RTSP camera by URL, watch it live.

Saving (add or edit) synchronously opens the stream for a moment to confirm
it's reachable; the row saves either way, the test result just sets
``status``/``last_error``. A "View live stream…" action (and an inline
preview on the change page) opens an MJPEG view backed by Redis — see
``cameras/management/commands/stream_cameras.py`` for the worker that keeps
that cache warm, and ``docs/camera-live-preview.md`` for the full design.
"""

import re
import time

from django.contrib import admin, messages
from django.http import Http404, StreamingHttpResponse
from django.shortcuts import redirect
from django.template.response import TemplateResponse
from django.urls import path, reverse
from django.utils import timezone
from django.utils.html import format_html

from cameras.models import Camera
from cameras.services import frame_cache, rtsp

_STATUS_COLORS = {
    "online": "#22c55e",
    "unknown": "#9ca3af",
    "offline": "#ef4444",
}

_CRED_RE = re.compile(r"^(rtsps?://)([^:@/]+):([^@/]+)@")

# Client polls Redis at this interval; the worker pushes faster than this so
# there's always a fresh frame waiting (see FRAME_PUSH_INTERVAL_SECONDS in
# stream_cameras.py).
_MJPEG_POLL_SECONDS = 0.1


def mask_rtsp_url(url: str) -> str:
    """Replace the password in ``rtsp://user:pass@host/...`` with dots."""
    return _CRED_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}:••••@", url or "")


def _status_badge(value: str):
    if not value:
        return "—"
    color = _STATUS_COLORS.get(value, "#9ca3af")
    return format_html('<b style="color:{};">●</b> {}', color, value)


def _mjpeg_frames(camera_id):
    while True:
        frame = frame_cache.get_frame(camera_id)
        if frame is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
        time.sleep(_MJPEG_POLL_SECONDS)


@admin.register(Camera)
class CameraAdmin(admin.ModelAdmin):
    list_display = ["name", "masked_url", "status_badge", "last_checked_at", "created_at"]
    list_filter = ["status"]
    search_fields = ["name", "rtsp_url"]
    readonly_fields = ["live_preview", "status", "last_error", "last_checked_at",
                       "created_at", "updated_at"]
    fields = ["name", "rtsp_url", "live_preview", "status", "last_error",
              "last_checked_at", "created_at", "updated_at"]
    actions = ["view_live_stream"]

    @admin.display(description="rtsp url")
    def masked_url(self, obj):
        return mask_rtsp_url(obj.rtsp_url)

    @admin.display(description="status", ordering="status")
    def status_badge(self, obj):
        return _status_badge(obj.status)

    @admin.display(description="Live preview")
    def live_preview(self, obj):
        if obj is None or not obj.pk:
            return "Save the camera first to preview it."
        mjpeg_url = reverse("admin:cameras_camera_mjpeg") + f"?camera={obj.pk}"
        live_url = reverse("admin:cameras_camera_live") + f"?camera={obj.pk}"
        return format_html(
            '<img src="{}" style="max-width:480px;width:100%;border-radius:6px;'
            'background:#000" />'
            '<p style="margin-top:6px"><a class="button" href="{}">Open full live view</a></p>',
            mjpeg_url, live_url,
        )

    def save_model(self, request, obj, form, change):
        # Only re-test when the URL (host/creds/path) actually changed. A save
        # is one live login attempt against the camera — re-testing on every
        # unrelated edit (or a repeated click of Save) burns extra failed
        # attempts for nothing, which is exactly what can trip an NVR's
        # brute-force lockout.
        url_changed = not change or "rtsp_url" in form.changed_data
        super().save_model(request, obj, form, change)

        if not url_changed:
            self.message_user(request, f"{obj.name}: saved (URL unchanged, skipped re-test).")
            return

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

    # ---------------------------------------------------------------- actions
    @admin.action(description="View live stream…")
    def view_live_stream(self, request, queryset):
        if queryset.count() != 1:
            self.message_user(request, "Select exactly one camera to view its live stream.",
                              level=messages.WARNING)
            return None
        camera = queryset.first()
        return redirect(reverse("admin:cameras_camera_live") + f"?camera={camera.pk}")

    # ------------------------------------------------------------------- urls
    def get_urls(self):
        custom = [
            path("mjpeg/", self.admin_site.admin_view(self.mjpeg_view),
                 name="cameras_camera_mjpeg"),
            path("live/", self.admin_site.admin_view(self.live_view),
                 name="cameras_camera_live"),
        ]
        return custom + super().get_urls()

    def mjpeg_view(self, request):
        camera = Camera.objects.filter(pk=request.GET.get("camera")).first()
        if camera is None:
            raise Http404("unknown camera")
        return StreamingHttpResponse(
            _mjpeg_frames(camera.pk),
            content_type="multipart/x-mixed-replace; boundary=frame",
        )

    def live_view(self, request):
        camera = Camera.objects.filter(pk=request.GET.get("camera")).first()
        if camera is None:
            raise Http404("unknown camera")
        context = {
            **self.admin_site.each_context(request),
            "title": f"Live — {camera.name}",
            "camera": camera,
            "mjpeg_url": reverse("admin:cameras_camera_mjpeg") + f"?camera={camera.pk}",
        }
        return TemplateResponse(request, "admin/cameras/camera_live.html", context)
