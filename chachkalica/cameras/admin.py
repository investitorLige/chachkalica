"""Cameras admin — register an RTSP camera by URL, watch it live, run a model on it.

Saving (add or edit) synchronously opens the stream for a moment to confirm
it's reachable; the row saves either way, the test result just sets
``status``/``last_error``. A "View live stream…" action (and an inline
preview on the change page) opens an MJPEG view backed by Redis — see
``cameras/management/commands/stream_cameras.py`` for the worker that keeps
that cache warm, and ``docs/camera-live-preview.md`` for the full design.

The same MJPEG view also serves an *annotated* stream (``?annotated=1``) with
detection boxes burnt in, fed by the ``infer_cameras`` worker off the
:class:`~cameras.models.CameraInference` config edited inline on the change
page — see ``docs/camera-live-inference.md``.
"""

import hashlib
import json
import re
import time

from django import forms
from django.contrib import admin, messages
from django.http import Http404, StreamingHttpResponse
from django.shortcuts import redirect
from django.template.response import TemplateResponse
from django.urls import path, reverse
from django.utils import timezone
from django.utils.html import format_html

from cameras.models import Camera, CameraInference
from cameras.services import frame_cache, rtsp
from training.services import pipeline_meta

_STATUS_COLORS = {
    "online": "#22c55e",
    "unknown": "#9ca3af",
    "offline": "#ef4444",
    "running": "#22c55e",
    "idle": "#9ca3af",
    "error": "#ef4444",
}

_CRED_RE = re.compile(r"^(rtsps?://)([^:@/]+):([^@/]+)@")

# Client polls Redis at this interval; the worker pushes faster than this so
# there's always a fresh frame waiting (see FRAME_PUSH_INTERVAL_SECONDS in
# stream_cameras.py).
_MJPEG_POLL_SECONDS = 0.1

# Re-send the current frame this often even when it hasn't changed. Nothing in
# the browser needs it — an <img> holds the last frame it was given — but a
# stream that can go silent indefinitely is at the mercy of whatever idle
# timeout sits between here and the viewer. One frame every few seconds keeps
# the connection provably alive for a rounding error's worth of bandwidth.
_MJPEG_KEEPALIVE_SECONDS = 5.0


def mask_rtsp_url(url: str) -> str:
    """Replace the password in ``rtsp://user:pass@host/...`` with dots."""
    return _CRED_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}:••••@", url or "")


def _status_badge(value: str):
    if not value:
        return "—"
    color = _STATUS_COLORS.get(value, "#9ca3af")
    return format_html('<b style="color:{};">●</b> {}', color, value)


def _mjpeg_frames(camera_id, annotated=False):
    """Stream a camera's cached frames as multipart JPEG, skipping repeats.

    Polling faster than the producers push (deliberately, so no frame is
    systematically missed) means the same frame is often found twice, and the
    frame is also whatever's in Redis when a stalled stream stops advancing.
    Sending those again would spend a megabyte to tell the browser what it is
    already displaying, so each frame goes out once, identified by digest.

    While ``annotated``, refreshes the ``:viewed`` heartbeat every poll —
    ``infer_cameras`` only runs a camera's model while that heartbeat is fresh,
    so this generator's lifetime (the life of the browser's connection) is what
    starts and stops live inference.
    """
    getter = frame_cache.get_annotated_frame if annotated else frame_cache.get_preview_frame
    last_digest = None
    last_sent = 0.0
    while True:
        if annotated:
            frame_cache.mark_viewed(camera_id)
        frame = getter(camera_id)
        if frame is not None:
            digest = hashlib.blake2b(frame, digest_size=16).digest()
            now = time.monotonic()
            if digest != last_digest or (now - last_sent) >= _MJPEG_KEEPALIVE_SECONDS:
                last_digest = digest
                last_sent = now
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )
        time.sleep(_MJPEG_POLL_SECONDS)


class CameraInferenceForm(forms.ModelForm):
    """Inline form for a camera's live-inference config.

    ``artifact_path`` becomes a dropdown of what's actually in the export output
    directory (same source as the video-inference page), and enabling the config
    is validated *here* by building the exact ``/predict_image`` payload the
    worker would build — so a missing checkpoint or an inconsistent pipeline is
    a form error on save rather than a ``status=error`` row discovered later.

    ``trained_model`` also carries a ``data-pipeline-defaults`` JSON attribute
    (one entry per offered model) so ``camera_inference_form.js`` can prefill the
    Pipeline section when the operator picks a model — same "serve it the way
    it was trained" behaviour as the video-inference page.
    """

    class Meta:
        model = CameraInference
        # The status half of the model belongs to the worker, not the operator —
        # excluded from the form entirely (it's still shown, via readonly_fields)
        # so a stale page can't post over what the worker just wrote.
        exclude = ["status", "last_error", "last_inference_at", "last_latency_ms",
                   "last_box_count"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from training.services import exports

        try:
            artifacts = exports.list_artifacts()
        except Exception:  # noqa: BLE001 - a misconfigured root must not break the page
            artifacts = []
        current = (self.instance.artifact_path or "").strip()
        choices = [("", "———")] + [
            (a["relpath"], f"{a['relpath']} ({a['kind']}, {a['size_mb']} MB)")
            for a in artifacts
        ]
        # Keep a previously-saved artifact selectable even if the file has since
        # been moved or deleted, so editing an unrelated field doesn't silently
        # blank the model out.
        if current and not any(c[0] == current for c in choices):
            choices.append((current, f"{current} (missing)"))
        self.fields["artifact_path"] = forms.ChoiceField(
            choices=choices, required=False,
            label="Exported artifact",
            help_text=self.Meta.model._meta.get_field("artifact_path").help_text,
        )
        # Both selects carry the same pipeline metadata schema
        # (training.services.pipeline_meta): exported artifacts read theirs from the
        # ".pipeline.json" sidecar beside the file, trained models from the frozen
        # record on the row. An artifact with nothing on record at all is left out
        # of the map — selecting it leaves the pipeline fields as they stand.
        artifact_defaults = {}
        for a in artifacts:
            d = exports.read_pipeline_defaults(a["relpath"])
            if d:
                artifact_defaults[a["relpath"]] = d
        self.fields["artifact_path"].widget.attrs["data-pipeline-defaults"] = json.dumps(
            artifact_defaults)

        trained_field = self.fields.get("trained_model")
        if trained_field is not None:
            trained_models = trained_field.queryset.select_related(
                "source_run_result__run__experiment"
            )
            defaults = {
                tm.pk: pipeline_meta.for_trained_model(tm) for tm in trained_models
            }
            trained_field.widget.attrs["data-pipeline-defaults"] = json.dumps(defaults)

    def clean(self):
        cleaned = super().clean()
        if cleaned.get("target_fps") is not None and cleaned["target_fps"] <= 0:
            self.add_error("target_fps", "Must be greater than 0.")

        model_source = cleaned.get("model_source")
        if model_source == CameraInference.TRAINED and not cleaned.get("trained_model"):
            self.add_error("trained_model", "Required when the source is a trained model.")
        if model_source == CameraInference.EXPORTED and not (cleaned.get("artifact_path") or "").strip():
            self.add_error("artifact_path", "Required when the source is an exported artifact.")

        if not cleaned.get("enabled") or self.errors:
            return cleaned

        # Same construction the worker performs, so the operator sees the real
        # error (missing checkpoint, pipeline needs a detector, ...) immediately.
        from cameras.services import live_inference

        probe = CameraInference(**{
            f: cleaned.get(f) for f in (
                "model_source", "trained_model", "artifact_path", "score_threshold",
                "pipeline", "detector_checkpoint", "detector_expand_ratio",
                "detector_min_box_size", "tile_size_px", "tile_width_pct",
                "tile_height_pct", "overlap", "merge_nms_iou", "chain",
            ) if f in cleaned
        })
        try:
            live_inference.build_predict_payload(probe)
        except RuntimeError as exc:
            raise forms.ValidationError(f"Cannot enable live inference: {exc}") from exc
        return cleaned


class CameraInferenceInline(admin.StackedInline):
    model = CameraInference
    form = CameraInferenceForm
    can_delete = True
    # 1, not 0: a camera with no config yet must still show a blank form to fill
    # in. Django caps a OneToOne inline at one row, and an untouched blank form
    # is not saved, so the add page is unaffected.
    extra = 1
    verbose_name_plural = "Live inference (run a model on this camera's frames)"
    readonly_fields = ["status_badge", "last_error", "last_inference_at",
                       "last_latency_ms", "last_box_count"]
    fieldsets = [
        (None, {
            "fields": ["enabled", "target_fps",
                       ("model_source", "trained_model", "artifact_path"),
                       "score_threshold"],
        }),
        ("Pipeline", {
            "classes": ["collapse"],
            "description": "How each frame is presented to the model. Leave as "
                           "'raw' to feed the whole frame straight in.",
            "fields": ["pipeline", "detector_checkpoint", "detector_expand_ratio",
                       "detector_min_box_size", "tile_size_px", "tile_width_pct",
                       "tile_height_pct", "overlap", "merge_nms_iou", "chain"],
        }),
        ("Worker status", {
            "fields": ["status_badge", "last_error", "last_inference_at",
                       "last_latency_ms", "last_box_count"],
        }),
    ]

    @admin.display(description="status")
    def status_badge(self, obj):
        return _status_badge(obj.status if obj and obj.pk else "")

    class Media:
        # Shows only the Pipeline-section fields the selected pipeline uses, and
        # prefills them from the picked model's training config — see the file
        # for both behaviours.
        js = ("cameras/camera_inference_form.js",)


@admin.register(Camera)
class CameraAdmin(admin.ModelAdmin):
    list_display = ["name", "masked_url", "status_badge", "inference_badge",
                    "last_checked_at", "created_at"]
    list_filter = ["status", "inference__enabled", "inference__status"]
    search_fields = ["name", "rtsp_url"]
    readonly_fields = ["live_preview", "status", "last_error", "last_checked_at",
                       "created_at", "updated_at"]
    fields = ["name", "rtsp_url", "live_preview", "status", "last_error",
              "last_checked_at", "created_at", "updated_at"]
    inlines = [CameraInferenceInline]
    actions = ["view_live_stream"]

    def get_queryset(self, request):
        return super().get_queryset(request).select_related("inference")

    @admin.display(description="rtsp url")
    def masked_url(self, obj):
        return mask_rtsp_url(obj.rtsp_url)

    @admin.display(description="status", ordering="status")
    def status_badge(self, obj):
        return _status_badge(obj.status)

    @admin.display(description="live inference")
    def inference_badge(self, obj):
        config = getattr(obj, "inference", None)
        if config is None:
            return "—"
        if not config.enabled:
            return "off"
        return format_html("{} · {}", _status_badge(config.status), config.model_label())

    @admin.display(description="Live preview")
    def live_preview(self, obj):
        if obj is None or not obj.pk:
            return "Save the camera first to preview it."
        mjpeg_url = reverse("admin:cameras_camera_mjpeg") + f"?camera={obj.pk}"
        live_url = reverse("admin:cameras_camera_live") + f"?camera={obj.pk}"
        config = getattr(obj, "inference", None)
        if config is not None and config.enabled:
            # Show the annotated stream by default once inference is on — that's
            # what the operator just turned on and wants to look at.
            mjpeg_url += "&annotated=1"
            live_url += "&annotated=1"
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
        annotated = request.GET.get("annotated") == "1"
        return StreamingHttpResponse(
            _mjpeg_frames(camera.pk, annotated=annotated),
            content_type="multipart/x-mixed-replace; boundary=frame",
        )

    def live_view(self, request):
        camera = Camera.objects.filter(pk=request.GET.get("camera")).first()
        if camera is None:
            raise Http404("unknown camera")
        annotated = request.GET.get("annotated") == "1"
        base = reverse("admin:cameras_camera_mjpeg") + f"?camera={camera.pk}"
        live = reverse("admin:cameras_camera_live") + f"?camera={camera.pk}"
        config = getattr(camera, "inference", None)
        context = {
            **self.admin_site.each_context(request),
            "title": f"Live — {camera.name}",
            "camera": camera,
            "inference": config,
            "annotated": annotated,
            "mjpeg_url": base + ("&annotated=1" if annotated else ""),
            # Link to the other view, so one page can flip between raw and boxes.
            "toggle_url": live if annotated else live + "&annotated=1",
            "boxes": frame_cache.get_boxes(camera.pk) if annotated else [],
        }
        return TemplateResponse(request, "admin/cameras/camera_live.html", context)
