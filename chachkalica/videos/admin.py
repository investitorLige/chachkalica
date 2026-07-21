"""Videos admin — import from disk, download from a link, and play.

Two ways to create a row (Add page):
  * pick a file already present under the videos root, or
  * paste a link, which a background rq job downloads into the videos root.

A "Play selected video…" action (and per-row ▶ play link) opens an HTML5 player
backed by a Range-aware streaming endpoint so seeking works.
"""

import re
from pathlib import Path

import django_rq
from django import forms
from django.contrib import admin, messages
from django.contrib.admin.helpers import ACTION_CHECKBOX_NAME
from django.http import FileResponse, Http404, StreamingHttpResponse
from django.shortcuts import redirect
from django.template.response import TemplateResponse
from django.urls import path, reverse
from django.utils.html import format_html

from fleet.admin import _status_badge as _job_status_badge
from videos import jobs
from videos.models import FrameExtractionJob, Video
from videos.services import downloader, frame_extraction
from videos.services.videos import list_video_files

_STATUS_COLORS = {
    "ready": "#22c55e",
    "downloading": "#f59e0b",
    "pending": "#9ca3af",
    "error": "#ef4444",
}

_QUALITY_CHOICES = [
    ("", "best available"),
    ("2160", "2160p (4K)"),
    ("1440", "1440p"),
    ("1080", "1080p"),
    ("720", "720p"),
    ("480", "480p"),
]


def _queue():
    return django_rq.get_queue("default")


def _status_badge(value: str):
    if not value:
        return "—"
    color = _STATUS_COLORS.get(value, "#9ca3af")
    return format_html('<b style="color:{};">●</b> {}', color, value)


def _parse_positive_int(raw: str, default):
    """Parse a positive int from POST data; blank -> ``default``, invalid -> ``None``."""
    raw = (raw or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value >= 1 else None


def _parse_int_in_range(raw: str, default, low: int, high: int):
    """Parse an int within ``[low, high]`` from POST data; blank -> ``default``, invalid -> ``None``."""
    raw = (raw or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if low <= value <= high else None


class VideoAddForm(forms.ModelForm):
    """Add-video form: import an existing file OR download from a link.

    ``import_file`` and ``source_url`` are declared (non-model) fields; exactly
    one must be provided. Mirrors ``fleet.admin.DatasetAdminForm`` in offering the
    on-disk files as a dropdown, minus those already registered.
    """

    import_file = forms.ChoiceField(
        required=False,
        label="Import existing file",
        help_text="A video already sitting in the videos folder.",
    )
    source_url = forms.URLField(
        required=False,
        label="Download from link",
        help_text="Paste a YouTube (or other) link to download into the videos folder.",
    )
    quality = forms.ChoiceField(
        choices=_QUALITY_CHOICES, required=False,
        label="Download quality",
        help_text="Only used when downloading from a link.",
    )

    class Meta:
        model = Video
        fields = ["name", "quality"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        taken = set(Video.objects.values_list("filename", flat=True))
        choices = [(f, f) for f in list_video_files() if f not in taken]
        self.fields["import_file"].choices = [("", "— select a file —")] + choices
        self.fields["name"].required = False
        self.fields["name"].help_text = "Optional — defaults to the file/video title."

    def clean(self):
        cleaned = super().clean()
        import_file = cleaned.get("import_file")
        source_url = cleaned.get("source_url")
        if bool(import_file) == bool(source_url):
            raise forms.ValidationError(
                "Provide exactly one of: an existing file to import, or a link to download."
            )
        if import_file and not cleaned.get("name"):
            cleaned["name"] = Path(import_file).stem
        return cleaned


@admin.register(Video)
class VideoAdmin(admin.ModelAdmin):
    add_form = VideoAddForm
    add_fieldsets = ((None, {"fields": ("import_file", "source_url", "quality", "name")}),)

    list_display = ["name", "status_badge", "source_url", "filename", "created_at", "play_link"]
    list_filter = ["status"]
    search_fields = ["name", "filename", "source_url"]
    readonly_fields = ["status", "filename", "player", "last_error", "created_at", "updated_at"]
    actions = ["play_video", "extract_frames", "import_all_new", "redownload"]

    # ------------------------------------------------------------------ forms
    def get_form(self, request, obj=None, **kwargs):
        defaults = {}
        if obj is None:
            defaults["form"] = self.add_form
        defaults.update(kwargs)
        return super().get_form(request, obj, **defaults)

    def get_fieldsets(self, request, obj=None):
        if obj is None:
            return self.add_fieldsets
        return [
            (None, {"fields": ["name", "status", "player"]}),
            ("File", {"fields": ["filename", "source_url", "quality"]}),
            ("Details", {"classes": ["collapse"],
                         "fields": ["last_error", "created_at", "updated_at"]}),
        ]

    def save_model(self, request, obj, form, change):
        if change:
            super().save_model(request, obj, form, change)
            return

        import_file = form.cleaned_data.get("import_file")
        source_url = form.cleaned_data.get("source_url")
        if import_file:
            obj.filename = import_file
            obj.source_url = ""
            obj.status = Video.READY
            if not obj.name:
                obj.name = Path(import_file).stem
            super().save_model(request, obj, form, change)
            return

        # Download flow: create the row now, let the worker fetch the bytes.
        obj.source_url = source_url
        obj.filename = ""
        obj.status = Video.DOWNLOADING
        if not obj.name:
            obj.name = jobs._unique_name(source_url)
        super().save_model(request, obj, form, change)
        _queue().enqueue(jobs.download_video, obj.id)
        self.message_user(
            request,
            f"Download queued for {source_url} — refresh to see it turn ready.",
        )

    # --------------------------------------------------------------- displays
    @admin.display(description="status", ordering="status")
    def status_badge(self, obj):
        return _status_badge(obj.status)

    @admin.display(description="")
    def play_link(self, obj):
        if not obj.exists():
            return "—"
        url = reverse("admin:videos_video_play") + f"?video={obj.pk}"
        return format_html('<a class="button" href="{}">▶ play</a>', url)

    @admin.display(description="Player")
    def player(self, obj):
        if obj is None or not obj.pk or not obj.exists():
            return "—"
        url = reverse("admin:videos_video_stream") + f"?video={obj.pk}"
        return format_html(
            '<video controls preload="metadata" style="max-width:640px;width:100%;'
            'border-radius:6px" src="{}"></video>',
            url,
        )

    # ---------------------------------------------------------------- actions
    @admin.action(description="Play selected video…")
    def play_video(self, request, queryset):
        if queryset.count() != 1:
            self.message_user(request, "Select exactly one video to play.",
                              level=messages.WARNING)
            return None
        video = queryset.first()
        if not video.exists():
            self.message_user(request, f"{video.name}: no file on disk yet.",
                              level=messages.WARNING)
            return None
        return redirect(reverse("admin:videos_video_play") + f"?video={video.pk}")

    @admin.action(description="Extract frames…")
    def extract_frames(self, request, queryset):
        if queryset.count() != 1:
            self.message_user(request, "Select exactly one video to extract frames from.",
                              level=messages.WARNING)
            return None
        video = queryset.first()
        if not video.exists():
            self.message_user(request, f"{video.name}: no file on disk yet.",
                              level=messages.WARNING)
            return None

        if request.POST.get("apply"):
            dataset_name = (request.POST.get("dataset_name") or "").strip()
            if not dataset_name:
                self.message_user(request, "Enter a dataset name.", level=messages.WARNING)
                return None

            every_n_frames = _parse_positive_int(request.POST.get("every_n_frames"), default=30)
            if every_n_frames is None:
                self.message_user(request, "Every-N-frames must be a positive integer.",
                                  level=messages.WARNING)
                return None

            max_frames_raw = (request.POST.get("max_frames_per_video") or "").strip()
            max_frames_per_video = None
            if max_frames_raw:
                max_frames_per_video = _parse_positive_int(max_frames_raw, default=None)
                if max_frames_per_video is None:
                    self.message_user(
                        request, "Max frames per video must be a positive integer.",
                        level=messages.WARNING)
                    return None

            image_ext = request.POST.get("image_ext") or ".jpg"
            if image_ext not in (".jpg", ".png"):
                image_ext = ".jpg"

            classes = [
                line.strip() for line in (request.POST.get("classes") or "").splitlines()
                if line.strip()
            ]
            if not classes:
                self.message_user(request, "Enter at least one class name.",
                                  level=messages.WARNING)
                return None

            only_with_people = bool(request.POST.get("only_with_people"))

            similarity_threshold = None
            if request.POST.get("remove_similar_frames"):
                similarity_threshold = _parse_int_in_range(
                    request.POST.get("similarity_threshold"), default=5, low=0, high=64)
                if similarity_threshold is None:
                    self.message_user(
                        request, "Similarity threshold must be an integer between 0 and 64.",
                        level=messages.WARNING)
                    return None

            job = FrameExtractionJob.objects.create(
                video=video,
                dataset_name=dataset_name,
                every_n_frames=every_n_frames,
                max_frames_per_video=max_frames_per_video,
                image_ext=image_ext,
                classes_text="\n".join(classes),
                only_with_people=only_with_people,
                similarity_threshold=similarity_threshold,
            )
            _queue().enqueue(jobs.extract_frames, job.id, job_timeout=jobs.JOB_TIMEOUT)
            self.message_user(
                request,
                f"Frame extraction queued for {video.name} → dataset {dataset_name!r} "
                "— refresh to see progress.",
            )
            return None

        context = {
            **self.admin_site.each_context(request),
            "title": f"Extract frames — {video.name}",
            "video": video,
            "suggested_name": frame_extraction.unique_dataset_name(video.name),
            "action": "extract_frames",
            "selected": [str(video.pk)],
            "action_checkbox_name": ACTION_CHECKBOX_NAME,
        }
        return TemplateResponse(request, "admin/videos/extract_frames.html", context)

    @admin.action(description="Import every new file in the videos folder")
    def import_all_new(self, request, queryset):
        taken = set(Video.objects.values_list("filename", flat=True))
        created = 0
        for filename in list_video_files():
            if filename in taken:
                continue
            name = jobs._unique_name(Path(filename).stem)
            Video.objects.create(name=name, filename=filename, status=Video.READY)
            created += 1
        if created:
            self.message_user(request, f"Imported {created} new video file(s).")
        else:
            self.message_user(request, "No new files to import.", level=messages.INFO)

    @admin.action(description="Re-download selected link-backed video(s)")
    def redownload(self, request, queryset):
        queued = 0
        for video in queryset:
            if not video.source_url:
                self.message_user(request, f"{video.name}: no source link to re-download.",
                                  level=messages.WARNING)
                continue
            video.status = Video.DOWNLOADING
            video.last_error = ""
            video.save(update_fields=["status", "last_error", "updated_at"])
            _queue().enqueue(jobs.download_video, video.id)
            queued += 1
        if queued:
            self.message_user(request, f"{queued} re-download job(s) queued — refresh to see progress.")

    # ------------------------------------------------------------------- urls
    def get_urls(self):
        custom = [
            path("play/", self.admin_site.admin_view(self.play_view),
                 name="videos_video_play"),
            path("stream/", self.admin_site.admin_view(self.stream_view),
                 name="videos_video_stream"),
        ]
        return custom + super().get_urls()

    def play_view(self, request):
        video = Video.objects.filter(pk=request.GET.get("video")).first()
        if video is None:
            raise Http404("unknown video")
        context = {
            **self.admin_site.each_context(request),
            "title": f"Play — {video.name}",
            "video": video,
            "exists": video.exists(),
            "stream_url": reverse("admin:videos_video_stream") + f"?video={video.pk}",
        }
        return TemplateResponse(request, "admin/videos/video_player.html", context)

    def stream_view(self, request):
        """Stream the mp4 bytes, honouring HTTP Range so the player can seek.

        Django's ``FileResponse`` does not implement Range on its own, so a
        ``Range`` request is served as a bounded ``206`` here.
        """
        video = Video.objects.filter(pk=request.GET.get("video")).first()
        if video is None or not video.exists():
            raise Http404("video file not found")
        path = video.path()
        size = path.stat().st_size
        content_type = "video/mp4"

        range_header = request.headers.get("Range", "")
        match = re.match(r"bytes=(\d+)-(\d*)", range_header.strip())
        if not match:
            resp = FileResponse(open(path, "rb"), content_type=content_type)
            resp["Accept-Ranges"] = "bytes"
            return resp

        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else size - 1
        end = min(end, size - 1)
        start = min(start, end)
        length = end - start + 1

        resp = StreamingHttpResponse(
            _file_chunks(path, start, length), status=206, content_type=content_type
        )
        resp["Content-Length"] = str(length)
        resp["Content-Range"] = f"bytes {start}-{end}/{size}"
        resp["Accept-Ranges"] = "bytes"
        return resp


def _file_chunks(path, start, length, chunk_size=8192):
    with open(path, "rb") as fh:
        fh.seek(start)
        remaining = length
        while remaining > 0:
            data = fh.read(min(chunk_size, remaining))
            if not data:
                break
            remaining -= len(data)
            yield data


@admin.register(FrameExtractionJob)
class FrameExtractionJobAdmin(admin.ModelAdmin):
    """Read-only log of "Extract frames…" runs — rows are only created by the action."""

    list_display = ["video", "dataset_name", "status_badge", "frames_extracted",
                     "dataset_link", "created_at"]
    list_filter = ["status"]
    search_fields = ["dataset_name", "video__name"]
    readonly_fields = [f.name for f in FrameExtractionJob._meta.fields]
    ordering = ["-created_at"]

    def has_add_permission(self, request):
        return False

    @admin.display(description="status", ordering="status")
    def status_badge(self, obj):
        return _job_status_badge(obj.status)

    @admin.display(description="dataset")
    def dataset_link(self, obj):
        if not obj.dataset_id:
            return "—"
        url = reverse("admin:fleet_dataset_change", args=[obj.dataset_id])
        return format_html('<a href="{}">{}</a>', url, obj.dataset.name)
