"""Videos admin — import from disk, download from a link, and play.

Two ways to create a row (Add page):
  * pick a file already present under the videos root, or
  * paste a link, which a background rq job downloads into the videos root.

A "Play selected video…" action (and per-row ▶ play link) opens an HTML5 player
backed by a Range-aware streaming endpoint so seeking works.
"""

import base64
from pathlib import Path

import django_rq
from django import forms
from django.contrib import admin, messages
from django.contrib.admin.helpers import ACTION_CHECKBOX_NAME
from django.http import Http404, JsonResponse
from django.shortcuts import redirect
from django.template.response import TemplateResponse
from django.urls import path, reverse
from django.utils.html import format_html

from fleet.admin import _status_badge as _job_status_badge
from training.services import exports, inference_form
from videos import jobs
from videos.models import (
    FrameExtractionJob, InferenceJob, MarketingVideo, RenderPreset, Video,
)
from videos.services import (
    downloader, frame_extraction, inference, render_style, streaming,
)
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


# The model + pipeline half of this form is shared with the dataset-inference
# action (see training.services.inference_form) — the two must offer the same
# knobs, since they run the same models through the same geometry. `frame_stride`
# is this run's own, being a property of a video rather than of the model.
_DEFAULT_SCORE_THRESHOLD = inference_form.DEFAULT_SCORE_THRESHOLD
_DEFAULT_FRAME_STRIDE = 1


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


def _parse_float_in_range(raw: str, default, low: float, high: float):
    """Parse a float within ``[low, high]`` from POST data; blank -> ``default``, invalid -> ``None``."""
    raw = (raw or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if low <= value <= high else None


# The optional parsers and their blank/nonsense sentinel live in
# training.services.inference_form, which the dataset-inference form parses with
# too; re-exported here under their old names because this module is where
# marketing_studio.admin imports them from.
_INVALID = inference_form._INVALID
_parse_optional_float = inference_form.parse_optional_float
_parse_optional_int = inference_form.parse_optional_int


def _form_values(defaults: dict, posted) -> dict:
    """The values to render the step-2 form with: the shared pipeline half plus
    this form's own ``frame_stride``."""
    return inference_form.form_values(
        defaults, posted, extra={"frame_stride": _DEFAULT_FRAME_STRIDE})


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
    actions = ["play_video", "extract_frames", "run_inference",
               "run_inference_marketing", "import_all_new", "redownload"]

    def get_actions(self, request):
        """Relabel the stock ``delete_selected`` — deleting a video also deletes
        its file under ``videos_root`` (see ``videos.signals``), not just the row."""
        actions = super().get_actions(request)
        if "delete_selected" in actions:
            func, name, _desc = actions["delete_selected"]
            actions["delete_selected"] = (
                func, name, "Delete selected video(s) (also deletes the file on disk)",
            )
        return actions

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

    @admin.action(description="Run model inference…")
    def run_inference(self, request, queryset):
        """Two-step "Run model inference…": pick where the model comes from, then
        configure that model plus the pipeline the frames go through.

        Step 1 asks trained-catalogue vs exported-artifact vs infer bundle; step 2
        is rendered for the chosen source, so the model selector is the
        ``TrainedModel`` list, a scan of the export output directory, or a scan of
        the bundle root (neither an exported ``.onnx``/``.engine`` nor a bundle has
        a DB row). All three land on the same submit, which pre-flights the
        resulting ``/predict_image`` payload before enqueuing so a missing
        detector/artifact is reported here rather than as a failed job.

        The bundle source differs in one way, deliberately: its pipeline fields are
        filled and locked by the "Sync bundle" button rather than being the
        operator's to set, and are re-derived from the manifest on submit (see
        ``training.services.bundles``). A bundle carries the geometry its weights
        were tuned with, so the bundle is the record of what runs.
        """
        return self._inference_wizard(request, queryset, marketing=False)

    @admin.action(description="Run model inference for marketing…")
    def run_inference_marketing(self, request, queryset):
        """The same run, configured to be *watched* rather than inspected.

        Identical machinery to "Run model inference…" — same model sources, same
        pipeline, same job row, same queue — with one addition: a Look section
        whose thirty-odd knobs are stored on the job as
        ``InferenceJob.render_style`` and drive
        :class:`videos.services.render_style.MarketingRenderer` instead of the
        plain ``draw_boxes`` overlay. Rounded or bracketed boxes, translucent
        fills, pill labels, a spotlit background, a per-class counter, a
        watermark, temporal easing so boxes glide instead of flickering, and a
        delivery resolution / encode quality for the file that comes out.

        A separate action rather than a checkbox on the existing one, because the
        two have different jobs: the plain action answers "what did the model
        do?", where a stable, boring overlay is a feature and re-tuning its looks
        would be noise. This one answers "can we show this to someone?".

        It also carries a preview: styling is iterative and re-encoding a clip per
        iteration is not, so the form renders one frame through the very same
        renderer (see :meth:`marketing_preview_view`) and re-uses the model's
        answer for that frame across style changes.
        """
        return self._inference_wizard(request, queryset, marketing=True)

    def _inference_wizard(self, request, queryset, *, marketing: bool):
        """The shared body of both inference actions; ``marketing`` picks the
        step-2 template, the Look section and the message wording."""
        if queryset.count() != 1:
            self.message_user(request, "Select exactly one video to run inference on.",
                              level=messages.WARNING)
            return None
        video = queryset.first()
        if not video.exists():
            self.message_user(request, f"{video.name}: no file on disk yet.",
                              level=messages.WARNING)
            return None

        action = "run_inference_marketing" if marketing else "run_inference"
        heading = "Run model inference for marketing" if marketing else "Run model inference"
        base_context = {
            **self.admin_site.each_context(request),
            "video": video,
            "action": action,
            "marketing": marketing,
            "selected": [str(video.pk)],
            "action_checkbox_name": ACTION_CHECKBOX_NAME,
        }

        # ------------------------------------------------- step 1: model source
        model_source = request.POST.get("model_source") or ""
        if model_source not in dict(InferenceJob.MODEL_SOURCE_CHOICES):
            if request.POST.get("configure") or request.POST.get("apply"):
                self.message_user(request, "Choose which kind of model to run.",
                                  level=messages.WARNING)
            return TemplateResponse(request, "admin/videos/run_inference_source.html", {
                **base_context,
                "title": f"{heading} — {video.name}",
                **inference_form.source_step_context(),
            })

        # ------------------------------------------------------- step 2: submit
        if request.POST.get("apply"):
            job, error = self._build_inference_job(request, video, model_source,
                                                   marketing=marketing)
            if error:
                self.message_user(request, error, level=messages.WARNING)
            else:
                job.save()
                _queue().enqueue(jobs.run_inference, job.id,
                                 job_timeout=jobs.INFERENCE_JOB_TIMEOUT)
                tab = "Marketing videos" if marketing else "Inferred videos"
                self.message_user(
                    request,
                    f"{'Marketing render' if marketing else 'Inference'} queued for "
                    f"{video.name} with {job.model_label()} [{job.pipeline}] — see the "
                    f"{tab} tab for progress.",
                )
                return None

        # ------------------------------------------ step 2: render the full form
        # The model lists, their pipeline prefills and the pipeline vocabulary are
        # the shared half of every serving form (training.services.inference_form);
        # this action adds the video and its own frame_stride around them.
        model_ctx = inference_form.model_context(model_source, request.POST)
        selected = model_ctx["selected_model"]
        context = {
            **base_context,
            "title": f"{heading} — {video.name}",
            **model_ctx,
            **inference_form.pipeline_context(),
            "values": _form_values(model_ctx["defaults"], request.POST),
            "show_frame_stride": True,
        }

        if not marketing:
            return TemplateResponse(request, "admin/videos/run_inference.html", context)

        classes = self._class_names_by_model(
            model_source, model_ctx["trained_models"], model_ctx["artifacts"],
            model_ctx["bundles"])
        context.update(self._style_context(request, classes.get(selected) or []))
        # Keyed by the same string the model <select> posts, so the Look section's
        # "label every box as" dropdown can refill itself when a different model
        # is picked — same trick as the pipeline prefills above.
        context["classes_by_model"] = classes
        return TemplateResponse(request, "admin/videos/run_inference_marketing.html",
                                context)

    @staticmethod
    def _class_names_by_model(model_source, trained_models, artifacts, bundle_list):
        """``{<model select value>: [class name, …]}`` for the model list this
        render is showing — the vocabulary the relabel dropdown offers.

        Three sources for the three kinds of model, and any of them may come back
        empty: a bundle carries its class map in its manifest, a catalogued model
        on its row, and an exported artifact in the sidecars beside it (or in the
        catalogue entry it was exported from). Only the marketing form asks for
        this, because building it for the exported case reads a sidecar per
        artifact.
        """
        if model_source == InferenceJob.BUNDLE:
            return {b["relpath"]: list(b["classes"]) for b in bundle_list}
        if model_source == InferenceJob.EXPORTED:
            return {a["relpath"]: exports.read_class_names(a["relpath"])
                    for a in artifacts}
        return {str(tm.pk): list(tm.classes or []) for tm in trained_models}

    def _style_context(self, request, class_names=()):
        """Everything the Look section needs: the values to render it with, and
        the vocabularies its selects are built from — including ``class_names``,
        the selected model's class space, which the relabel dropdown offers.

        The starting values are the *last* marketing style used on this instance
        rather than the module defaults, so a house look, once dialled in, is one
        press away on the next clip — and a form re-rendered after a validation
        warning keeps what was typed, same rule as the pipeline fields above.
        """
        if request.POST.get("apply"):
            style, _error = render_style.parse_form(request.POST)
        else:
            previous = (
                InferenceJob.objects
                .exclude(render_style={})
                .order_by("-created_at")
                .values_list("render_style", flat=True)
                .first()
            )
            style = render_style.normalize(previous or {})
        # A relabel carried over from the last render (or loaded from a preset)
        # can name a class the selected model does not have — it is only ever
        # drawn, never matched — so it stays in the list rather than silently
        # reverting to "off" the moment the form re-renders.
        forced = style["force_class"]
        class_choices = list(class_names)
        if forced and forced not in class_choices:
            class_choices.append(forced)
        return {
            "style_values": render_style.form_values(style),
            "class_choices": class_choices,
            # Passed as form-shaped values (``style_<knob>`` keys), not raw style
            # dicts: loading a preset is then "assign each value to the input of
            # that name" and the page needs to know nothing about what a knob is.
            "presets": [
                {"pk": preset.pk, "name": preset.name,
                 "values": render_style.form_values(
                     render_style.normalize(preset.style))}
                for preset in RenderPreset.objects.all()
            ],
            "preset_save_url": reverse("admin:videos_video_render_preset_save"),
            "palette_choices": render_style.PALETTE_CHOICES,
            "color_mode_choices": render_style.COLOR_MODE_CHOICES,
            "box_style_choices": render_style.BOX_STYLE_CHOICES,
            "label_text_choices": render_style.LABEL_TEXT_CHOICES,
            "label_style_choices": render_style.LABEL_STYLE_CHOICES,
            "label_position_choices": render_style.LABEL_POSITION_CHOICES,
            "label_color_choices": render_style.LABEL_COLOR_CHOICES,
            "font_choices": render_style.FONT_CHOICES,
            "animation_choices": render_style.ANIMATION_CHOICES,
            "counter_choices": render_style.CORNER_CHOICES,
            "watermark_position_choices": [
                c for c in render_style.CORNER_CHOICES if c[0] != "none"],
            "preview_url": reverse("admin:videos_video_marketing_preview"),
        }

    def _build_inference_job(self, request, video, model_source, *, marketing=False):
        """Validate the step-2 POST into an unsaved :class:`InferenceJob`.

        Returns ``(job, None)`` on success or ``(None, message)`` on the first
        problem found, so the caller can re-render the form with a warning. With
        ``marketing`` the Look section is parsed too, onto ``render_style`` — the
        one field that separates the two actions' output.
        """
        # Which model, and the geometry it runs through: the shared half of every
        # serving form, parsed and pruned in one place
        # (training.services.inference_form).
        model_fields, error = inference_form.parse_model_source(request.POST, model_source)
        if error:
            return None, error
        knobs, error = inference_form.parse_pipeline_fields(request.POST)
        if error:
            return None, error

        frame_stride = _parse_positive_int(request.POST.get("frame_stride"), default=1)
        if frame_stride is None:
            return None, "Frame stride must be a positive integer."

        style = {}
        if marketing:
            style, style_error = render_style.parse_form(request.POST)
            if style_error:
                return None, style_error

        job = InferenceJob(
            video=video,
            model_source=model_source,
            frame_stride=frame_stride,
            render_style=style,
            output_filename=inference.unique_output_filename(video.name),
            **model_fields,
            **knobs,
        )

        # A bundle's geometry is the bundle's, not the form's: whatever was posted
        # for the pipeline fields is replaced by the manifest's own values. This is
        # the server-side half of the locked fields on the form — a stale page or a
        # bundle that changed on disk since the sync still runs what the bundle
        # says today.
        try:
            job.sync_bundle()
        except ValueError as exc:
            return None, str(exc)

        # Pre-flight the exact payload the job will send, so a missing artifact,
        # an out-of-root path, or a detector-less person pipeline fails here.
        try:
            inference.build_predict_payload(job)
        except RuntimeError as exc:
            return None, str(exc)
        return job, None

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
            path("marketing-preview/",
                 self.admin_site.admin_view(self.marketing_preview_view),
                 name="videos_video_marketing_preview"),
            path("render-preset/save/",
                 self.admin_site.admin_view(self.render_preset_save_view),
                 name="videos_video_render_preset_save"),
        ]
        return custom + super().get_urls()

    def marketing_preview_view(self, request):
        """Render one frame the way the finished marketing video will look.

        POSTs the whole step-2 form (the marketing template's preview button
        submits the same field names, so nothing has to be kept in sync) plus
        ``video``, ``position`` (0–1 along the clip) and optionally ``refresh=1``
        to re-run the model rather than re-use its cached answer for that frame.
        Returns the frame as a data URI.

        POST-only and staff-only for the same reason as the bundle-sync endpoint:
        a cache miss runs the model, which takes the trainer's GPU. The base64
        round-trip costs ~33% over serving bytes, and buys a reply that carries
        the box count and the cache flag alongside the image, and a page that
        needs no second request to show it.
        """
        if request.method != "POST":
            return JsonResponse({"error": "POST required."}, status=405)

        video = Video.objects.filter(pk=request.POST.get("video")).first()
        if video is None or not video.exists():
            return JsonResponse({"error": "No such video on disk."}, status=404)

        model_source = request.POST.get("model_source") or ""
        if model_source not in dict(InferenceJob.MODEL_SOURCE_CHOICES):
            return JsonResponse({"error": "Choose which kind of model to run."}, status=400)

        job, error = self._build_inference_job(request, video, model_source, marketing=True)
        if error:
            return JsonResponse({"error": error}, status=400)

        try:
            position = float(request.POST.get("position") or 0.5)
        except ValueError:
            position = 0.5

        # The model is remote (the trainer container) and the video is a file
        # someone else's tooling wrote: every failure here belongs in the preview
        # panel as text, not as an opaque 500 behind a fetch.
        try:
            result = inference.preview_frame(
                job, position, use_cache=not request.POST.get("refresh"))
        except Exception as exc:  # noqa: BLE001 - reported in the preview panel
            return JsonResponse({"error": f"{type(exc).__name__}: {exc}"}, status=500)

        encoded = base64.b64encode(result["jpeg"]).decode()
        return JsonResponse({
            "image": f"data:image/jpeg;base64,{encoded}",
            "width": result["width"],
            "height": result["height"],
            "boxes": result["boxes"],
            "detections": result["detections"],
            "frame_index": result["frame_index"],
            "frames_total": result["frames_total"],
            "cached": result["cached"],
        })

    def render_preset_save_view(self, request):
        """Save the Look section of the marketing form as a named preset.

        POSTs the whole form plus ``preset_name`` — same "the form is the
        payload" contract as the preview endpoint, so this knows nothing about
        individual knobs either. Saving by name is an **upsert**: pressing save
        again with a preset's own name is how you amend it, which is what
        someone who has just tweaked a loaded preset means by "save".

        The reply is the preset in the form's own shape, so the page can add it
        to the dropdown and select it without a reload.
        """
        if request.method != "POST":
            return JsonResponse({"error": "POST required."}, status=405)

        name = (request.POST.get("preset_name") or "").strip()
        if not name:
            return JsonResponse({"error": "Name the preset first."}, status=400)
        if len(name) > 120:
            return JsonResponse({"error": "That name is too long (120 max)."},
                                status=400)

        style, error = render_style.parse_form(request.POST)
        if error:
            return JsonResponse({"error": error}, status=400)

        preset, created = RenderPreset.objects.update_or_create(
            name=name, defaults={"style": style})
        return JsonResponse({
            "pk": preset.pk,
            "name": preset.name,
            "created": created,
            "values": render_style.form_values(style),
        })

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
        """Stream the mp4 bytes, honouring HTTP Range so the player can seek."""
        video = Video.objects.filter(pk=request.GET.get("video")).first()
        if video is None or not video.exists():
            raise Http404("video file not found")
        return streaming.serve(request, video.path())


@admin.register(RenderPreset)
class RenderPresetAdmin(admin.ModelAdmin):
    """Presets are created by the marketing form's "Save as preset" button; this
    page is for renaming, deleting, and reading back exactly what a look is."""

    list_display = ["name", "summary", "updated_at"]
    search_fields = ["name"]
    readonly_fields = ["created_at", "updated_at"]
    ordering = ["name"]

    @admin.display(description="look")
    def summary(self, obj):
        return obj.summary()


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


@admin.register(InferenceJob)
class InferenceJobAdmin(admin.ModelAdmin):
    """Read-only log of "Run model inference…" runs — rows are only created by the action."""

    list_display = ["video", "model_display", "pipeline", "look", "status_badge",
                     "frames_processed", "play_link", "download_link", "created_at"]
    list_filter = ["status", "model_source", "pipeline", "trained_model"]
    search_fields = ["video__name", "trained_model__name", "artifact_path",
                     "bundle_path"]
    readonly_fields = [f.name for f in InferenceJob._meta.fields]
    ordering = ["-created_at"]

    def has_add_permission(self, request):
        return False

    def get_queryset(self, request):
        # Marketing renders live under their own "Marketing videos" tab (see
        # MarketingVideoAdmin below) so they don't clutter this one — same split
        # as training.admin.TrainingRunAdmin / FineTuningRunAdmin.
        return super().get_queryset(request).filter(render_style={})

    def get_actions(self, request):
        """Relabel the stock ``delete_selected`` so it's clear the annotated
        mp4 goes with the row, not just orphaned under ``inferred/``.

        The file removal itself lives in ``videos.signals`` (a ``post_delete``
        receiver) rather than here, so it also fires when a row disappears by
        cascade (e.g. deleting the row's ``TrainedModel``) — a path that never
        goes through this ``ModelAdmin`` at all.
        """
        actions = super().get_actions(request)
        if "delete_selected" in actions:
            func, name, _desc = actions["delete_selected"]
            actions["delete_selected"] = (
                func, name,
                "Delete selected inferred videos (also deletes the annotated file)",
            )
        return actions

    @admin.display(description="status", ordering="status")
    def status_badge(self, obj):
        return _job_status_badge(obj.status)

    @admin.display(description="model")
    def model_display(self, obj):
        """The trained model's name, or the exported artifact's / bundle's path."""
        if obj.model_source == InferenceJob.EXPORTED:
            return format_html('<span title="exported artifact">{}</span>',
                               obj.artifact_path or "—")
        if obj.model_source == InferenceJob.BUNDLE:
            return format_html('<span title="infer bundle">{}</span>',
                               obj.bundle_path or "—")
        if not obj.trained_model_id:
            return "—"
        url = reverse("admin:training_trainedmodel_change", args=[obj.trained_model_id])
        return format_html('<a href="{}">{}</a>', url, obj.trained_model.name)

    @admin.display(description="look")
    def look(self, obj):
        """Which of the two overlays this row was rendered with, and the couple of
        style choices that actually change what the clip looks like from across
        the room — enough to tell two marketing renders of the same clip apart in
        the list, with the full style dict on the row's own page."""
        if not obj.render_style:
            return format_html('<span style="color:#9ca3af">plain</span>')
        style = render_style.normalize(obj.render_style)
        # The relabel earns its place here: it is the one style choice that changes
        # what the clip *says* rather than how it looks, so a row rendered with one
        # should never be mistaken for the model's own labels.
        forced = style["force_class"]
        return format_html(
            '<span title="{}">🎬 {} · {}{}</span>',
            "marketing render", style["box_style"], style["palette"],
            f" · all “{forced}”" if forced else "",
        )

    @admin.display(description="")
    def play_link(self, obj):
        if not obj.output_exists():
            return "—"
        url = reverse(f"admin:{self._url_name('play')}") + f"?job={obj.pk}"
        return format_html('<a class="button" href="{}">▶ play</a>', url)

    @admin.display(description="")
    def download_link(self, obj):
        # Same trick as the player page's own download button: a same-origin
        # link with a `download` attribute makes the browser save the file
        # rather than navigate to it, with no server-side attachment handling
        # needed — the stream endpoint already serves the raw mp4 bytes.
        if not obj.output_exists():
            return "—"
        url = reverse(f"admin:{self._url_name('stream')}") + f"?job={obj.pk}"
        return format_html(
            '<a class="button" href="{}" download="{}">⭳ download</a>',
            url, obj.output_filename,
        )

    def _url_name(self, suffix: str) -> str:
        """A URL name scoped to this ModelAdmin's own model.

        MarketingVideoAdmin subclasses this admin over the MarketingVideo proxy
        (see videos.models.MarketingVideo) rather than overriding get_urls, so
        deriving the name from ``self.model`` — the same trick Django's own
        admin uses for its changelist/add/change routes, and
        training.admin.TrainingRunAdmin uses for the same reason — is what
        keeps its routes (``videos_marketingvideo_play`` etc.) distinct from
        InferenceJobAdmin's, instead of two ``path()`` entries silently sharing
        one name and ``reverse()`` picking whichever was registered last.
        """
        opts = self.model._meta
        return f"{opts.app_label}_{opts.model_name}_{suffix}"

    def get_urls(self):
        custom = [
            path("play/", self.admin_site.admin_view(self.play_view),
                 name=self._url_name("play")),
            path("stream/", self.admin_site.admin_view(self.stream_view),
                 name=self._url_name("stream")),
        ]
        return custom + super().get_urls()

    def play_view(self, request):
        job = InferenceJob.objects.select_related("video").filter(
            pk=request.GET.get("job")).first()
        if job is None:
            raise Http404("unknown inference job")
        context = {
            **self.admin_site.each_context(request),
            "title": f"Inferred — {job.video.name}",
            "job": job,
            "exists": job.output_exists(),
            "stream_url": reverse(f"admin:{self._url_name('stream')}") + f"?job={job.pk}",
        }
        return TemplateResponse(request, "admin/videos/inference_player.html", context)

    def stream_view(self, request):
        """Stream the annotated mp4, honouring HTTP Range so the player can seek."""
        job = InferenceJob.objects.filter(pk=request.GET.get("job")).first()
        if job is None or not job.output_exists():
            raise Http404("inference output not found")
        return streaming.serve(request, job.output_path())


@admin.register(MarketingVideo)
class MarketingVideoAdmin(InferenceJobAdmin):
    """"Marketing videos" tab: the same log as "Inferred videos", scoped to the
    runs made by "Run model inference for marketing…".

    A plain subclass over the ``MarketingVideo`` proxy (see
    ``videos.models.MarketingVideo``) — every display, action, and the play/
    stream views (see ``InferenceJobAdmin._url_name``) come along for free.
    Only the queryset differs, and only in which half of the split it keeps.
    """

    def get_queryset(self, request):
        return InferenceJob.objects.exclude(render_style={})
