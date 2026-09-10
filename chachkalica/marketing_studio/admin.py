"""Marketing Studio admin — three tabs and one action.

Videos (the studio's own library), Inferred videos (the render log), and Render
presets (saved looks). The single action, "Run model inference for marketing…",
is the marketing wizard from ``videos.admin`` with the model-source step
removed: here a render always runs an infer bundle, so the form opens directly
on the bundle it will run.

Everything stateless is imported rather than copied — the style vocabulary and
renderer (``videos.services.render_style``), the ffmpeg render loop and preview
cache (``videos.services.inference``), the downloader and the Range-aware
streamer. What this app keeps to itself is the state: its own tables, its own
video root, its own bundle root, its own URLs.
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
from marketing_studio import jobs
from marketing_studio.models import Render, RenderPreset, Video
from marketing_studio.services import paths
from marketing_studio.services.library import list_video_files
from training import pipelines
from training.services import bundles, pipeline_meta
from videos.admin import (
    _INVALID, _QUALITY_CHOICES, _form_values, _parse_float_in_range,
    _parse_optional_float, _parse_optional_int, _parse_positive_int,
    _status_badge,
)
from videos.services import inference, render_style, streaming


def _queue():
    return django_rq.get_queue("default")


class VideoAddForm(forms.ModelForm):
    """Add-video form: import an existing file OR download from a link.

    Twin of ``videos.admin.VideoAddForm``, over this app's own model and root.
    """

    import_file = forms.ChoiceField(
        required=False,
        label="Import existing file",
        help_text="A video already sitting in the Marketing Studio videos folder.",
    )
    source_url = forms.URLField(
        required=False,
        label="Download from link",
        help_text="Paste a YouTube (or other) link to download into the studio's folder.",
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
    actions = ["play_video", "run_marketing_render", "import_all_new", "redownload"]

    def get_actions(self, request):
        """Relabel the stock ``delete_selected`` — deleting a video also deletes
        its file under the studio's root (see ``marketing_studio.signals``)."""
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
        url = reverse(f"admin:{self._url_name('play')}") + f"?video={obj.pk}"
        return format_html('<a class="button" href="{}">▶ play</a>', url)

    @admin.display(description="Player")
    def player(self, obj):
        if obj is None or not obj.pk or not obj.exists():
            return "—"
        url = reverse(f"admin:{self._url_name('stream')}") + f"?video={obj.pk}"
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
        return redirect(
            reverse(f"admin:{self._url_name('play')}") + f"?video={video.pk}")

    @admin.action(description="Run model inference for marketing…")
    def run_marketing_render(self, request, queryset):
        """The studio's one action: a bundle, a look, and a rendered clip.

        The counterpart of ``videos.admin.VideoAdmin.run_inference_marketing``,
        minus its first step. That action has to ask which *kind* of model to
        run before it can draw a form; here the answer is always "a bundle", so
        the page the operator lands on is already the real one.
        """
        if queryset.count() != 1:
            self.message_user(request, "Select exactly one video to render.",
                              level=messages.WARNING)
            return None
        video = queryset.first()
        if not video.exists():
            self.message_user(request, f"{video.name}: no file on disk yet.",
                              level=messages.WARNING)
            return None

        bundle_list = bundles.list_bundles(paths.bundle_settings())

        if request.POST.get("apply"):
            render, error = self._build_render(request, video)
            if error:
                self.message_user(request, error, level=messages.WARNING)
            else:
                render.save()
                _queue().enqueue(jobs.run_render, render.id,
                                 job_timeout=jobs.RENDER_JOB_TIMEOUT)
                self.message_user(
                    request,
                    f"Marketing render queued for {video.name} with "
                    f"{render.model_label()} [{render.pipeline}] — see the "
                    f"Inferred videos tab for progress.",
                )
                return None

        # No preselection and no prefill: a bundle's geometry arrives via the
        # explicit "Sync bundle" press, which is also the only thing that reports
        # whether the bundle loads here (bundles.validate). Landing on a
        # preselected bundle with its fields already filled would imply that
        # check had happened.
        selected = request.POST.get("bundle_path") or ""
        defaults = pipeline_meta.raw()

        classes = {b["relpath"]: list(b["classes"]) for b in bundle_list}
        context = {
            **self.admin_site.each_context(request),
            "title": f"Run model inference for marketing — {video.name}",
            "video": video,
            "action": "run_marketing_render",
            "selected": [str(video.pk)],
            "action_checkbox_name": ACTION_CHECKBOX_NAME,
            "bundles": bundle_list,
            "bundles_root": str(paths.bundles_root()),
            "bundle_sync_url": reverse(f"admin:{self._url_name('bundle_sync')}"),
            "selected_model": selected,
            "values": _form_values(defaults, request.POST),
            "default_score_threshold": 0.5,
            "pipeline_choices": Render.PIPELINE_CHOICES,
            "detector_pipelines": " ".join(sorted(pipelines.DETECTOR_PIPELINES)),
            "tiling_pipelines": " ".join([
                pipelines.BATCH_DETECT, pipelines.BATCH_PEOPLE, pipelines.CHAIN]),
            "chain_pipelines": pipelines.CHAIN,
            # Every pipeline merges several per-frame predictions back together;
            # only "raw" has nothing to merge.
            "merging_pipelines": " ".join(
                value for value, _label in pipelines.PIPELINE_CHOICES),
            # Keyed by the same string the bundle <select> posts, so the Look
            # section's "label every box as" dropdown can refill itself when a
            # different bundle is picked.
            "classes_by_model": classes,
        }
        context.update(self._style_context(request, classes.get(selected) or []))
        return TemplateResponse(
            request, "admin/marketing_studio/run_render.html", context)

    def _style_context(self, request, class_names=()):
        """Everything the Look section needs: the values to render it with, and
        the vocabularies its selects are built from — including ``class_names``,
        the selected bundle's class space, which the relabel dropdown offers.

        The starting values are the *last* style used **in this app** rather than
        the module defaults, so a house look, once dialled in, is one press away
        on the next clip. A form re-rendered after a validation warning keeps
        what was typed instead.
        """
        if request.POST.get("apply"):
            style, _error = render_style.parse_form(request.POST)
        else:
            # No .exclude(render_style={}) here, unlike the videos app: every row
            # in this table is a marketing render, so there is no plain half to
            # skip past.
            previous = (
                Render.objects
                .order_by("-created_at")
                .values_list("render_style", flat=True)
                .first()
            )
            style = render_style.normalize(previous or {})
        # A relabel carried over from the last render (or loaded from a preset)
        # can name a class the selected bundle does not have — it is only ever
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
            "preset_save_url": reverse(
                f"admin:{self._url_name('render_preset_save')}"),
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
            "preview_url": reverse(f"admin:{self._url_name('preview')}"),
        }

    def _build_render(self, request, video):
        """Validate the form POST into an unsaved :class:`Render`.

        Returns ``(render, None)`` on success or ``(None, message)`` on the first
        problem found, so the caller can re-render the form with a warning.
        Counterpart of ``videos.admin.VideoAdmin._build_inference_job`` with the
        trained/exported branches gone and the Look section always parsed.
        """
        bundle_path = (request.POST.get("bundle_path") or "").strip()
        if not bundle_path:
            return None, "Choose a bundle."
        # Structural checks only — the load test belongs to the "Sync bundle"
        # button, where the operator asked for it; submitting a render must not
        # take the trainer's GPU lock.
        result = bundles.validate(bundle_path, ts=paths.bundle_settings())
        if not result["ok"]:
            failures = "; ".join(
                f"{c['label']}: {c['detail']}"
                for c in result["checks"] if c["status"] == "fail"
            )
            return None, f"{bundle_path} is not usable — {failures}"

        pipeline = request.POST.get("pipeline") or Render.RAW
        if pipeline not in dict(Render.PIPELINE_CHOICES):
            return None, f"Unknown pipeline {pipeline!r}."

        score_threshold = _parse_float_in_range(
            request.POST.get("score_threshold"), default=0.5, low=0.0, high=1.0)
        if score_threshold is None:
            return None, "Score threshold must be between 0 and 1."

        frame_stride = _parse_positive_int(request.POST.get("frame_stride"), default=1)
        if frame_stride is None:
            return None, "Frame stride must be a positive integer."

        numbers = {}
        for field, raw, low, high, label in [
            ("detector_expand_ratio", request.POST.get("detector_expand_ratio"),
             0.0, 10.0, "Person-box expand ratio must be between 0 and 10."),
            ("tile_width_pct", request.POST.get("tile_width_pct"),
             0.0001, 100.0, "Tile width % must be in (0, 100]."),
            ("tile_height_pct", request.POST.get("tile_height_pct"),
             0.0001, 100.0, "Tile height % must be in (0, 100]."),
            ("overlap", request.POST.get("overlap"),
             0.0, 0.99, "Tile overlap must be in [0, 1)."),
            ("merge_nms_iou", request.POST.get("merge_nms_iou"),
             0.0, 1.0, "Merge NMS IoU must be between 0 and 1."),
            ("detector_min_box_size", request.POST.get("detector_min_box_size"),
             0.0, 100000.0, "Person-crop minimum size must be 0 or more pixels."),
        ]:
            value = _parse_optional_float(raw, low, high)
            if value is _INVALID:
                return None, label
            numbers[field] = value

        tile_size_px = _parse_optional_int(request.POST.get("tile_size_px"), 1, 100000)
        if tile_size_px is _INVALID:
            return None, "Tile size must be a positive number of pixels."

        chain = [
            part.strip() for part in (request.POST.get("chain") or "").split(",")
            if part.strip()
        ]

        style, style_error = render_style.parse_form(request.POST)
        if style_error:
            return None, style_error

        # The form hides irrelevant rows with CSS, which still submits them — drop
        # the ones the chosen pipeline can't use so the saved row is an honest
        # record of what actually ran. 'chain' can contain anything, so it keeps
        # every knob.
        detector_checkpoint = (request.POST.get("detector_checkpoint") or "").strip()
        uses_detector = pipeline in pipelines.DETECTOR_PIPELINES or pipeline == pipelines.CHAIN
        uses_tiling = pipeline in (
            pipelines.BATCH_DETECT, pipelines.BATCH_PEOPLE, pipelines.CHAIN)
        if not uses_detector:
            detector_checkpoint = ""
            numbers["detector_expand_ratio"] = None
            numbers["detector_min_box_size"] = None
        if not uses_tiling:
            tile_size_px = None
            numbers["tile_width_pct"] = None
            numbers["tile_height_pct"] = None
            numbers["overlap"] = None
        # merge_nms_iou applies to every pipeline that produces several predictions
        # per frame to reconcile — that is, all of them except "raw", which runs the
        # model once on the whole frame and has nothing to merge.
        if pipeline == Render.RAW:
            numbers["merge_nms_iou"] = None
        if pipeline != pipelines.CHAIN:
            chain = []

        render = Render(
            video=video,
            bundle_path=bundle_path,
            score_threshold=score_threshold,
            frame_stride=frame_stride,
            pipeline=pipeline,
            detector_checkpoint=detector_checkpoint,
            tile_size_px=tile_size_px,
            chain=chain,
            render_style=style,
            # root= is what keeps this name unique against the studio's own
            # folder rather than the Videos tab's; see services.paths.output_root.
            output_filename=inference.unique_output_filename(
                video.name, root=paths.output_root()),
            **numbers,
        )

        # A bundle's geometry is the bundle's, not the form's: whatever was posted
        # for the pipeline fields is replaced by the manifest's own values. This is
        # the server-side half of the locked fields on the form — a stale page or a
        # bundle that changed on disk since the sync still runs what the bundle
        # says today.
        try:
            render.sync_bundle()
        except ValueError as exc:
            return None, str(exc)

        # Pre-flight the exact payload the job will send, so a missing artifact,
        # an out-of-root path, or a detector-less person pipeline fails here.
        try:
            inference.build_predict_payload(render)
        except RuntimeError as exc:
            return None, str(exc)
        return render, None

    @admin.action(description="Import every new file in the studio's videos folder")
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
            self.message_user(
                request, f"{queued} re-download job(s) queued — refresh to see progress.")

    # ------------------------------------------------------------------- urls
    def _url_name(self, suffix: str) -> str:
        """A URL name scoped to this ModelAdmin's own model.

        Derived from ``self.model`` rather than hardcoded, the same trick
        Django's own admin uses for its changelist/add/change routes and
        ``videos.admin.InferenceJobAdmin`` documents at length: the moment
        anyone subclasses this admin over a proxy model for a second tab, two
        ``path()`` entries would otherwise share one name and ``reverse()``
        would pick whichever was registered last.
        """
        opts = self.model._meta
        return f"{opts.app_label}_{opts.model_name}_{suffix}"

    def get_urls(self):
        # Custom routes first: the change view's "<path:object_id>/" catch-all
        # swallows anything registered after it.
        custom = [
            path("play/", self.admin_site.admin_view(self.play_view),
                 name=self._url_name("play")),
            path("stream/", self.admin_site.admin_view(self.stream_view),
                 name=self._url_name("stream")),
            path("preview/", self.admin_site.admin_view(self.preview_view),
                 name=self._url_name("preview")),
            path("render-preset/save/",
                 self.admin_site.admin_view(self.render_preset_save_view),
                 name=self._url_name("render_preset_save")),
            path("bundles/sync/",
                 self.admin_site.admin_view(self.bundle_sync_view),
                 name=self._url_name("bundle_sync")),
        ]
        return custom + super().get_urls()

    def bundle_sync_view(self, request):
        """Validate one of the studio's bundles and return the values it dictates.

        The same contract as ``fleetsite.admin_views.bundle_sync_view`` — POST
        ``bundle`` (+ optional ``load_test=1``), reply ``{ok, name, defaults,
        checks}`` — but aimed at this app's bundle root. The project-level
        endpoint passes no ``ts``, so it always resolves against
        ``TrainingSettings.bundles_root`` and would sync the wrong geometry here.

        A ModelAdmin route rather than a project URL, so it is mirrored onto
        ``section_site`` for free by ``AdminSectionsConfig.ready()`` — which is
        exactly the second declaration in ``admin_sections/section_urlconf.py``
        that the global "bundle-sync" name has to pay for.
        """
        if request.method != "POST":
            return JsonResponse({"error": "POST required."}, status=405)

        bundle = (request.POST.get("bundle") or "").strip()
        if not bundle:
            return JsonResponse({"error": "No bundle selected."}, status=400)
        load_test = request.POST.get("load_test") in ("1", "true", "on")

        # bundles.validate never raises for a bad bundle -- it reports. Anything
        # that escapes it is a bug or a broken trainer, and the button should say
        # so rather than the fetch failing with an opaque 500.
        try:
            return JsonResponse(bundles.validate(
                bundle, load_test=load_test, ts=paths.bundle_settings()))
        except Exception as exc:  # noqa: BLE001 - surfaced in the checklist
            return JsonResponse({"error": f"{type(exc).__name__}: {exc}"}, status=500)

    def preview_view(self, request):
        """Render one frame the way the finished render will look.

        POSTs the whole form (the template's preview button submits the same
        field names, so nothing has to be kept in sync) plus ``video``,
        ``position`` (0–1 along the clip) and optionally ``refresh=1`` to re-run
        the model rather than reuse its cached answer for that frame. Returns
        the frame as a data URI.

        POST-only and staff-only for the same reason as the bundle-sync
        endpoint: a cache miss runs the model, which takes the trainer's GPU.
        """
        if request.method != "POST":
            return JsonResponse({"error": "POST required."}, status=405)

        video = Video.objects.filter(pk=request.POST.get("video")).first()
        if video is None or not video.exists():
            return JsonResponse({"error": "No such video on disk."}, status=404)

        render, error = self._build_render(request, video)
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
                render, position, use_cache=not request.POST.get("refresh"),
                root=paths.output_root())
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
        """Save the Look section of the form as a named preset.

        POSTs the whole form plus ``preset_name`` — same "the form is the
        payload" contract as the preview endpoint. Saving by name is an
        **upsert**: pressing save again with a preset's own name is how you
        amend it. The reply is the preset in the form's own shape, so the page
        can add it to the dropdown and select it without a reload.

        These presets are this app's own; a look saved here does not appear on
        the Videos tab's Render presets, and the same name may exist on both.
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
            "title": video.name,
            "video": video,
            "exists": video.exists(),
            "stream_url": reverse(
                f"admin:{self._url_name('stream')}") + f"?video={video.pk}",
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
    """Presets are created by the render form's "Save as preset" button; this
    page is for renaming, deleting, and reading back exactly what a look is."""

    list_display = ["name", "summary", "updated_at"]
    search_fields = ["name"]
    readonly_fields = ["created_at", "updated_at"]
    ordering = ["name"]

    @admin.display(description="look")
    def summary(self, obj):
        return obj.summary()


@admin.register(Render)
class RenderAdmin(admin.ModelAdmin):
    """"Inferred videos": the read-only log of marketing renders.

    Rows are only created by the action, so there is no add page — and unlike
    ``videos.admin.InferenceJobAdmin`` there is no queryset split either, since
    every row in this table is a marketing render. Which also means
    ``get_queryset`` defers to ``super()`` and keeps ``admin_sections``' scoping
    working.
    """

    list_display = ["video", "bundle_display", "pipeline", "look", "status_badge",
                    "frames_processed", "play_link", "download_link", "created_at"]
    list_filter = ["status", "pipeline"]
    search_fields = ["video__name", "bundle_path"]
    readonly_fields = [f.name for f in Render._meta.fields]
    ordering = ["-created_at"]

    def has_add_permission(self, request):
        return False

    def get_actions(self, request):
        """Relabel the stock ``delete_selected`` so it's clear the rendered mp4
        goes with the row, not just orphaned under ``inferred/``.

        The file removal itself lives in ``marketing_studio.signals`` (a
        ``post_delete`` receiver) rather than here, so it also fires when a row
        disappears by cascade from its Video — a path that never goes through
        this ``ModelAdmin`` at all.
        """
        actions = super().get_actions(request)
        if "delete_selected" in actions:
            func, name, _desc = actions["delete_selected"]
            actions["delete_selected"] = (
                func, name,
                "Delete selected inferred videos (also deletes the rendered file)",
            )
        return actions

    @admin.display(description="status", ordering="status")
    def status_badge(self, obj):
        return _job_status_badge(obj.status)

    @admin.display(description="bundle", ordering="bundle_path")
    def bundle_display(self, obj):
        return format_html('<span title="infer bundle">{}</span>',
                           obj.bundle_path or "—")

    @admin.display(description="look")
    def look(self, obj):
        """The couple of style choices that change what the clip looks like from
        across the room — enough to tell two renders of the same clip apart."""
        style = render_style.normalize(obj.render_style)
        # The relabel earns its place here: it is the one style choice that
        # changes what the clip *says* rather than how it looks, so a row
        # rendered with one should never be mistaken for the model's own labels.
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
        # A same-origin link with a `download` attribute makes the browser save
        # the file rather than navigate to it, with no server-side attachment
        # handling — the stream endpoint already serves the raw mp4 bytes.
        if not obj.output_exists():
            return "—"
        url = reverse(f"admin:{self._url_name('stream')}") + f"?job={obj.pk}"
        return format_html(
            '<a class="button" href="{}" download="{}">⭳ download</a>',
            url, obj.output_filename,
        )

    def _url_name(self, suffix: str) -> str:
        """A URL name scoped to this ModelAdmin's own model — see
        ``VideoAdmin._url_name`` for why this is derived rather than written."""
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
        render = Render.objects.select_related("video").filter(
            pk=request.GET.get("job")).first()
        if render is None:
            raise Http404("unknown render")
        context = {
            **self.admin_site.each_context(request),
            "title": f"Inferred — {render.video.name}",
            "job": render,
            "exists": render.output_exists(),
            "stream_url": reverse(
                f"admin:{self._url_name('stream')}") + f"?job={render.pk}",
        }
        return TemplateResponse(request, "admin/videos/inference_player.html", context)

    def stream_view(self, request):
        """Stream the rendered mp4, honouring HTTP Range so the player can seek."""
        render = Render.objects.filter(pk=request.GET.get("job")).first()
        if render is None or not render.output_exists():
            raise Http404("render output not found")
        return streaming.serve(request, render.output_path())
