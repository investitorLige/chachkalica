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
from training import pipelines
from training.models import TrainedModel
from training.services import bundles, exports, pipeline_meta
from videos import jobs
from videos.models import FrameExtractionJob, InferenceJob, Video
from videos.services import downloader, frame_extraction, inference
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


# Every input on the run-inference form: the shared pipeline metadata vocabulary
# (training.services.pipeline_meta.FIELDS) plus `frame_stride`, which is a property
# of this run rather than of the model. Deriving the tuple from FIELDS rather than
# restating it means a knob added to the metadata shows up here automatically.
_FORM_FIELDS = (*pipeline_meta.FIELDS, "frame_stride")

_DEFAULT_SCORE_THRESHOLD = 0.5
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


# Sentinel for the *optional* parsers below, which must tell "operator left it
# blank" (a valid None) apart from "operator typed nonsense".
_INVALID = object()


def _parse_optional_float(raw: str, low: float, high: float):
    """Parse an optional float in ``[low, high]``; blank -> ``None``, bad -> ``_INVALID``."""
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return _INVALID
    return value if low <= value <= high else _INVALID


def _parse_optional_int(raw: str, low: int, high: int):
    """Parse an optional int in ``[low, high]``; blank -> ``None``, bad -> ``_INVALID``."""
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return _INVALID
    return value if low <= value <= high else _INVALID


def _form_values(defaults: dict, posted) -> dict:
    """The values to render the step-2 form with.

    A re-render after a validation warning keeps whatever the operator typed
    (``posted`` wins); a first render uses the selected model's pipeline metadata,
    so the page *arrives* configured the way the model was trained instead of
    blank. ``chain`` is a list in the metadata and a comma-separated string in the
    form.
    """
    if posted.get("apply"):
        return {key: posted.get(key, "") for key in _FORM_FIELDS}

    values = {key: defaults.get(key) for key in _FORM_FIELDS}
    values["chain"] = ", ".join(defaults.get("chain") or [])
    values["frame_stride"] = _DEFAULT_FRAME_STRIDE
    if values.get("score_threshold") is None:
        values["score_threshold"] = _DEFAULT_SCORE_THRESHOLD
    return {key: "" if value is None else value for key, value in values.items()}


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
    actions = ["play_video", "extract_frames", "run_inference", "import_all_new", "redownload"]

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
        if queryset.count() != 1:
            self.message_user(request, "Select exactly one video to run inference on.",
                              level=messages.WARNING)
            return None
        video = queryset.first()
        if not video.exists():
            self.message_user(request, f"{video.name}: no file on disk yet.",
                              level=messages.WARNING)
            return None

        base_context = {
            **self.admin_site.each_context(request),
            "video": video,
            "action": "run_inference",
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
                "title": f"Run model inference — {video.name}",
                "model_source_choices": InferenceJob.MODEL_SOURCE_CHOICES,
                "default_model_source": InferenceJob.TRAINED,
                "artifact_count": len(exports.list_artifacts()),
                "exports_root": str(exports.exports_root()),
                "bundle_count": len(bundles.list_bundles()),
                "bundles_root": str(bundles.bundles_root()),
            })

        # ------------------------------------------------------- step 2: submit
        if request.POST.get("apply"):
            job, error = self._build_inference_job(request, video, model_source)
            if error:
                self.message_user(request, error, level=messages.WARNING)
            else:
                job.save()
                _queue().enqueue(jobs.run_inference, job.id,
                                 job_timeout=jobs.INFERENCE_JOB_TIMEOUT)
                self.message_user(
                    request,
                    f"Inference queued for {video.name} with {job.model_label()} "
                    f"[{job.pipeline}] — see the Inferred videos tab for progress.",
                )
                return None

        # ------------------------------------------ step 2: render the full form
        # Both model lists are newest-first and the first entry is preselected, so
        # the page arrives with a model chosen *and* its pipeline metadata filled
        # in — the common case ("run this model the way it was trained") needs no
        # selection at all. Picking a different model reapplies its own metadata
        # client-side from the maps below.
        artifacts = exports.list_artifacts()
        is_exported = model_source == InferenceJob.EXPORTED
        is_bundle = model_source == InferenceJob.BUNDLE
        bundle_list = bundles.list_bundles() if is_bundle else []
        trained_models = list(
            TrainedModel.objects
            .select_related("source_run_result__run__experiment")
            .order_by("-created_at", "name")
        )
        trained_defaults = {
            str(tm.pk): pipeline_meta.for_trained_model(tm) for tm in trained_models
        }
        # An artifact with neither a ".pipeline.json" sidecar nor a catalogued model
        # behind it has nothing on record (see exports.read_pipeline_defaults) and is
        # simply left out of the map — selecting it leaves the form as it stands.
        artifact_defaults = {
            a["relpath"]: d
            for a in artifacts
            for d in [exports.read_pipeline_defaults(a["relpath"])] if d
        }

        if is_bundle:
            # No preselection and no prefill: a bundle's geometry arrives via the
            # explicit "Sync bundle" press, which is also the only thing that
            # reports whether the bundle loads here (bundles.validate). Landing on
            # a preselected bundle with its fields already filled would imply that
            # check had happened.
            selected = request.POST.get("bundle_path") or ""
            defaults = pipeline_meta.raw()
        elif is_exported:
            selected = request.POST.get("artifact_path") or (
                artifacts[0]["relpath"] if artifacts else "")
            defaults = artifact_defaults.get(selected) or pipeline_meta.raw()
        else:
            selected = request.POST.get("trained_model") or (
                str(trained_models[0].pk) if trained_models else "")
            defaults = trained_defaults.get(selected) or pipeline_meta.raw()

        context = {
            **base_context,
            "title": f"Run model inference — {video.name}",
            "model_source": model_source,
            "is_exported": is_exported,
            "is_bundle": is_bundle,
            "trained_models": trained_models,
            # Passed as dicts, not pre-serialized: the template renders them with
            # ``json_script``, which escapes the HTML-significant characters
            # ``json.dumps`` does not. Model names, artifact filenames and detector
            # paths all reach these maps, and a ``</script>`` in any of them would
            # otherwise close the script element early.
            "trained_defaults": trained_defaults,
            "artifacts": artifacts,
            "artifact_defaults": artifact_defaults,
            "bundles": bundle_list,
            "bundles_root": str(bundles.bundles_root()) if is_bundle else "",
            "bundle_sync_url": reverse("bundle-sync"),
            "selected_model": selected,
            "values": _form_values(defaults, request.POST),
            "default_score_threshold": _DEFAULT_SCORE_THRESHOLD,
            "exports_root": str(exports.exports_root()),
            "pipeline_choices": InferenceJob.PIPELINE_CHOICES,
            "detector_pipelines": " ".join(sorted(pipelines.DETECTOR_PIPELINES)),
            "tiling_pipelines": " ".join([
                pipelines.BATCH_DETECT, pipelines.BATCH_PEOPLE, pipelines.CHAIN]),
            "chain_pipelines": pipelines.CHAIN,
            # Every pipeline merges several per-frame predictions back together;
            # only "raw" has nothing to merge.
            "merging_pipelines": " ".join(
                value for value, _label in pipelines.PIPELINE_CHOICES),
        }
        return TemplateResponse(request, "admin/videos/run_inference.html", context)

    def _build_inference_job(self, request, video, model_source):
        """Validate the step-2 POST into an unsaved :class:`InferenceJob`.

        Returns ``(job, None)`` on success or ``(None, message)`` on the first
        problem found, so the caller can re-render the form with a warning.
        """
        trained_model = None
        artifact_path = ""
        bundle_path = ""
        if model_source == InferenceJob.EXPORTED:
            artifact_path = (request.POST.get("artifact_path") or "").strip()
            if not artifact_path:
                return None, "Choose an exported artifact."
        elif model_source == InferenceJob.BUNDLE:
            bundle_path = (request.POST.get("bundle_path") or "").strip()
            if not bundle_path:
                return None, "Choose a bundle."
            # Structural checks only — the load test belongs to the "Sync bundle"
            # button, where the operator asked for it; submitting a job must not
            # take the trainer's GPU lock.
            result = bundles.validate(bundle_path)
            if not result["ok"]:
                failures = "; ".join(
                    f"{c['label']}: {c['detail']}"
                    for c in result["checks"] if c["status"] == "fail"
                )
                return None, f"{bundle_path} is not usable — {failures}"
        else:
            trained_model = TrainedModel.objects.filter(
                pk=request.POST.get("trained_model") or None
            ).first()
            if trained_model is None:
                return None, "Choose a trained model."

        pipeline = request.POST.get("pipeline") or InferenceJob.RAW
        if pipeline not in dict(InferenceJob.PIPELINE_CHOICES):
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
        if pipeline == InferenceJob.RAW:
            numbers["merge_nms_iou"] = None
        if pipeline != pipelines.CHAIN:
            chain = []

        job = InferenceJob(
            video=video,
            model_source=model_source,
            trained_model=trained_model,
            artifact_path=artifact_path,
            bundle_path=bundle_path,
            score_threshold=score_threshold,
            frame_stride=frame_stride,
            pipeline=pipeline,
            detector_checkpoint=detector_checkpoint,
            tile_size_px=tile_size_px,
            chain=chain,
            output_filename=inference.unique_output_filename(video.name),
            **numbers,
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


@admin.register(InferenceJob)
class InferenceJobAdmin(admin.ModelAdmin):
    """Read-only log of "Run model inference…" runs — rows are only created by the action."""

    list_display = ["video", "model_display", "pipeline", "status_badge",
                     "frames_processed", "play_link", "created_at"]
    list_filter = ["status", "model_source", "pipeline", "trained_model"]
    search_fields = ["video__name", "trained_model__name", "artifact_path",
                     "bundle_path"]
    readonly_fields = [f.name for f in InferenceJob._meta.fields]
    ordering = ["-created_at"]

    def has_add_permission(self, request):
        return False

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

    @admin.display(description="")
    def play_link(self, obj):
        if not obj.output_exists():
            return "—"
        url = reverse("admin:videos_inferencejob_play") + f"?job={obj.pk}"
        return format_html('<a class="button" href="{}">▶ play</a>', url)

    def get_urls(self):
        custom = [
            path("play/", self.admin_site.admin_view(self.play_view),
                 name="videos_inferencejob_play"),
            path("stream/", self.admin_site.admin_view(self.stream_view),
                 name="videos_inferencejob_stream"),
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
            "stream_url": reverse("admin:videos_inferencejob_stream") + f"?job={job.pk}",
        }
        return TemplateResponse(request, "admin/videos/inference_player.html", context)

    def stream_view(self, request):
        """Stream the annotated mp4, honouring HTTP Range so the player can seek."""
        job = InferenceJob.objects.filter(pk=request.GET.get("job")).first()
        if job is None or not job.output_exists():
            raise Http404("inference output not found")
        path = job.output_path()
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
