"""The "Generative augmentation" admin section.

Three tabs — Prompt library, Studio, Dataset builds (see :mod:`genaug.models`).
The Studio tab's rows each open one studio page, which is where the work
happens: prompt, look at the previews, keep what works, build.

Like the VLM section, the live parts poll a small JSON endpoint rather than
using websockets — this project is WSGI-only.
"""

from __future__ import annotations

import re
from pathlib import Path

import django_rq
from django.contrib import admin, messages
from django.http import FileResponse, Http404, HttpResponse, JsonResponse
from django.shortcuts import redirect
from django.template.response import TemplateResponse
from django.urls import path, reverse
from django.utils.html import format_html

from fleet.services import datasets as datasets_svc
from genaug import jobs
from genaug.models import (
    AugBuild, AugPreview, AugPrompt, AugSession, AugSessionPrompt, Status, TRANSFORMER_CHOICES,
    EDITOR_CHOICES,
)
from genaug.services import backend, builder, previews, sources
from vlm.services import label_render

_STATUS_COLORS = {
    "ok": "#22c55e", "running": "#3b82f6", "queued": "#9ca3af",
    "cancel_requested": "#f59e0b", "cancelled": "#9ca3af", "error": "#ef4444",
}


def _queue():
    return django_rq.get_queue("default")


def _badge(value: str):
    return format_html(
        '<span style="background:{};color:#fff;padding:2px 8px;border-radius:9px;'
        'font-size:11px">{}</span>', _STATUS_COLORS.get(value, "#9ca3af"), value)


def _float(raw, default, lo=None, hi=None):
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    if lo is not None and value < lo:
        return default
    if hi is not None and value > hi:
        return default
    return value


def _int(raw, default, lo=None, hi=None):
    value = _float(raw, None)
    if value is None:
        return default
    value = int(value)
    if (lo is not None and value < lo) or (hi is not None and value > hi):
        return default
    return value


def _slug(raw: str, fallback: str = "prompt") -> str:
    slug = re.sub(r"[^a-z0-9_]+", "_", (raw or "").lower()).strip("_")
    return slug[:64] or fallback


def _image_response(image_path: Path, shapes):
    if not image_path.is_file():
        raise Http404("image not found")
    if not shapes:
        return FileResponse(open(image_path, "rb"))
    try:
        return HttpResponse(label_render.render(image_path, shapes), content_type="image/jpeg")
    except RuntimeError as exc:
        raise Http404(str(exc))


@admin.register(AugPrompt)
class AugPromptAdmin(admin.ModelAdmin):
    list_display = ["name", "description", "builtin", "short_text", "updated_at"]
    search_fields = ["name", "description", "text"]
    readonly_fields = ["builtin"]

    @admin.display(description="prompt")
    def short_text(self, obj):
        text = " ".join(obj.text.split())
        return text[:120] + ("…" if len(text) > 120 else "")


@admin.register(AugSession)
class AugSessionAdmin(admin.ModelAdmin):
    list_display = ["name", "dataset", "label_source", "prompt_count", "kept_count",
                    "updated_at", "studio_link"]
    autocomplete_fields = ["dataset", "annotator"]
    fields = ["name", "dataset", "label_source", "annotator", "explicit_labels_path",
              "preview_count", "editor", "transformer", "lightning"]

    @admin.display(description="prompts")
    def prompt_count(self, obj):
        return obj.prompts.count()

    @admin.display(description="kept")
    def kept_count(self, obj):
        return obj.prompts.filter(keep=True).count()

    @admin.display(description="")
    def studio_link(self, obj):
        return format_html('<a class="button" href="{}">Open studio</a>',
                           self._studio_url(obj))

    @staticmethod
    def _studio_url(session, anchor: str = "") -> str:
        url = reverse("admin:genaug_augsession_studio") + f"?session={session.pk}"
        return url + (f"#{anchor}" if anchor else "")

    def save_model(self, request, obj, form, change):
        super().save_model(request, obj, form, change)
        if not obj.preview_images or "dataset" in form.changed_data \
                or "preview_count" in form.changed_data or "label_source" in form.changed_data:
            obj.preview_images = previews.pick_preview_images(obj)
            obj.save(update_fields=["preview_images", "updated_at"])

    def response_add(self, request, obj, post_url_continue=None):
        return redirect(self._studio_url(obj))

    # -- studio ------------------------------------------------------------

    def get_urls(self):
        view = self.admin_site.admin_view
        custom = [
            path("studio/", view(self.studio_view), name="genaug_augsession_studio"),
            path("studio/state/", view(self.state_view), name="genaug_augsession_state"),
            path("studio/generate/", view(self.generate_view), name="genaug_augsession_generate"),
            path("studio/prompt/", view(self.prompt_action_view), name="genaug_augsession_prompt"),
            path("studio/settings/", view(self.settings_view), name="genaug_augsession_settings"),
            path("studio/shuffle/", view(self.shuffle_view), name="genaug_augsession_shuffle"),
            path("studio/build/", view(self.build_view), name="genaug_augsession_build"),
            path("studio/image/", view(self.image_view), name="genaug_augsession_image"),
            path("studio/backend/", view(self.backend_view), name="genaug_augsession_backend"),
        ]
        return custom + super().get_urls()

    def _get_session(self, request) -> AugSession:
        session = AugSession.objects.select_related("dataset", "annotator").filter(
            pk=request.GET.get("session") or request.POST.get("session")).first()
        if session is None:
            raise Http404("unknown studio session")
        return session

    def studio_view(self, request):
        session = self._get_session(request)
        classes = sources.read_classes(session.dataset)
        prompts = list(session.prompts.prefetch_related("previews"))
        kept = [p for p in prompts if p.keep]
        edit_times = [pv.edit_ms for p in prompts for pv in p.previews.all() if pv.edit_ms]
        n_images = len(sources.list_images(session.dataset))
        context = {
            **self.admin_site.each_context(request),
            "title": f"Studio — {session.name}",
            "session": session,
            "prompts": prompts,
            "kept": kept,
            "presets": AugPrompt.objects.all(),
            "presets_data": {str(p.pk): {"name": p.name, "text": p.text,
                                         "negative": p.negative_prompt}
                             for p in AugPrompt.objects.all()},
            "classes": classes,
            "n_images": n_images,
            "avg_edit_s": round(sum(edit_times) / len(edit_times) / 1000, 1) if edit_times else None,
            "editor_choices": EDITOR_CHOICES,
            "transformer_choices": TRANSFORMER_CHOICES,
            "default_output_name": f"{session.dataset.name}__genaug_{session.pk}",
            "background_key": builder.BACKGROUND,
            "urls": {
                "state": reverse("admin:genaug_augsession_state") + f"?session={session.pk}",
                "generate": reverse("admin:genaug_augsession_generate"),
                "prompt": reverse("admin:genaug_augsession_prompt"),
                "settings": reverse("admin:genaug_augsession_settings"),
                "shuffle": reverse("admin:genaug_augsession_shuffle"),
                "build": reverse("admin:genaug_augsession_build"),
                "image": reverse("admin:genaug_augsession_image"),
                "backend": reverse("admin:genaug_augsession_backend"),
                "edit": reverse("admin:genaug_augsession_change", args=[session.pk]),
                "builds": reverse("admin:genaug_augbuild_changelist"),
            },
        }
        return TemplateResponse(request, "admin/genaug/studio.html", context)

    def state_view(self, request):
        """Every prompt's and preview's status, for the studio's poll."""
        session = self._get_session(request)
        image_url = reverse("admin:genaug_augsession_image")
        data = []
        for prompt in session.prompts.prefetch_related("previews"):
            data.append({
                "id": prompt.pk, "status": prompt.status, "error": prompt.last_error,
                "keep": prompt.keep,
                "previews": [{
                    "id": pv.pk, "status": pv.status, "error": pv.error,
                    "accepted": pv.accepted, "validation": pv.validation,
                    "edit_ms": pv.edit_ms,
                    "url": f"{image_url}?preview={pv.pk}" if pv.output_relpath else None,
                } for pv in prompt.previews.all()],
            })
        return JsonResponse({"prompts": data})

    def generate_view(self, request):
        if request.method != "POST":
            raise Http404("POST only")
        session = self._get_session(request)
        text = (request.POST.get("text") or "").strip()
        if not text:
            self.message_user(request, "Write a prompt (or pick a preset) first.",
                              level=messages.WARNING)
            return redirect(self._studio_url(session))
        if not session.preview_images:
            self.message_user(request, "This dataset has no images to preview on.",
                              level=messages.ERROR)
            return redirect(self._studio_url(session))
        preset = AugPrompt.objects.filter(pk=request.POST.get("preset") or None).first()
        editor = request.POST.get("editor") or session.editor
        editor_config = {
            "editor": editor if editor in dict(EDITOR_CHOICES) else session.editor,
            "transformer": request.POST.get("transformer") or session.transformer,
            "lightning": bool(request.POST.get("lightning")),
        }
        prompt = previews.create_prompt(
            session,
            name=_slug(request.POST.get("name") or (preset.name if preset else "")),
            text=text,
            negative_prompt=(request.POST.get("negative_prompt") or "").strip(),
            seed=_int(request.POST.get("seed"), 42),
            num_inference_steps=_int(request.POST.get("num_inference_steps"), 40, 1, 100),
            true_cfg_scale=_float(request.POST.get("true_cfg_scale"), 4.0, 0.0, 20.0),
            preset=preset, editor_config=editor_config,
        )
        _queue().enqueue(jobs.run_prompt_previews, prompt.pk,
                         job_timeout=jobs.PREVIEW_JOB_TIMEOUT)
        return redirect(self._studio_url(session, f"p{prompt.pk}"))

    def prompt_action_view(self, request):
        """keep / drop / delete / reroll / save a studio prompt."""
        if request.method != "POST":
            raise Http404("POST only")
        prompt = AugSessionPrompt.objects.select_related("session").filter(
            pk=request.POST.get("prompt")).first()
        if prompt is None:
            raise Http404("unknown prompt")
        session, action = prompt.session, request.POST.get("action")
        anchor = f"p{prompt.pk}"

        if action in ("keep", "drop"):
            prompt.keep = action == "keep"
            prompt.save(update_fields=["keep"])
            if request.headers.get("x-requested-with") == "fetch":
                return JsonResponse({"keep": prompt.keep})
        elif action == "delete":
            prompt.delete()
            anchor = ""
        elif action == "reroll":
            new = previews.create_prompt(
                session, name=prompt.name, text=prompt.text,
                negative_prompt=prompt.negative_prompt, seed=prompt.seed + 1,
                num_inference_steps=prompt.num_inference_steps,
                true_cfg_scale=prompt.true_cfg_scale, preset=prompt.preset,
                editor_config=prompt.editor_config,
            )
            _queue().enqueue(jobs.run_prompt_previews, new.pk,
                             job_timeout=jobs.PREVIEW_JOB_TIMEOUT)
            anchor = f"p{new.pk}"
        elif action == "save_preset":
            name = _slug(request.POST.get("preset_name") or prompt.name)
            preset, created = AugPrompt.objects.update_or_create(
                name=name, defaults={"text": prompt.text,
                                     "negative_prompt": prompt.negative_prompt})
            self.message_user(request, f"{'Saved' if created else 'Updated'} prompt "
                                       f"'{preset.name}' in the Prompt library.")
        else:
            raise Http404("unknown action")
        return redirect(self._studio_url(session, anchor))

    def settings_view(self, request):
        """Update the session's editor defaults and thresholds, then re-check."""
        if request.method != "POST":
            raise Http404("POST only")
        session = self._get_session(request)
        session.validation_enabled = bool(request.POST.get("validation_enabled"))
        session.max_bbox_drift = _float(request.POST.get("max_bbox_drift"), session.max_bbox_drift, 0, 1)
        session.min_box_similarity = _float(request.POST.get("min_box_similarity"),
                                            session.min_box_similarity, -1, 1)
        session.max_global_shift = _float(request.POST.get("max_global_shift"),
                                          session.max_global_shift, 0, 1)
        session.save()
        count = previews.revalidate(session)
        self.message_user(request, f"Thresholds saved; re-checked {count} preview(s) — no GPU needed.")
        return redirect(self._studio_url(session))

    def shuffle_view(self, request):
        if request.method != "POST":
            raise Http404("POST only")
        session = self._get_session(request)
        import random

        session.preview_images = previews.pick_preview_images(
            session, shuffle_seed=random.randrange(1 << 30))
        session.save(update_fields=["preview_images", "updated_at"])
        self.message_user(request, "New preview images picked. Prompts you generate from "
                                   "now on use them; earlier prompts keep theirs.")
        return redirect(self._studio_url(session))

    def build_view(self, request):
        """Queue a dataset build over this session's kept prompts."""
        if request.method != "POST":
            raise Http404("POST only")
        session = self._get_session(request)
        kept = list(session.prompts.filter(keep=True).order_by("created_at"))
        if not kept:
            self.message_user(request, "Keep at least one prompt before building.",
                              level=messages.WARNING)
            return redirect(self._studio_url(session))
        output_name = (request.POST.get("output_name") or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9_.\-]+", output_name or ""):
            self.message_user(request, "Output dataset name may only use letters, digits, "
                                       "'_', '-' and '.'.", level=messages.WARNING)
            return redirect(self._studio_url(session))
        if output_name == session.dataset.name:
            self.message_user(request, "The output must be a new dataset, not the source.",
                              level=messages.ERROR)
            return redirect(self._studio_url(session))
        existing = sources.dataset_dir(session.dataset).parent / output_name
        reusing = AugBuild.objects.filter(output_name=output_name).exists()
        if existing.exists() and not reusing:
            self.message_user(request, f"{existing} already exists and was not made by a "
                                       f"build — pick another name.", level=messages.ERROR)
            return redirect(self._studio_url(session))

        editors = {p.editor_config.get("editor") for p in kept}
        configs = {tuple(sorted(p.editor_config.items())) for p in kept}
        if len(configs) > 1:
            self.message_user(request, "Kept prompts use different editor settings "
                                       f"({', '.join(sorted(editors))}…). A build runs one "
                                       "editor — drop the odd ones out first.",
                              level=messages.WARNING)
            return redirect(self._studio_url(session))
        editor_config = kept[0].editor_config

        class_sampling = {}
        for name in sources.read_classes(session.dataset) + [builder.BACKGROUND]:
            raw = request.POST.get(f"class__{name}")
            if raw not in (None, ""):
                class_sampling[name] = _float(raw, 1.0, 0.0)

        # Kept prompts can share a name (a re-roll keeps it); make them unique
        # so generated file names stay distinct and traceable.
        seen, snapshot = {}, []
        for prompt in kept:
            gen = prompt.generation()
            count = seen.get(gen["name"], 0)
            seen[gen["name"]] = count + 1
            if count:
                gen["name"] = f"{gen['name']}_{count + 1}"
            snapshot.append(gen)

        build = AugBuild.objects.create(
            session=session, source_dataset=session.dataset,
            source_labels_dir=str(previews.session_labels_dir(session)),
            output_name=output_name, prompts_snapshot=snapshot,
            fraction=_float(request.POST.get("fraction"), 0.3, 0.0, 1.0),
            variants_per_image=_int(request.POST.get("variants_per_image"), 2, 1, 20),
            class_sampling=class_sampling,
            seed=_int(request.POST.get("seed"), 42),
            include_originals=bool(request.POST.get("include_originals")),
            force=bool(request.POST.get("force")),
            editor=editor_config.get("editor", "firered"),
            transformer=editor_config.get("transformer", "q4_k_m"),
            lightning=bool(editor_config.get("lightning")),
            validation_enabled=session.validation_enabled,
            max_bbox_drift=session.max_bbox_drift,
            min_box_similarity=session.min_box_similarity,
            max_global_shift=session.max_global_shift,
        )
        _queue().enqueue(jobs.run_build, build.pk, job_timeout=jobs.BUILD_JOB_TIMEOUT)
        self.message_user(request, f"Build #{build.pk} queued → dataset '{output_name}'.")
        return redirect(reverse("admin:genaug_augbuild_report") + f"?build={build.pk}")

    def image_view(self, request):
        """A preview's generated image, or a session source image; ``boxes=1`` burns labels on.

        The generated image gets the *source's* boxes drawn on it on purpose:
        the question a preview answers is whether those boxes still fit.
        """
        boxes = bool(request.GET.get("boxes"))
        if request.GET.get("preview"):
            preview = AugPreview.objects.select_related(
                "prompt__session__dataset", "prompt__session__annotator").filter(
                pk=request.GET.get("preview")).first()
            if preview is None or not preview.output_relpath:
                raise Http404("no such preview")
            session = preview.prompt.session
            image = sources.genaug_root() / preview.output_relpath
            source_name = preview.source_filename
        else:
            session = self._get_session(request)
            source_name = Path(request.GET.get("source") or "").name
            image = sources.image_dir(session.dataset) / source_name
        shapes = []
        if boxes:
            shapes = datasets_svc.label_shapes(previews.session_labels_dir(session),
                                               source_name, sources.read_classes(session.dataset))
        return _image_response(image, shapes)

    def backend_view(self, request):
        try:
            return JsonResponse({"ok": True, **backend.health()})
        except Exception as exc:  # noqa: BLE001 - reported to the page, not raised
            return JsonResponse({"ok": False, "error": str(exc)})


@admin.register(AugBuild)
class AugBuildAdmin(admin.ModelAdmin):
    list_display = ["id", "output_name", "source_dataset", "status_badge", "progress",
                    "acceptance", "created_at", "report_link"]
    list_filter = ["status"]

    def has_add_permission(self, request):
        return False  # builds start from a studio

    def has_change_permission(self, request, obj=None):
        return False

    @admin.display(description="status")
    def status_badge(self, obj):
        return _badge(obj.status)

    @admin.display(description="variants")
    def progress(self, obj):
        return f"{obj.variants_attempted} / {obj.variants_planned}"

    @admin.display(description="accepted")
    def acceptance(self, obj):
        judged = obj.variants_accepted + obj.variants_rejected
        return f"{100 * obj.variants_accepted / judged:.0f}%" if judged else "—"

    @admin.display(description="")
    def report_link(self, obj):
        return format_html('<a class="button" href="{}?build={}">Report</a>',
                           reverse("admin:genaug_augbuild_report"), obj.pk)

    def get_urls(self):
        view = self.admin_site.admin_view
        custom = [
            path("report/", view(self.report_view), name="genaug_augbuild_report"),
            path("report/progress/", view(self.progress_view), name="genaug_augbuild_progress"),
            path("report/cancel/", view(self.cancel_view), name="genaug_augbuild_cancel"),
            path("report/image/", view(self.image_view), name="genaug_augbuild_image"),
        ]
        return custom + super().get_urls()

    def _get_build(self, request) -> AugBuild:
        build = AugBuild.objects.select_related("source_dataset", "output_dataset", "session").filter(
            pk=request.GET.get("build") or request.POST.get("build")).first()
        if build is None:
            raise Http404("unknown build")
        return build

    def _counters(self, build: AugBuild) -> dict:
        judged = build.variants_accepted + build.variants_rejected
        return {
            "status": build.status, "terminal": build.is_terminal(),
            "last_error": build.last_error, "waiting_reason": build.waiting_reason,
            "images_total": build.images_total, "images_selected": build.images_selected,
            "variants_planned": build.variants_planned,
            "variants_attempted": build.variants_attempted,
            "variants_accepted": build.variants_accepted,
            "variants_rejected": build.variants_rejected,
            "variants_errored": build.variants_errored,
            "variants_cached": build.variants_cached,
            "acceptance_rate": round(100 * build.variants_accepted / judged, 2) if judged else None,
        }

    def report_view(self, request):
        build = self._get_build(request)
        records = builder.read_manifest_tail(build, 48)
        context = {
            **self.admin_site.each_context(request),
            "title": f"Build — {build.output_name}",
            "build": build,
            "counters": self._counters(build),
            "summary": jobs.summary(build),
            "records": records,
            "image_url": reverse("admin:genaug_augbuild_image") + f"?build={build.pk}",
            "progress_url": reverse("admin:genaug_augbuild_progress") + f"?build={build.pk}",
            "cancel_url": reverse("admin:genaug_augbuild_cancel"),
            "studio_url": (reverse("admin:genaug_augsession_studio") + f"?session={build.session_id}"
                           if build.session_id else None),
            "dataset_url": (reverse("admin:fleet_dataset_change", args=[build.output_dataset_id])
                            if build.output_dataset_id else None),
        }
        return TemplateResponse(request, "admin/genaug/build_report.html", context)

    def progress_view(self, request):
        return JsonResponse(self._counters(self._get_build(request)))

    def cancel_view(self, request):
        if request.method != "POST":
            raise Http404("POST only")
        build = self._get_build(request)
        if not build.is_terminal():
            build.status = Status.CANCEL_REQUESTED
            build.save(update_fields=["status"])
            self.message_user(request, "Stopping after the current image. Everything "
                                       "accepted so far stays in the dataset.")
        return redirect(reverse("admin:genaug_augbuild_report") + f"?build={build.pk}")

    def image_view(self, request):
        """A manifest record's source or cached variant, optionally with boxes."""
        build = self._get_build(request)
        name = Path(request.GET.get("source") or "").name
        kind = request.GET.get("kind", "variant")
        record = next((r for r in builder.read_manifest_tail(build, 10_000)
                       if r.get("source") == name and r.get("generated") == request.GET.get("generated")),
                      None)
        if record is None:
            raise Http404("not in this build's manifest")
        if kind == "source":
            image = sources.image_dir(build.source_dataset) / name
        else:
            cache = Path(record.get("cache") or "")
            root = sources.genaug_root().resolve()
            try:
                cache.resolve().relative_to(root)
            except ValueError:
                raise Http404("variant outside the genaug cache")
            image = cache
        shapes = []
        if request.GET.get("boxes"):
            shapes = datasets_svc.label_shapes(Path(build.source_labels_dir), name,
                                               sources.read_classes(build.source_dataset))
        return _image_response(image, shapes)
