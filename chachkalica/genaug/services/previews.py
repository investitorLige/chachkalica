"""Studio previews: pick the preview images, run a prompt over them, re-check."""

from __future__ import annotations

import random

from genaug.models import AugPreview, AugSession, AugSessionPrompt, Status
from genaug.services import generate, sources


def pick_preview_images(session: AugSession, *, shuffle_seed: int | None = None) -> list[str]:
    """Choose ``preview_count`` labeled images with at least one box.

    Default is an even stride over the sorted dataset (a sorted dataset's first
    N images are usually one scene); ``shuffle_seed`` draws a random set
    instead, for the studio's "other images" button. Images with boxes are
    preferred, since an image with nothing annotated cannot show whether the
    edit kept the labels right.
    """
    images = sources.list_images(session.dataset)
    labels = session_labels_dir(session)
    boxed = []
    for image in images:
        objects = sources.read_label_objects(labels, image.name) if labels else None
        if objects:
            boxed.append(image.name)
    pool = boxed or [image.name for image in images]
    count = min(session.preview_count, len(pool))
    if count == 0:
        return []
    if shuffle_seed is not None:
        return sorted(random.Random(shuffle_seed).sample(pool, count))
    step = len(pool) / count
    return [pool[int(i * step)] for i in range(count)]


def session_labels_dir(session: AugSession):
    return sources.labels_dir(session.dataset, session.label_source, session.annotator,
                              session.explicit_labels_path)


def create_prompt(session: AugSession, *, name, text, negative_prompt="", seed=42,
                  num_inference_steps=40, true_cfg_scale=4.0, preset=None,
                  editor_config=None) -> AugSessionPrompt:
    prompt = AugSessionPrompt.objects.create(
        session=session, preset=preset, name=name, text=text,
        negative_prompt=negative_prompt, seed=seed,
        num_inference_steps=num_inference_steps, true_cfg_scale=true_cfg_scale,
        editor_config=editor_config or session.editor_config(),
    )
    AugPreview.objects.bulk_create([
        AugPreview(prompt=prompt, source_filename=name_)
        for name_ in session.preview_images
    ])
    return prompt


def run_prompt(prompt: AugSessionPrompt) -> dict:
    """Generate and validate every preview of one prompt. Errors stay per image."""
    session = prompt.session
    labels = session_labels_dir(session)
    image_root = sources.image_dir(session.dataset)
    settings = session.validation_settings()
    generation = prompt.generation()
    done = errors = 0

    for preview in prompt.previews.all():
        preview.status = Status.RUNNING
        preview.save(update_fields=["status"])
        try:
            source = image_root / preview.source_filename
            path, _cached, edit_ms = generate.generate(
                source, generation, prompt.editor_config, prompt.seed)
            objects = sources.read_label_objects(labels, preview.source_filename) or []
            verdict = generate.validate(source, path, objects, settings)
        except Exception as exc:  # noqa: BLE001 - one bad image must not end the prompt
            preview.status, preview.error = Status.ERROR, str(exc)
            preview.save(update_fields=["status", "error"])
            errors += 1
            continue
        preview.output_relpath = str(path.relative_to(sources.genaug_root()))
        preview.validation, preview.accepted = verdict, bool(verdict["valid"])
        preview.edit_ms = edit_ms
        preview.status, preview.error = Status.OK, ""
        preview.save(update_fields=["output_relpath", "validation", "accepted", "edit_ms",
                                    "status", "error"])
        done += 1
    return {"done": done, "errors": errors}


def revalidate(session: AugSession) -> int:
    """Re-judge every finished preview under the session's current thresholds."""
    labels = session_labels_dir(session)
    image_root = sources.image_dir(session.dataset)
    settings = session.validation_settings()
    count = 0
    previews = AugPreview.objects.filter(prompt__session=session, status=Status.OK)
    for preview in previews.exclude(output_relpath=""):
        variant = sources.genaug_root() / preview.output_relpath
        if not variant.is_file():
            continue
        objects = sources.read_label_objects(labels, preview.source_filename) or []
        verdict = generate.validate(image_root / preview.source_filename, variant, objects, settings)
        preview.validation, preview.accepted = verdict, bool(verdict["valid"])
        preview.save(update_fields=["validation", "accepted"])
        count += 1
    return count
