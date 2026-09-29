"""Build an augmented dataset: originals + validated generated variants.

Output is an ordinary dataset under the source root, so an experiment simply
picks it as a train dataset (label source "source labels")::

    <source_root>/<output_name>/
        images/                 originals (hard-linked) + accepted variants
        labels/                 one YOLO .txt per image, same stem
        classes.txt             copied verbatim from the source dataset
        genaug_manifest.jsonl   one record per attempted variant
        genaug_build.json       the build's full configuration

**Why originals are included.** An experiment pairs every model with every
train dataset as a *separate* run, so a variants-only dataset could not be
trained on alongside its source — it would be a second run, not more data.
Hard links make the copy free on disk.

**Selection.** ``fraction`` of the labeled images are augmented, drawn by
weighted sampling without replacement. An image's weight comes from the
selection scorers (:data:`SCORERS`); today that is class-aware sampling — the
largest ``class_sampling`` weight among the classes it contains, or the
``__background__`` weight for a labeled-empty image. A scorer is a function
``(objects, class_names, config) -> weight``, so preferring small objects,
rare classes or known false negatives later is one more function.

**Leakage.** Only the dataset the build is pointed at is read; nothing is
generated from any other split. The build records its source, and
``training.services.config_gen`` refuses an experiment that trains on a build
while validating or testing on that build's source.
"""

from __future__ import annotations

import json
import os
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path


from fleet.models import Dataset
from fleet.services.paths import source_root
from genaug.models import AugBuild, Status
from genaug.services import backend, generate, sources

MANIFEST = "genaug_manifest.jsonl"
BUILD_CONFIG = "genaug_build.json"
BACKGROUND = "__background__"
_JPEG_QUALITY = 95


def class_weight(objects: list[dict], class_names: list[str], config: dict) -> float:
    sampling = config.get("class_sampling") or {}
    if not objects:
        return float(sampling.get(BACKGROUND, 1.0))
    weights = []
    for obj in objects:
        cid = obj["class_id"]
        name = class_names[cid] if 0 <= cid < len(class_names) else str(cid)
        weights.append(float(sampling.get(name, 1.0)))
    return max(weights)


#: Selection scorers, multiplied together. See the module docstring.
SCORERS = [class_weight]


def select_images(candidates: list[tuple[str, list[dict]]], class_names: list[str],
                  config: dict, fraction: float, seed: int) -> list[str]:
    """Weighted sample of ``round(fraction * len(candidates))`` image names.

    Efraimidis–Spirakis: each image draws ``u ** (1 / w)`` and the largest keys
    win, which is weighted sampling without replacement in one pass. A weight
    of 0 excludes an image outright.
    """
    rng = random.Random(seed)
    keyed = []
    for name, objects in candidates:
        weight = 1.0
        for scorer in SCORERS:
            weight *= scorer(objects, class_names, config)
        if weight > 0:
            keyed.append((rng.random() ** (1.0 / weight), name))
    count = min(len(keyed), round(fraction * len(candidates)))
    keyed.sort(reverse=True)
    return sorted(name for _key, name in keyed[:count])


def assign_prompts(prompts: list[dict], variants: int, rng: random.Random) -> list[tuple[dict, int]]:
    """``(prompt, seed)`` for each of an image's variants.

    Distinct prompts first (a random subset when there are more prompts than
    variants); when asked for more variants than there are prompts, prompts
    repeat with the seed bumped, so every variant is still a different image
    and still has a deterministic name.
    """
    order = list(prompts)
    rng.shuffle(order)
    assigned = []
    for index in range(variants):
        prompt = order[index % len(order)]
        assigned.append((prompt, int(prompt["seed"]) + index // len(order)))
    return assigned


def variant_stem(source_stem: str, editor: str, prompt_name: str, seed: int) -> str:
    return f"{source_stem}__{editor}_{prompt_name}__seed{seed}"


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def run_build(build: AugBuild) -> dict:
    """Generate the dataset for one :class:`AugBuild`. Returns the final counters."""
    dataset = build.source_dataset
    out_dir = source_root() / build.output_name
    images_out, labels_out = out_dir / "images", out_dir / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    labels_in = Path(build.source_labels_dir)
    class_names = sources.read_classes(dataset)
    classes_file = sources.dataset_dir(dataset) / "classes.txt"
    if classes_file.is_file():
        shutil.copyfile(classes_file, out_dir / "classes.txt")

    editor_config = build.editor_config()
    settings = build.validation_settings()
    config = {"class_sampling": build.class_sampling}
    _write_build_config(build, out_dir)

    # -- originals + candidates -----------------------------------------
    images = sources.list_images(dataset)
    candidates = []
    for image in images:
        objects = sources.read_label_objects(labels_in, image.name)
        if build.include_originals:
            _link_or_copy(image, images_out / image.name)
            if objects is not None:
                shutil.copyfile(sources.label_file(labels_in, image.name),
                                labels_out / f"{image.stem}.txt")
        if objects is not None:
            candidates.append((image.name, objects))
    by_name = {name: objects for name, objects in candidates}

    selected = select_images(candidates, class_names, config, build.fraction, build.seed)
    AugBuild.objects.filter(pk=build.pk).update(
        images_total=len(images), images_selected=len(selected),
        variants_planned=len(selected) * build.variants_per_image,
    )

    # -- variants --------------------------------------------------------
    rng = random.Random(build.seed)
    counters = {"variants_attempted": 0, "variants_accepted": 0, "variants_rejected": 0,
                "variants_errored": 0, "variants_cached": 0}
    image_root = sources.image_dir(dataset)
    cancelled = False

    def should_stop() -> bool:
        return AugBuild.objects.filter(pk=build.pk, status=Status.CANCEL_REQUESTED).exists()

    def on_busy(message: str) -> None:
        AugBuild.objects.filter(pk=build.pk).update(waiting_reason=message[:500])

    with open(out_dir / MANIFEST, "a", encoding="utf-8") as manifest:
        for name in selected:
            if should_stop():
                cancelled = True
                break
            source = image_root / name
            label_path = sources.label_file(labels_in, name)
            for prompt, seed in assign_prompts(build.prompts_snapshot, build.variants_per_image, rng):
                stem = variant_stem(source.stem, editor_config["editor"], prompt["name"], seed)
                record = {
                    "build_id": build.pk, "split": "train",
                    "source": name, "source_path": str(source),
                    "source_label": str(label_path) if label_path else None,
                    "generated": f"{stem}.jpg", "augmentation": prompt["name"],
                    "prompt": prompt["text"], "negative_prompt": prompt.get("negative_prompt", ""),
                    "seed": seed, "editor": editor_config,
                    "params": {"num_inference_steps": prompt.get("num_inference_steps"),
                               "true_cfg_scale": prompt.get("true_cfg_scale")},
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                counters["variants_attempted"] += 1
                try:
                    variant, cached, edit_ms = generate.generate(
                        source, prompt, editor_config, seed, force=build.force,
                        on_busy=on_busy, should_stop=should_stop)
                    verdict = generate.validate(source, variant, by_name[name], settings)
                except backend.GpuBusy:
                    cancelled = True  # the wait was abandoned by a cancel
                    break
                except Exception as exc:  # noqa: BLE001 - log it, keep going
                    record.update(accepted=False, error=f"{type(exc).__name__}: {exc}")
                    counters["variants_errored"] += 1
                    manifest.write(json.dumps(record) + "\n")
                    manifest.flush()
                    _save_counters(build, counters)
                    continue
                counters["variants_cached"] += int(cached)
                record.update(cache=str(variant), cached=cached, edit_ms=edit_ms,
                              validation=verdict, accepted=bool(verdict["valid"]))
                if verdict["valid"]:
                    _write_variant(variant, images_out / f"{stem}.jpg")
                    if label_path is not None:
                        shutil.copyfile(label_path, labels_out / f"{stem}.txt")
                    counters["variants_accepted"] += 1
                else:
                    counters["variants_rejected"] += 1
                manifest.write(json.dumps(record) + "\n")
                manifest.flush()
                _save_counters(build, counters)
            if cancelled:
                break

    _register_dataset(build, out_dir)
    return {**counters, "cancelled": cancelled, "images_selected": len(selected)}


def _write_variant(variant: Path, dst: Path) -> None:
    import cv2

    image = cv2.imread(str(variant), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"could not decode {variant}")
    tmp = dst.with_suffix(".part.jpg")
    if not cv2.imwrite(str(tmp), image, [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY]):
        raise RuntimeError(f"could not write {tmp}")
    os.replace(tmp, dst)


def _save_counters(build: AugBuild, counters: dict) -> None:
    AugBuild.objects.filter(pk=build.pk).update(**counters)


def _write_build_config(build: AugBuild, out_dir: Path) -> None:
    data = {
        "build_id": build.pk, "source_dataset": build.source_dataset.name,
        "source_labels_dir": build.source_labels_dir, "output_name": build.output_name,
        "prompts": build.prompts_snapshot, "editor": build.editor_config(),
        "validation": build.validation_settings(), "fraction": build.fraction,
        "variants_per_image": build.variants_per_image, "class_sampling": build.class_sampling,
        "seed": build.seed, "include_originals": build.include_originals, "force": build.force,
        "created_at": build.created_at.isoformat() if build.created_at else None,
    }
    (out_dir / BUILD_CONFIG).write_text(json.dumps(data, indent=2), encoding="utf-8")


def _register_dataset(build: AugBuild, out_dir: Path) -> None:
    """Make the output show up as a Dataset (and so in the experiment picker)."""
    dataset = build.output_dataset or Dataset.objects.filter(name=build.output_name).first()
    if dataset is None:
        dataset = Dataset.objects.create(name=build.output_name, has_labels=True)
    elif not dataset.has_labels:
        dataset.has_labels = True
        dataset.save(update_fields=["has_labels", "updated_at"])
    AugBuild.objects.filter(pk=build.pk).update(output_dataset=dataset)


def read_manifest_tail(build: AugBuild, limit: int = 60) -> list[dict]:
    path = source_root() / build.output_name / MANIFEST
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()[-limit:]
    records = []
    for line in reversed(lines):
        try:
            records.append(json.loads(line))
        except ValueError:
            continue
    return records

