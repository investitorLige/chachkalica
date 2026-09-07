"""Curated VLM families and their published checkpoints.

The same split the detection side already uses: this module (Django) owns the
*catalogue* the admin dropdown renders, while ``ml_backends/vlm/adapters/``
(the backend container) owns the code that actually loads a model. Keep the
family keys here in sync with ``FAMILY_ADAPTERS`` in
``chachkalica/ml_backends/vlm/registry.py``.

Shaped after ``training.model_specs.WEIGHTS_CATALOG``: per-family lists of
``{value, label, variant}``, rendered by a variant-aware ``<select>`` that hides
the entries belonging to other families.

**huggingface.co does not verify from this network.** It resolves and answers,
but a TLS-intercepting proxy (``CN=atlas-MASTER``, a sub-CA of the internal
``atlas-ATLASMASTER-CA``) re-signs the certificate, and that root is published
only over LDAP inside ``atlas.local`` — so it is in no container's trust store
and every ``from_pretrained`` dies on ``CERTIFICATE_VERIFY_FAILED`` rather than
on a connection error. Weights are therefore served from the hand-populated
offline cache at ``data/hf_cache`` (``HF_HOME`` + ``HF_HUB_OFFLINE=1``, exactly
as the trainer already does). So every option is
existence-gated against that cache — the same trick
``model_specs.bytetrack_yolox_options()`` uses for on-disk checkpoints — and a
repo nobody has copied in yet is shown as unavailable at add time rather than
failing on the first frame of a run.
"""

from pathlib import Path

from django.conf import settings

TRANSFORMERS = "transformers"
OLLAMA = "ollama"

BACKEND_CHOICES = [
    (TRANSFORMERS, "HF transformers (local GPU)"),
    (OLLAMA, "Ollama (not wired up yet)"),
]

FAMILY_CHOICES = [
    ("qwen2_vl", "Qwen2.5-VL"),
    ("smolvlm", "SmolVLM / Moondream"),
    ("internvl_llava", "InternVL / LLaVA-style"),
]

VLM_WEIGHTS_CATALOG: dict[str, list[dict]] = {
    "qwen2_vl": [
        {"value": "Qwen/Qwen2.5-VL-3B-Instruct",
         "label": "Qwen2.5-VL 3B Instruct", "variant": "3B"},
        {"value": "Qwen/Qwen2.5-VL-7B-Instruct",
         "label": "Qwen2.5-VL 7B Instruct", "variant": "7B"},
    ],
    "smolvlm": [
        # Both SmolVLM repos ship an ``onnx/`` folder of transformers.js exports
        # (25 GB for the 2B, 3 GB for the 256M) that dwarfs the safetensors we
        # actually load. Excluding it is the difference between a 4.5 GB copy
        # and a 29.5 GB one.
        {"value": "HuggingFaceTB/SmolVLM-Instruct",
         "label": "SmolVLM Instruct", "variant": "2B",
         "exclude": ["onnx/*"]},
        {"value": "HuggingFaceTB/SmolVLM-256M-Instruct",
         "label": "SmolVLM 256M Instruct", "variant": "256M",
         "exclude": ["onnx/*"]},
        # moondream2's remote code hardcodes a SECOND repo it never declares:
        # `Tokenizer.from_pretrained("moondream/starmie-v1")` in moondream.py.
        # 3.7 MB, but without it the model raises LocalEntryNotFoundError on
        # load — so the gate has to check for it too, or the dropdown offers a
        # model that cannot run.
        {"value": "vikhyatk/moondream2",
         "label": "Moondream 2", "variant": "2B",
         "requires": ["moondream/starmie-v1"]},
    ],
    "internvl_llava": [
        {"value": "OpenGVLab/InternVL2_5-4B",
         "label": "InternVL 2.5 4B", "variant": "4B"},
        {"value": "llava-hf/llava-1.5-7b-hf",
         "label": "LLaVA 1.5 7B", "variant": "7B"},
    ],
}

#: Free-text escape hatch, mirroring ``model_specs.WEIGHTS_CUSTOM``.
WEIGHTS_CUSTOM = "__custom__"


def hf_cache_root() -> Path:
    """Where ``from_pretrained`` will look for an already-downloaded snapshot.

    Matches the ``HF_HOME=/app/data/hf_cache`` the trainer container sets, but
    resolved from this project's own BASE_DIR so the admin can check it without
    caring whether it is running inside that container.
    """
    return Path(settings.BASE_DIR) / "data" / "hf_cache" / "hub"


def cache_dir_name(repo_id: str) -> str:
    """The directory huggingface_hub creates for ``org/name``."""
    return "models--" + repo_id.replace("/", "--")


def _snapshot_present(repo_id: str) -> bool:
    """Whether one repo id has a directory in the offline cache."""
    try:
        return (hf_cache_root() / cache_dir_name(repo_id)).is_dir()
    except OSError:
        return False


def is_cached(repo_id: str) -> bool:
    """Whether this checkpoint can actually load from the offline cache.

    A local filesystem path (rather than a repo id) is treated as cached — the
    operator is pointing at something they placed themselves.

    A catalogue entry's ``requires`` companions must be present too. Some repos
    pull a second repo from inside their ``trust_remote_code`` modelling file,
    where nothing in the snapshot declares it; copying only the headline repo
    yields a directory that exists and a model that still cannot load.
    """
    if not repo_id:
        return False
    if "/" not in repo_id or repo_id.startswith((".", "/", "~")):
        return Path(repo_id).exists()
    if not _snapshot_present(repo_id):
        return False
    entry = entry_for(repo_id) or {}
    return all(_snapshot_present(dep) for dep in entry.get("requires", []))


def missing_repos(repo_id: str) -> list[str]:
    """Every repo id this entry needs that is not in the cache yet."""
    entry = entry_for(repo_id) or {}
    wanted = [repo_id, *entry.get("requires", [])]
    return [r for r in wanted if not _snapshot_present(r)]


def weights_options(family: str | None = None) -> list[tuple[str, str]]:
    """``(value, label)`` pairs for the weights dropdown.

    Every family's entries are returned together — the form's JS narrows them to
    the selected family via ``data-family``, the same way
    ``experiment_model_form.js`` narrows by ``data-variant``. Uncached repos stay
    in the list but are labelled and disabled, so the reason a model is missing
    is visible rather than mysterious.
    """
    options: list[tuple[str, str]] = [("", "— select weights —")]
    for key, entries in VLM_WEIGHTS_CATALOG.items():
        if family and key != family:
            continue
        for entry in entries:
            label = entry["label"]
            if entry.get("variant"):
                label = f"{label} ({entry['variant']})"
            if not is_cached(entry["value"]):
                label = f"{label} — not in hf_cache"
            options.append((entry["value"], label))
    options.append((WEIGHTS_CUSTOM, "Custom repo id / local path…"))
    return options


def download_args(repo_id: str) -> str:
    """The ``huggingface-cli download`` arguments for one catalogued repo.

    Just the repo id, plus any ``exclude`` globs the entry carries — some repos
    ship alternative serialisations (ONNX, GGUF) an order of magnitude larger
    than the safetensors the adapters actually load, and there is no reason to
    move those across the network by hand.
    """
    entry = entry_for(repo_id) or {}
    parts = [repo_id]
    for pattern in entry.get("exclude", []):
        parts.append(f'--exclude "{pattern}"')
    return " ".join(parts)


def entry_for(repo_id: str) -> dict | None:
    for entries in VLM_WEIGHTS_CATALOG.values():
        for entry in entries:
            if entry["value"] == repo_id:
                return entry
    return None
