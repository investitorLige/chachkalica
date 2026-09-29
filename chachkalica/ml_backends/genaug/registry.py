"""Which editor loads which model, and which HF repos each one needs.

Keep the keys in sync with ``chachkalica/genaug/models.py``'s
``EDITOR_CHOICES`` — that is what the studio offers, this is what loads.
"""

from __future__ import annotations

from .editors.base import GenerativeImageEditor


def build_editor(name: str, config: dict) -> GenerativeImageEditor:
    """Construct (but do not load) one editor."""
    if name == "firered":
        from .editors.firered import FireRedEditor

        return FireRedEditor(config)
    if name == "mock":
        from .editors.mock import MockEditor

        return MockEditor(config)
    raise ValueError(f"unknown editor {name!r} (known: firered, mock)")


def required_repos(name: str, transformer: str, lightning: bool) -> list[tuple[str, str | None]]:
    """``(repo, filename-or-None)`` pairs that must be in the offline cache."""
    if name != "firered":
        return []
    from .editors.firered import BASE_REPO, COMFY_REPO, GGUF_FILES, LIGHTNING_FILE

    needed: list[tuple[str, str | None]] = [(BASE_REPO, "text_encoder/config.json")]
    if transformer == "bf16":
        needed.append((BASE_REPO, "transformer/diffusion_pytorch_model.safetensors.index.json"))
    else:
        # The GGUF only replaces the transformer weights; its config still
        # comes from the base repo.
        needed.append((BASE_REPO, "transformer/config.json"))
        needed.append((COMFY_REPO, GGUF_FILES.get(transformer, "")))
    if lightning:
        needed.append((COMFY_REPO, LIGHTNING_FILE))
    return needed
