"""Which adapter loads which family.

Same shape as ``friendy_chachkalica/registry.py``'s ``MODEL_REGISTRY``: a flat
name → builder map plus one function to build from it.

**Keep the family keys in sync with ``chachkalica/vlm/weights_catalog.py``'s
``FAMILY_CHOICES``** — that module is what the admin dropdown renders, this is
what actually loads the weights, and a key present in one but not the other
produces a model row that cannot run.
"""

from __future__ import annotations

from .adapters.base import VlmAdapter
from .adapters.ollama import OllamaAdapter
from .adapters.transformers_internvl_llava import InternVLLlavaAdapter
from .adapters.transformers_qwen2_vl import Qwen2VLAdapter
from .adapters.transformers_smolvlm import MoondreamAdapter, SmolVLMAdapter

TRANSFORMERS = "transformers"
OLLAMA = "ollama"

FAMILY_ADAPTERS = {
    "qwen2_vl": Qwen2VLAdapter,
    "smolvlm": SmolVLMAdapter,
    "internvl_llava": InternVLLlavaAdapter,
}


def build_adapter(backend: str, family: str, weights: str,
                  quantization: str = "") -> VlmAdapter:
    """Construct (but do not load) the adapter for one model configuration."""
    if backend == OLLAMA:
        return OllamaAdapter(weights, quantization)
    if backend != TRANSFORMERS:
        raise ValueError(f"unknown backend: {backend!r}")

    # Moondream shares the "small and fast" family with SmolVLM in the admin
    # dropdown, but its API is different enough to need its own adapter.
    if family == "smolvlm" and "moondream" in weights.lower():
        return MoondreamAdapter(weights, quantization)

    try:
        adapter_cls = FAMILY_ADAPTERS[family]
    except KeyError:
        raise ValueError(
            f"unknown family: {family!r} (known: {', '.join(sorted(FAMILY_ADAPTERS))})"
        ) from None
    return adapter_cls(weights, quantization)
