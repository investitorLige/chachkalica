"""Ollama adapter — the second backend, not yet wired up.

This is the whole reason ``VlmAdapter`` has such a narrow interface: when an
Ollama server is available, filling in ``infer`` below is the only change
needed. Nothing in the Django app, the worker, the HTTP client, or the admin
knows or cares which backend a model uses.

The shape it will take: Ollama's ``/api/generate`` accepts base64 images
alongside the prompt, so this reads the frame off the shared mount and posts it.
Left raising rather than half-implemented, so a model row configured for Ollama
fails loudly with an accurate message instead of silently producing nothing.
"""

from __future__ import annotations

import os

from .base import VlmAdapter


class OllamaAdapter(VlmAdapter):
    def __init__(self, weights: str, quantization: str = ""):
        # For this backend `weights` is an Ollama model tag, e.g. "llava:7b".
        self.weights = weights
        self.label = f"ollama:{weights}"
        self.base_url = os.environ.get("OLLAMA_URL", "http://ollama:11434").rstrip("/")

    def load(self) -> None:
        raise NotImplementedError(
            "The Ollama backend is not wired up yet. Point a VLM model row at the "
            "transformers backend, or implement OllamaAdapter.infer against "
            f"{self.base_url}/api/generate."
        )

    def infer(self, image_path: str, prompt: str, max_new_tokens: int) -> str:
        raise NotImplementedError("The Ollama backend is not wired up yet.")
