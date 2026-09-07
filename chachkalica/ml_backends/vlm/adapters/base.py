"""The one interface every VLM adapter implements.

Deliberately tiny. ``infer(image_path, prompt, max_new_tokens) -> str`` is the
entire contract between this container and the rest of the system, which is what
makes adding a second backend (Ollama) a matter of writing one class rather than
touching the worker, the HTTP client, the models, or the admin.

Frames arrive as *paths* on the shared ``data/`` mount rather than as bytes over
the wire — the convention the trainer's ``/predict_image`` and the camera
live-inference path already follow.
"""

from __future__ import annotations


class VlmAdapter:
    """A loaded vision-language model that can answer a question about an image."""

    #: Set by concrete adapters so the service can report what is warm.
    label: str = "vlm"

    def load(self) -> None:
        """Bring the model into memory. Called once, before the first infer."""
        raise NotImplementedError

    def infer(self, image_path: str, prompt: str, max_new_tokens: int) -> str:
        """Answer ``prompt`` about the image at ``image_path``."""
        raise NotImplementedError


def torch_dtype(quantization: str):
    """The dtype to load in when not quantizing.

    bf16 where the GPU supports it, fp16 otherwise, fp32 on CPU — VLMs are large
    enough that fp32 on a shared GPU is rarely the right default.
    """
    import torch

    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def quantization_kwargs(quantization: str) -> dict:
    """``from_pretrained`` kwargs for an optional bitsandbytes quantization."""
    if not quantization:
        return {}
    from transformers import BitsAndBytesConfig

    if quantization == "8bit":
        return {"quantization_config": BitsAndBytesConfig(load_in_8bit=True)}
    if quantization == "4bit":
        return {"quantization_config": BitsAndBytesConfig(load_in_4bit=True)}
    raise ValueError(f"unknown quantization: {quantization!r}")


def device_map() -> str:
    import torch

    return "auto" if torch.cuda.is_available() else "cpu"
