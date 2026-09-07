"""SmolVLM and Moondream adapters — the small, fast end of the catalogue.

These are the two families realistic at anything near a useful frame rate, which
is why they are worth having even though they say less per frame than Qwen.

Moondream is not an ``AutoModelForImageTextToText`` model and does not use a chat
template; it exposes its own ``query`` method and needs ``trust_remote_code``.
That is different enough to be its own class rather than a branch inside the
SmolVLM one.
"""

from __future__ import annotations

from PIL import Image

from .base import VlmAdapter, device_map, quantization_kwargs, torch_dtype


class SmolVLMAdapter(VlmAdapter):
    def __init__(self, weights: str, quantization: str = ""):
        self.weights = weights
        self.quantization = quantization
        self.label = f"smolvlm:{weights}"
        self.model = None
        self.processor = None

    def load(self) -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        kwargs = {
            "dtype": torch_dtype(self.quantization),
            "device_map": device_map(),
            **quantization_kwargs(self.quantization),
        }
        self.model = AutoModelForImageTextToText.from_pretrained(self.weights, **kwargs)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.weights)

    def infer(self, image_path: str, prompt: str, max_new_tokens: int) -> str:
        import torch

        image = Image.open(image_path).convert("RGB")
        messages = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt}],
        }]
        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self.processor(
            text=text, images=[image], return_tensors="pt"
        ).to(self.model.device)

        with torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        trimmed = generated[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True
        )[0].strip()


class MoondreamAdapter(VlmAdapter):
    """Moondream 2 — its own API, not a Vision2Seq model."""

    def __init__(self, weights: str, quantization: str = ""):
        self.weights = weights
        self.quantization = quantization
        self.label = f"moondream:{weights}"
        self.model = None

    def load(self) -> None:
        from transformers import AutoModelForCausalLM

        self.model = AutoModelForCausalLM.from_pretrained(
            self.weights,
            trust_remote_code=True,
            dtype=torch_dtype(self.quantization),
            device_map=device_map(),
        )
        self.model.eval()

    def infer(self, image_path: str, prompt: str, max_new_tokens: int) -> str:
        image = Image.open(image_path).convert("RGB")
        result = self.model.query(image, prompt)
        # Newer revisions return {"answer": ...}; older ones return the string.
        if isinstance(result, dict):
            return str(result.get("answer", "")).strip()
        return str(result).strip()
