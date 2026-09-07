"""InternVL and LLaVA-style adapters.

Both are ``AutoModelForImageTextToText``-shaped in recent transformers releases and
carry their own chat templates, so a single class covers them; the family exists
mainly as a well-known baseline to compare the others against.
"""

from __future__ import annotations

from PIL import Image

from .base import VlmAdapter, device_map, quantization_kwargs, torch_dtype


class InternVLLlavaAdapter(VlmAdapter):
    def __init__(self, weights: str, quantization: str = ""):
        self.weights = weights
        self.quantization = quantization
        self.label = f"internvl_llava:{weights}"
        self.model = None
        self.processor = None

    def load(self) -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        kwargs = {
            "dtype": torch_dtype(self.quantization),
            "device_map": device_map(),
            # InternVL still ships parts of its implementation in the repo.
            "trust_remote_code": True,
            **quantization_kwargs(self.quantization),
        }
        self.model = AutoModelForImageTextToText.from_pretrained(self.weights, **kwargs)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            self.weights, trust_remote_code=True
        )

    def infer(self, image_path: str, prompt: str, max_new_tokens: int) -> str:
        import torch

        image = Image.open(image_path).convert("RGB")
        messages = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt}],
        }]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=text, images=image, return_tensors="pt"
        ).to(self.model.device)

        with torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        trimmed = generated[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True
        )[0].strip()
