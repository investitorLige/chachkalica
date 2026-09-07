"""Qwen2.5-VL adapter.

Qwen ships its own chat template and a helper (``qwen_vl_utils.process_vision_info``)
that resolves the image references inside a message list. We keep the message
shape it expects rather than hand-rolling the prompt, because the model is
noticeably worse when the template is wrong.
"""

from __future__ import annotations

from PIL import Image

from .base import VlmAdapter, device_map, quantization_kwargs, torch_dtype


class Qwen2VLAdapter(VlmAdapter):
    def __init__(self, weights: str, quantization: str = ""):
        self.weights = weights
        self.quantization = quantization
        self.label = f"qwen2_vl:{weights}"
        self.model = None
        self.processor = None

    def load(self) -> None:
        import torch  # noqa: F401 - imported for its side effect on device init
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        kwargs = {
            "dtype": torch_dtype(self.quantization),
            "device_map": device_map(),
            **quantization_kwargs(self.quantization),
        }
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.weights, **kwargs
        )
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
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=[image], return_tensors="pt", padding=True
        ).to(self.model.device)

        with torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        # Strip the prompt tokens back off; decoding the whole sequence would
        # echo the question back as part of every alert.
        trimmed = generated[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
