"""A fast, GPU-free stand-in editor — for testing the pipeline, not for data.

It reads a few keywords from the prompt and applies the matching classical
degradation (darken, blur, JPEG, colour cast, noise), deterministically per
seed. That exercises everything around the model — the studio, previews,
validation, the dataset build, the manifest — while FireRed's weights are not
in the cache or the GPU is busy training. Builds made with it are marked
``mock`` in their manifest and model label, so they cannot be mistaken for the
real thing.
"""

from __future__ import annotations

import io
import random

from PIL import Image, ImageEnhance, ImageFilter

from .base import EditParams, GenerativeImageEditor


class MockEditor(GenerativeImageEditor):
    label = "mock"

    def __init__(self, config: dict | None = None):
        pass

    def load(self) -> None:
        pass

    def edit(self, image: Image.Image, prompt: str, seed: int, params: EditParams) -> Image.Image:
        rng = random.Random(f"{seed}:{prompt}")
        text = prompt.lower()
        out = image.convert("RGB")
        if any(word in text for word in ("night", "dark", "low-light", "low light")):
            out = ImageEnhance.Brightness(out).enhance(0.35 + 0.15 * rng.random())
            out = ImageEnhance.Contrast(out).enhance(0.75)
        if "blur" in text:
            out = out.filter(ImageFilter.GaussianBlur(1.0 + rng.random()))
        if any(word in text for word in ("fluorescent", "colour", "color", "tint")):
            r, g, b = out.split()
            g = g.point(lambda v: min(255, int(v * 1.08)))
            b = b.point(lambda v: int(v * 0.9))
            out = Image.merge("RGB", (r, g, b))
        if "compress" in text or "cctv" in text:
            buffer = io.BytesIO()
            out.save(buffer, "JPEG", quality=rng.randint(8, 18))
            out = Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")
        if "noise" in text or "sensor" in text:
            # From the seeded rng, not Image.effect_noise (unseeded global state),
            # so the same seed gives the same bytes.
            noise = Image.frombytes("L", out.size, rng.randbytes(out.size[0] * out.size[1]))
            noise = noise.convert("RGB")
            out = Image.blend(out, noise, 0.08)
        return out
