"""FireRed-Image-Edit-1.1, sized to fit a 16 GB GPU.

In bf16 FireRed is ~57 GB: a 20B-parameter Qwen-Image transformer (41 GB) plus
a Qwen2.5-VL-7B text encoder (16.6 GB). The configuration here is what makes
it fit one RTX 4080 SUPER:

* **Transformer: FireRedTeam's own q4_k_m GGUF** (13.1 GB, from the
  ``FireRed-Image-Edit-1.1-ComfyUI`` repo). Its tensor names are already
  diffusers' (checked against the file header), which is what diffusers'
  ``from_single_file`` needs for ``QwenImageTransformer2DModel`` — that class
  has no key conversion. ``transformer="bf16"`` loads the full-precision one
  from the main repo instead, for a machine with the memory for it.
* **Text encoder: bitsandbytes NF4** (~5.5 GB), quantized at load from the
  main repo's bf16 weights.
* **Model CPU offload.** Text encoder, transformer and VAE take turns on the
  GPU instead of sitting there together. Peak is roughly the transformer plus
  activations, ~14.5 GB at 1 MP — so this only fits while nothing else (a
  training run, SAM, a VLM) holds the GPU. The service checks free memory
  before loading and says so, rather than OOMing half-way through.
* **VAE tiling**, so decoding a 1 MP latent does not spike past the rest.

``lightning`` adds FireRed's official 1.1 Lightning LoRA (8 steps, CFG 1),
roughly 5x fewer transformer passes per image. Its scheduler settings are
Qwen-Image-Lightning's published ones (the LoRA is distilled from the same
recipe).

FireRed's own "optimized" path (quanto int8 + DBCache + torch.compile) needs
~30 GB of VRAM by FireRed's own account, so it is not offered here.
"""

from __future__ import annotations

import logging
import math
import os
import time

from .base import EditParams, GenerativeImageEditor

logger = logging.getLogger("genaug-backend")

BASE_REPO = "FireRedTeam/FireRed-Image-Edit-1.1"
COMFY_REPO = "FireRedTeam/FireRed-Image-Edit-1.1-ComfyUI"
GGUF_FILES = {
    "q4_k_m": "FireRed-Image-Edit-1.1-transformer-q4_k_m.gguf",
    "q4_1": "FireRed-Image-Edit-1.1-transformer-q4_1.gguf",
}
LIGHTNING_FILE = "FireRed-Image-Edit-1.1-Lightning-8steps-v1.2.safetensors"

# Qwen-Image-Lightning's scheduler: a fixed exponential time shift of log(3)
# instead of the base model's resolution-dependent one.
LIGHTNING_SCHEDULER = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}

#: Free VRAM (GiB) required before loading with model offload. Override with
#: GENAUG_MIN_FREE_GB when a different card or quantization changes the need.
DEFAULT_MIN_FREE_GB = {"q4_k_m": 14.0, "q4_1": 14.0, "bf16": 44.0}


class FireRedEditor(GenerativeImageEditor):
    def __init__(self, config: dict):
        self.transformer_kind = config.get("transformer") or "q4_k_m"
        if self.transformer_kind not in (*GGUF_FILES, "bf16"):
            raise ValueError(f"unknown transformer {self.transformer_kind!r}")
        self.lightning = bool(config.get("lightning"))
        self.offload = config.get("offload") or "model"
        self.text_encoder_quant = config.get("text_encoder_quant", "nf4")
        self.label = f"firered-1.1:{self.transformer_kind}" + ("+lightning" if self.lightning else "")
        self.pipe = None
        self.dtype = None
        self.load_seconds = None

    # -- loading ---------------------------------------------------------

    def load(self) -> None:
        import torch

        if not torch.cuda.is_available():
            # A 20B diffusion model on CPU is hours per image; failing loudly
            # beats a job that looks alive and never finishes.
            raise RuntimeError("CUDA is not available in genaug-backend — FireRed needs a GPU")

        free, _total = torch.cuda.mem_get_info()
        need = float(os.environ.get("GENAUG_MIN_FREE_GB")
                     or DEFAULT_MIN_FREE_GB[self.transformer_kind])
        if free / 2**30 < need:
            raise GpuBusyError(
                f"only {free / 2**30:.1f} GiB of GPU memory is free and FireRed "
                f"({self.transformer_kind}) needs ~{need:.0f} GiB — is a training run, "
                f"SAM or a VLM holding the GPU? Try again when it is idle."
            )

        started = time.monotonic()
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        from diffusers import QwenImageEditPlusPipeline

        transformer = self._load_transformer()
        text_encoder = self._load_text_encoder()
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            BASE_REPO, transformer=transformer, text_encoder=text_encoder,
            torch_dtype=self.dtype,
        )
        if self.lightning:
            from diffusers import FlowMatchEulerDiscreteScheduler

            pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(LIGHTNING_SCHEDULER)
            pipe.load_lora_weights(COMFY_REPO, weight_name=LIGHTNING_FILE)

        pipe.vae.enable_tiling()
        if self.offload == "model":
            pipe.enable_model_cpu_offload()
        elif self.offload == "none":
            pipe.to("cuda")
        else:
            raise ValueError(f"unknown offload mode {self.offload!r}")
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe
        self.load_seconds = time.monotonic() - started
        logger.info("loaded %s in %.0fs (dtype=%s, offload=%s)",
                    self.label, self.load_seconds, self.dtype, self.offload)

    def _load_transformer(self):
        from diffusers import QwenImageTransformer2DModel

        if self.transformer_kind == "bf16":
            return QwenImageTransformer2DModel.from_pretrained(
                BASE_REPO, subfolder="transformer", torch_dtype=self.dtype)

        from diffusers import GGUFQuantizationConfig
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(COMFY_REPO, GGUF_FILES[self.transformer_kind])
        return QwenImageTransformer2DModel.from_single_file(
            path,
            quantization_config=GGUFQuantizationConfig(compute_dtype=self.dtype),
            config=BASE_REPO, subfolder="transformer", torch_dtype=self.dtype,
        )

    def _load_text_encoder(self):
        from transformers import Qwen2_5_VLForConditionalGeneration

        kwargs = {"dtype": self.dtype}
        if self.text_encoder_quant == "nf4":
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.dtype,
            )
        return Qwen2_5_VLForConditionalGeneration.from_pretrained(
            BASE_REPO, subfolder="text_encoder", **kwargs)

    # -- editing ---------------------------------------------------------

    def edit(self, image, prompt: str, seed: int, params: EditParams):
        import torch
        from PIL import Image

        steps, cfg = params.num_inference_steps, params.true_cfg_scale
        if self.lightning:
            # The distilled LoRA is trained for exactly this; more steps or real
            # CFG on top of it only burns time and over-sharpens.
            steps, cfg = min(steps, 8), 1.0
        generator = torch.Generator(device="cuda").manual_seed(int(seed))
        with torch.inference_mode():
            result = self.pipe(
                image=[image],
                prompt=prompt,
                negative_prompt=params.negative_prompt or " ",
                true_cfg_scale=cfg,
                num_inference_steps=steps,
                generator=generator,
                num_images_per_prompt=1,
            ).images[0]
        if result.size != image.size:
            # The pipeline works at ~1 MP with sides rounded to 32; the labels
            # are for the source's own pixel grid.
            result = result.resize(image.size, Image.LANCZOS)
        return result.convert("RGB")

    def unload(self) -> None:
        self.pipe = None

    def describe(self) -> dict:
        info = {"label": self.label, "dtype": str(self.dtype), "offload": self.offload,
                "load_seconds": self.load_seconds}
        try:
            import torch

            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_allocated_gib"] = round(torch.cuda.memory_allocated() / 2**30, 2)
            info["gpu_peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        except Exception:  # noqa: BLE001 - reporting must not fail the request
            pass
        return info


class GpuBusyError(RuntimeError):
    """Not enough free VRAM to load — a retryable condition, not a failure."""
