"""Runtime options for the GPU-resident pipelines.

Deliberately separate from ``chachak.config.PipelineConfig``: that describes *what the
pipeline detects* — the geometry, thresholds and class space the weights were tuned with, all
of it recorded in a bundle's ``pipeline.json``. This describes *how this runtime executes it*,
which changes nothing about the answer and is therefore safe to expose as CLI flags.

Stdlib only, so it can be imported without torch (see the package docstring).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Optional

#: How a frame with more person crops than ``max_crops_per_frame`` is handled.
CROP_OVERFLOW_MODES = ("spill", "drop_lowest", "error")


@dataclass(frozen=True)
class GpuOptions:
    """Execution knobs for :mod:`gpu_infer.pipeline`.

    The defaults are the fastest configuration whose *detections* still match
    ``chachak.pipeline`` exactly, with one exception called out below.

    ``gpu_decode``
        Decode JPEGs with nvJPEG on the device instead of PIL on the host. **Off by default,
        and that is a measured decision rather than caution.**

        It is the single largest remaining win — PIL decode measured 44 of 54 ms/frame at 4K.
        But nvJPEG does not merely round differently from libjpeg-turbo. Measured on this
        repo's own PPE frames (1080x810 4:2:0): the median per-pixel difference is **0** and
        the mean is sub-LSB, but the tail is heavy — about **12% of pixels differ by >= 1/255**
        and **0.6% by >= 8/255**, with a maximum near **64/255** concentrated on sharp edges,
        which is what different chroma upsampling produces. Downstream that moved a box by
        ~11% of the frame on one of four sample images.

        So this is not a rounding knob, it is a different input. Turn it on deliberately, for
        a throughput-bound job where that is acceptable, and check
        ``tests/test_differential_gpu.py``'s decode report for the model in question first.
        Everything else in this package stays bit-identical either way.

        Non-JPEG files, a missing torchvision, and any file nvJPEG rejects all fall back to
        PIL automatically, so enabling it is never a hard failure.

    ``max_crops_per_frame``
        Sizes the preallocated crop batch and the NMS overlap matrix. Not a cap on
        detections — see ``crop_overflow``.

    ``crop_overflow``
        What to do when a frame yields more person crops than ``max_crops_per_frame``.
        ``"spill"`` (the default) issues additional chained submissions, which preserves the
        original run-every-person behaviour exactly. ``"drop_lowest"`` keeps the highest-scoring
        crops via a fixed-size ``topk``, which is sync-free but **changes detections**.
        ``"error"`` raises.

    ``nms_rounds``
        Trip count for the round-based greedy NMS in :mod:`gpu_infer.nms`. Each round resolves
        one level of the suppression chain; real detections at IoU >= 0.5 bottom out at two to
        four. A fixed count is what keeps the loop sync-free — testing for convergence per
        round would drain the queue, which is the cost this whole package exists to avoid.

    ``nms_strict``
        Raise if ``nms_rounds`` was not enough to resolve every box, instead of leaving the
        remainder suppressed. The check is a device-side flag folded into the batch's single
        existing synchronization, so it costs nothing. Leave it on: silently over-suppressing
        is exactly the failure this package must not introduce.

    ``nms_topk``
        Optional cap on detections entering the merge NMS. ``None`` (the default) matches
        ``chachak.boxes.merge_predictions``, which has no cap. Setting it bounds the
        ``[K, K]`` overlap matrix at the cost of dropping the lowest-scoring detections past
        the cap.

    ``async_engine``
        Submit engine work without synchronizing. This is the mechanism, not a tuning knob —
        turning it off makes the package a slower ``chachak.pipeline`` with no upside, and
        exists only so a suspected ordering bug can be bisected against the synchronous path.

    ``profile``
        Print a per-stage timing breakdown. Forces synchronization at stage boundaries, so the
        totals it reports are *higher* than the same run without it.
    """

    gpu_decode: bool = False
    max_crops_per_frame: int = 16
    crop_overflow: str = "spill"
    nms_rounds: int = 8
    nms_strict: bool = True
    nms_topk: Optional[int] = None
    async_engine: bool = True
    profile: bool = False

    def __post_init__(self) -> None:
        if self.max_crops_per_frame < 1:
            raise ValueError(
                f"max_crops_per_frame must be at least 1, got {self.max_crops_per_frame}"
            )
        if self.crop_overflow not in CROP_OVERFLOW_MODES:
            raise ValueError(
                f"crop_overflow must be one of {CROP_OVERFLOW_MODES}, "
                f"got {self.crop_overflow!r}"
            )
        if self.nms_rounds < 1:
            raise ValueError(f"nms_rounds must be at least 1, got {self.nms_rounds}")
        if self.nms_topk is not None and self.nms_topk < 1:
            raise ValueError(
                f"nms_topk must be at least 1 when set, got {self.nms_topk}"
            )

    @classmethod
    def strict_parity(cls, **overrides: Any) -> "GpuOptions":
        """The configuration whose output is byte-identical to ``chachak.pipeline``.

        Every difference from the defaults is a decode or a selection choice, never a
        geometry one — the crop resample is the verbatim port in both cases. This is what
        the differential test runs, and what to reach for when a result has to be compared
        against a number the training repo produced.
        """
        return cls(
            gpu_decode=False,
            crop_overflow="spill",
            nms_topk=None,
            nms_strict=True,
            **overrides,
        )

    def replace(self, **changes: Any) -> "GpuOptions":
        """A copy with ``changes`` applied, re-validated."""
        return replace(self, **changes)

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> "GpuOptions":
        """Build from a mapping, ignoring keys this version does not know.

        Unknown keys are dropped rather than rejected so a newer bundle's options block
        cannot make an older runtime refuse to start — the same tolerance
        ``chachak.config.pipeline_config_from_dict`` extends to the pipeline request.
        """
        if not raw:
            return cls()
        known = {field for field in cls.__dataclass_fields__}
        return cls(**{key: value for key, value in raw.items() if key in known})
