"""``meta.json`` — Contract B: the sidecar that tells the service how to
pre-process an input and interpret the graph's outputs.

The exporter (``friendy_chachkalica/ml/onnx_export``) writes one of these next to
every ``.onnx``. The service reads it and needs nothing else from the training
stack. ``schema_version`` is guarded on load so an older service refuses a newer
artifact loudly instead of silently mis-decoding it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from .errors import MetaSchemaError

SCHEMA_VERSION = 1

# resize_mode values (what the *service* does to the input before the graph):
#   "none"         — pass the image through unchanged (the graph handles any
#                    internal resize, e.g. torchvision detection models).
#   "square"       — resize to (size, size) ignoring aspect ratio.
#   "longest_side" — scale so the longest side == max_size, preserving aspect
#                    (only ever downscales; no padding to a fixed canvas).
#   "letterbox"    — scale (up or down) so the longest side == size exactly,
#                    preserving aspect, then pad bottom-right to (size, size).
RESIZE_MODES = {"none", "square", "longest_side", "letterbox"}
INPUT_SCALES = {"unit", "byte"}  # 0..1 floats vs 0..255 floats

# box_coords — the frame the graph emits boxes in:
#   "input_pixels"     — xyxy in the graph's input-tensor pixel space; the service
#                        inverts its recorded resize+pad to map back (RetinaNet,
#                        YOLOX).
#   "input_normalized" — xyxy already normalized to [0,1] over the model input
#                        (DETR-family: the box is invariant to the absolute input
#                        size). Maps straight to Friendy xywhn with no transform
#                        inverse (RT-DETR, RF-DETR).
BOX_COORDS = {"input_pixels", "input_normalized"}


@dataclass(frozen=True)
class InputSpec:
    resize_mode: str = "none"
    size: Optional[int] = None          # square
    max_size: Optional[int] = None      # longest_side
    multiple: int = 0                   # pad each side up to this (0 = no pad)
    pad_value: float = 0.0
    input_scale: str = "unit"

    def validate(self) -> None:
        if self.resize_mode not in RESIZE_MODES:
            raise MetaSchemaError(f"input.resize_mode {self.resize_mode!r} not in {RESIZE_MODES}")
        if self.input_scale not in INPUT_SCALES:
            raise MetaSchemaError(f"input.input_scale {self.input_scale!r} not in {INPUT_SCALES}")
        if self.resize_mode in ("square", "letterbox") and not self.size:
            raise MetaSchemaError(f"input.resize_mode {self.resize_mode!r} requires input.size")
        if self.resize_mode == "longest_side" and not self.max_size:
            raise MetaSchemaError("input.resize_mode 'longest_side' requires input.max_size")


@dataclass(frozen=True)
class Normalize:
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


@dataclass(frozen=True)
class ModelMeta:
    arch: str
    num_classes: int
    class_map: dict[int, str]
    score_threshold: float
    input: InputSpec
    normalize: Optional[Normalize] = None
    layout: str = "rgb"
    box_coords: str = "input_pixels"
    # Clip boxes to the original image bounds after the coordinate inverse.
    # The rule is parity, not padding: this must be True for exactly the archs
    # whose torch path clips, so the service can't diverge from the adapter it was
    # exported from. True for YOLOX (clip_xyxy), RF-DETR (letterbox padding means
    # predictions land in the pad margin) and RT-DETR (its decoder emits normalized
    # boxes that can extend past the canvas with no padding involved); a no-op for
    # archs that already emit in-bounds boxes (RetinaNet clips internally). Each
    # arch's adapter ``predict()`` and its ``onnx_export/arch/*.py`` build_meta call
    # have to agree — an unclipped export scores lower mAP than its own torch path
    # on edge-touching objects, which reads as an export regression.
    clip_boxes: bool = False
    schema_version: int = SCHEMA_VERSION

    # ------------------------------------------------------------------ load
    @classmethod
    def from_dict(cls, raw: dict) -> "ModelMeta":
        version = raw.get("schema_version")
        if version != SCHEMA_VERSION:
            raise MetaSchemaError(
                f"meta.json schema_version {version!r} != supported {SCHEMA_VERSION}. "
                "Re-export the model or upgrade the service."
            )
        for key in ("arch", "num_classes", "class_map", "score_threshold", "input"):
            if key not in raw:
                raise MetaSchemaError(f"meta.json missing required field {key!r}")

        norm_raw = raw.get("normalize")
        normalize = (
            Normalize(mean=tuple(norm_raw["mean"]), std=tuple(norm_raw["std"]))
            if norm_raw
            else None
        )
        spec = InputSpec(**{k: v for k, v in raw["input"].items() if k in InputSpec.__dataclass_fields__})
        spec.validate()

        meta = cls(
            arch=str(raw["arch"]),
            num_classes=int(raw["num_classes"]),
            # JSON object keys are strings; normalize to int ids.
            class_map={int(k): str(v) for k, v in raw["class_map"].items()},
            score_threshold=float(raw["score_threshold"]),
            input=spec,
            normalize=normalize,
            layout=str(raw.get("layout", "rgb")),
            box_coords=str(raw.get("box_coords", "input_pixels")),
            clip_boxes=bool(raw.get("clip_boxes", False)),
            schema_version=SCHEMA_VERSION,
        )
        if meta.box_coords not in BOX_COORDS:
            raise MetaSchemaError(
                f"box_coords {meta.box_coords!r} unsupported; expected one of {BOX_COORDS}"
            )
        return meta

    @classmethod
    def load(cls, path: str | Path) -> "ModelMeta":
        with open(path) as file:
            return cls.from_dict(json.load(file))

    # ------------------------------------------------------------------ save
    def to_dict(self) -> dict:
        raw = asdict(self)
        if self.normalize is None:
            raw["normalize"] = None
        # Emit class_map with string keys for clean JSON round-tripping.
        raw["class_map"] = {str(k): v for k, v in self.class_map.items()}
        return raw

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as file:
            json.dump(self.to_dict(), file, indent=2)
        return path
