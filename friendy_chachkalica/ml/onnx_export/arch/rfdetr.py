"""RF-DETR ONNX exporter.

RF-DETR (Roboflow's LW-DETR) is a DETR-family, NMS-free detector. Its underlying
``LWDETR`` network emits, per query, a normalized ``cxcywh`` box (``pred_boxes``,
in ``[0,1]`` over the square model input) plus ``num_classes + 1`` class logits —
the extra last slot is the no-object/background class. The training adapter's
``predict`` (``adapters/rfdetr.py``) runs the ``rfdetr`` package's ``PostProcess``:
sigmoid the logits, ``topk`` the flattened ``[queries*(C+1)]`` scores down to
``num_select`` (300), derive ``label = idx % (C+1)`` / ``box = idx // (C+1)``,
``cxcywh -> xyxy``.

RF-DETR resizes to a fixed square canvas via **letterboxing**: aspect-preserving
resize so the longest side hits the variant's resolution, then bottom-right pad
to fill the square. Because the canvas includes real padding, a normalized
model-output coordinate is *not* the same as a normalized original-image
coordinate — the padding must be scaled out. So both the adapter and this
exporter convert the model's normalized box to **canvas-pixel** coordinates
(multiply by ``resolution``) and emit ``box_coords: "input_pixels"``; the
service inverts the canvas-space box back to original pixels using the same
``scale``/pad-offset-0 math it already applies to every other pixel-space arch
(see ``onnx_infer/preprocess.py``'s ``"letterbox"`` resize_mode and
``onnx_infer/postprocess.py``'s ``input_pixels`` inverse — no new service code
needed). ``clip_boxes: True`` because predictions can legitimately land in the
padded margin and must be clamped to the original image bounds.

So the export wrapper bakes exactly that head math — sigmoid + top-k selection +
``cxcywh -> xyxy`` + scale to canvas pixels + clamp to the canvas — and emits
canvas-pixel xyxy boxes. No NMS, no threshold in the graph (the service applies
it after top-k, mirroring the adapter). The service handler is then a plain
:class:`PassthroughHandler`; the service replicates the adapter's input
pipeline (letterbox resize to the variant's resolution + ImageNet normalize) so
the graph sees the same tensor.

**Export mode.** ``LWDETR.export()`` swaps the module's ``forward`` for
``forward_export`` (which returns the ``(boxes, logits)`` tuple) and recursively
puts the multi-scale deformable-attention submodules into an ONNX-traceable mode.
It mutates the module in place, so we run it on a ``deepcopy`` — leaving the
adapter's own model (the torch reference path) untouched. Input H/W are fixed
(RF-DETR is a fixed-resolution square model), so we export with static spatial
dims.

**Batch-aware, fixed detection count.** ``num_select`` (300) is a fixed constant
per image, so the wrapper does the sigmoid/top-k/gather per row of the batch
instead of indexing batch away, emitting ``(boxes[B,K,4], scores[B,K],
labels[B,K])`` — a real batch axis TensorRT can build a wider optimization
profile around (see ``trt_export/arch/__init__.py``'s ``BATCH_AWARE_ARCHS``).
The background class (the extra last logit slot) used to be dropped by boolean
mask (``keep = labels < num_classes``), which made the per-image count
data-dependent and would have collapsed right back to no batch axis. Instead
the background rows are kept at their fixed top-k slot with their score forced
below any real threshold (``top_scores`` set to ``-1.0`` where ``labels ==
num_classes``) — the same "let the service's score-threshold filter do the
dropping" pattern already used for the real threshold, applied to background
too. That filter already runs unconditionally in ``onnx_infer/postprocess.py``'s
``to_friendy`` for every arch, so this needs no new service-side code.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

try:
    from ..common import build_meta, export_detection_wrapper
except ImportError:  # run as a flat script
    from common import build_meta, export_detection_wrapper


def export_rfdetr(adapter, *, num_classes, params, class_map, onnx_path: str | Path) -> dict:
    import torch
    from torch import nn

    score_threshold = float(adapter.score_threshold)
    mean = [float(v) for v in adapter.image_mean]
    std = [float(v) for v in adapter.image_std]
    resolution = int(adapter.resolution)
    num_select = int(getattr(adapter.postprocess, "num_select", 300))

    # LWDETR.export() mutates in place (forward -> forward_export, deform-attn ->
    # export mode); run it on a copy so the adapter's reference path is untouched.
    model = deepcopy(adapter.model).eval()
    model.export()

    class RFDetrExport(nn.Module):
        def __init__(self, model: nn.Module) -> None:
            super().__init__()
            self.model = model

        def forward(self, pixel_values):
            # forward_export -> (boxes [B,Q,4] normalized cxcywh, logits [B,Q,C+1])
            boxes_n, logits = self.model(pixel_values)
            num_cls = logits.shape[-1]  # C + 1 (includes background)

            prob = torch.sigmoid(logits)
            flat_prob = prob.reshape(prob.shape[0], -1)  # [B, Q*(C+1)]
            top_scores, top_idx = torch.topk(flat_prob, num_select, dim=1)  # [B, K]
            box_idx = top_idx // num_cls
            labels = top_idx % num_cls

            cx, cy, w, h = (
                boxes_n[..., 0], boxes_n[..., 1], boxes_n[..., 2], boxes_n[..., 3]
            )
            xyxy = torch.stack(
                [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1
            )  # [B, Q, 4] normalized xyxy
            # Per-batch-row gather: box_idx[b, k] indexes xyxy[b], not a shared
            # first dim, so this can't be a plain xyxy[box_idx] fancy-index
            # (that would index the batch axis itself once B > 1).
            boxes_out = torch.gather(
                xyxy, 1, box_idx.unsqueeze(-1).expand(-1, -1, 4)
            )  # [B, K, 4]
            # Scale to canvas-pixel coordinates (box_coords="input_pixels") — the
            # canvas includes letterbox padding, so normalized coords alone are
            # not portable back to the original image; the service inverts the
            # scale/pad it recorded for this image (see module docstring).
            boxes_out = (boxes_out * resolution).clamp(0.0, resolution)

            # Background (last slot, label == num_classes) stays at its fixed
            # top-k slot rather than being dropped — dropping it would make the
            # per-image count data-dependent again. Force its score below any
            # real threshold instead; the service's score filter drops it from
            # there (see module docstring).
            is_background = labels == num_classes
            top_scores = torch.where(
                is_background, torch.full_like(top_scores, -1.0), top_scores
            )
            return boxes_out, top_scores, labels.to(torch.int64)

    wrapper = RFDetrExport(model)
    export_detection_wrapper(
        wrapper,
        onnx_path,
        dummy_hw=(resolution, resolution),
        dynamic_input_hw=False,
        batch_aware=True,
    )

    return build_meta(
        arch="rfdetr",
        num_classes=num_classes,
        class_map=class_map,
        score_threshold=score_threshold,
        resize_mode="letterbox",
        size=resolution,
        multiple=0,
        pad_value=0.0,
        input_scale="unit",
        normalize={"mean": mean, "std": std},
        box_coords="input_pixels",
        clip_boxes=True,
    )
