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
``cxcywh -> xyxy`` + scale to canvas pixels + clamp to the canvas + background-
class drop — and emits canvas-pixel xyxy boxes. No NMS, no threshold in the
graph (the service applies it after top-k, mirroring the adapter). The service
handler is then a plain :class:`PassthroughHandler`; the service replicates the
adapter's input pipeline (letterbox resize to the variant's resolution +
ImageNet normalize) so the graph sees the same tensor.

**Export mode.** ``LWDETR.export()`` swaps the module's ``forward`` for
``forward_export`` (which returns the ``(boxes, logits)`` tuple) and recursively
puts the multi-scale deformable-attention submodules into an ONNX-traceable mode.
It mutates the module in place, so we run it on a ``deepcopy`` — leaving the
adapter's own model (the torch reference path) untouched. Input H/W are fixed
(RF-DETR is a fixed-resolution square model), so we export with static spatial
dims; only the detection count is dynamic (the background drop makes it vary).
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
            # forward_export -> (boxes [1,Q,4] normalized cxcywh, logits [1,Q,C+1])
            boxes_n, logits = self.model(pixel_values)
            boxes_n, logits = boxes_n[0], logits[0]
            num_cls = logits.shape[1]  # C + 1 (includes background)

            prob = torch.sigmoid(logits)
            top_scores, top_idx = torch.topk(prob.reshape(-1), num_select)
            box_idx = top_idx // num_cls
            labels = top_idx % num_cls

            cx, cy, w, h = boxes_n[:, 0], boxes_n[:, 1], boxes_n[:, 2], boxes_n[:, 3]
            xyxy = torch.stack(
                [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1
            )  # [Q, 4] normalized xyxy
            # Scale to canvas-pixel coordinates (box_coords="input_pixels") — the
            # canvas includes letterbox padding, so normalized coords alone are
            # not portable back to the original image; the service inverts the
            # scale/pad it recorded for this image (see module docstring).
            xyxy = (xyxy[box_idx] * resolution).clamp(0.0, resolution)

            # Drop the background class (last slot); real classes are 0..num_classes-1.
            keep = labels < num_classes
            return xyxy[keep], top_scores[keep], labels[keep].to(torch.int64)

    wrapper = RFDetrExport(model)
    export_detection_wrapper(
        wrapper, onnx_path, dummy_hw=(resolution, resolution), dynamic_input_hw=False
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
