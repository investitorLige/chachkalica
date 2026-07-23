"""RetinaNet → TensorRT via EfficientNMS.

The standard RetinaNet ONNX runs torchvision's full ``postprocess_detections``
(per-level top-k + score filter + ``batched_nms`` + clip) — data-dependent
subgraphs TensorRT can't compile. Here we re-export the graph only up to **raw
decoded boxes + per-class scores** (torchvision's transform + backbone + head +
anchor decode + sigmoid — all shape-dependent but NOT data-dependent, so TRT
compiles them with an optimization profile) and let ``EfficientNMS_TRT`` do the
score-threshold + class-aware NMS.

torchvision resizes internally (aspect-preserving) and reports the resized size in
``images.image_sizes``; we map the decoded boxes from that resized frame back to
the graph's input-pixel frame (a single per-axis scale), so the engine's boxes are
input-pixel xyxy and the meta stays ``resize_mode: none`` / ``box_coords:
input_pixels`` — identical pre/post-processing to the ONNX path.
"""

from __future__ import annotations

from pathlib import Path

try:
    from ..efficientnms import append_efficientnms, export_raw_boxes_scores
except ImportError:  # run flat
    from trt_export.efficientnms import append_efficientnms, export_raw_boxes_scores  # type: ignore


def prep_retinanet(adapter, meta: dict, out_onnx_path) -> None:
    import torch
    from torch import nn

    out_onnx_path = Path(out_onnx_path)
    model = adapter.model.eval()
    # torchvision RetinaNet postprocessing knobs (fall back to the documented defaults).
    score_thresh = float(getattr(model, "score_thresh", 0.05))
    nms_thresh = float(getattr(model, "nms_thresh", 0.5))
    detections_per_img = int(getattr(model, "detections_per_img", 300))

    class RetinaRaw(nn.Module):
        def __init__(self, m: nn.Module) -> None:
            super().__init__()
            self.model = m

        def forward(self, pixel_values):
            orig_h = pixel_values.shape[-2]
            orig_w = pixel_values.shape[-1]
            # transform takes a list of [C,H,W]; it normalizes + aspect-resizes.
            images, _ = self.model.transform([pixel_values[0]], None)
            features = list(self.model.backbone(images.tensors).values())
            head_outputs = self.model.head(features)
            anchors = self.model.anchor_generator(images, features)  # list length 1

            cls_logits = head_outputs["cls_logits"]     # [1, A, C]
            bbox_reg = head_outputs["bbox_regression"]  # [1, A, 4]
            boxes = self.model.box_coder.decode_single(bbox_reg[0], anchors[0])  # [A,4] resized frame

            resized_h, resized_w = images.image_sizes[0]
            # resized -> input-pixel frame (per-axis scale; tensors so it tracks
            # dynamic input H/W under ONNX export).
            sx = orig_w / torch.as_tensor(resized_w, dtype=boxes.dtype, device=boxes.device)
            sy = orig_h / torch.as_tensor(resized_h, dtype=boxes.dtype, device=boxes.device)
            scale = torch.stack([sx, sy, sx, sy]).reshape(1, 4)
            boxes = boxes * scale

            scores = cls_logits[0].sigmoid()  # [A, C]
            return boxes.unsqueeze(0), scores.unsqueeze(0)  # [1,A,4], [1,A,C]

    # torchvision clips detections to the image inside its postprocess; EfficientNMS
    # does not, so the engine's boxes can spill past the edge. Have the runtime clip
    # (this engine's meta only — the ONNX path keeps its torchvision-internal clip).
    meta["clip_boxes"] = True

    wrapper = RetinaRaw(model)
    raw_path = out_onnx_path.with_suffix(".raw.onnx")
    export_raw_boxes_scores(wrapper, raw_path)
    append_efficientnms(
        raw_path,
        out_onnx_path,
        score_threshold=score_thresh,
        iou_threshold=nms_thresh,
        max_output_boxes=detections_per_img,
        box_coding=0,
        background_class=-1,
        score_activation=0,
        class_agnostic=0,
    )
