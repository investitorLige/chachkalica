"""Faster R-CNN → TensorRT via EfficientNMS (two-stage surgery).

Faster R-CNN's standard ONNX bakes data-dependent NMS/top-k in *two* places — the
RPN's proposal NMS (feeding RoIAlign) and the RoI head's per-class NMS — and the
latter lands inside ONNX ``If`` subgraphs. TensorRT compiles neither: its Myelin
backend rejects data-dependent shapes anywhere but the top-level scope ("Data-
dependent shapes are only allowed in the top-level scope"), so the passthrough
graph parses but fails at build. (The onnxruntime fp16 converter also mangles
those ``If`` subgraphs — same root cause.)

So, like RetinaNet/YOLOX, we re-export a graph that stops at **raw decoded boxes +
per-class scores** using only shape-dependent (never data-dependent) ops, and let
``EfficientNMS_TRT`` do the final threshold + class-aware NMS + top-k. Two changes
are specific to the two-stage design:

  * **RPN**: the variable proposal NMS is replaced by a *fixed* top-K selection by
    objectness (``rpn_post_nms_top_n``), so RoIAlign always sees a static number
    of proposals. This drops the RPN's NMS-driven proposal diversity, so the
    engine's outputs are an *approximation* of the torch/ONNX model, not a bit-
    exact match — the accepted cost of a compilable two-stage graph.
  * **per-class boxes**: ``FastRCNNPredictor`` regresses a distinct box per class,
    so we feed ``EfficientNMS_TRT`` 4-D boxes ``[1, K, C, 4]`` (the plugin's
    per-class box form) alongside scores ``[1, K, C]`` — class-aware NMS then uses
    each class's own box, matching torchvision's ``batched_nms`` over class ids.

Boxes are decoded in torchvision's internal resized frame and mapped back to the
graph's input-pixel frame (a single per-axis scale, as in ``retinanet.py``), so
the meta stays ``resize_mode: none`` / ``box_coords: input_pixels``. Background
(class 0) is dropped before NMS, so ``background_class=-1``.
"""

from __future__ import annotations

from pathlib import Path

try:
    from ..efficientnms import append_efficientnms
except ImportError:  # run flat (cwd on sys.path)
    try:
        from trt_export.efficientnms import append_efficientnms  # type: ignore
    except ImportError:
        from efficientnms import append_efficientnms  # type: ignore

INPUT_NAME = "pixel_values"  # graph input; matches efficientnms.export_raw_boxes_scores


def prep_fasterrcnn(adapter, meta: dict, out_onnx_path) -> None:
    import torch
    from torch import nn
    import torch.nn.functional as F
    from torchvision.models.detection.rpn import concat_box_prediction_layers
    from torchvision.ops import boxes as box_ops

    out_onnx_path = Path(out_onnx_path)
    model = adapter.model.eval()

    # torchvision RoI-head postprocessing knobs (documented defaults if unset).
    score_thresh = float(getattr(model.roi_heads, "score_thresh", 0.05))
    nms_thresh = float(getattr(model.roi_heads, "nms_thresh", 0.5))
    detections_per_img = int(getattr(model.roi_heads, "detections_per_img", 100))
    # Fixed proposal count RoIAlign will always receive (was the RPN's variable
    # post-NMS count). rpn.post_nms_top_n() returns the eval value in .eval().
    post_nms_top_n = int(model.rpn.post_nms_top_n())

    class FasterRCNNRaw(nn.Module):
        def __init__(self, m: nn.Module, top_k: int) -> None:
            super().__init__()
            self.model = m
            self.top_k = int(top_k)

        def forward(self, pixel_values):
            orig_h = pixel_values.shape[-2]
            orig_w = pixel_values.shape[-1]
            # transform: ImageNet normalize + aspect-resize (a single [C,H,W] image).
            images, _ = self.model.transform([pixel_values[0]], None)
            features = self.model.backbone(images.tensors)  # OrderedDict of FPN levels
            feature_list = list(features.values())

            # ---- RPN, but with a FIXED top-K proposal selection (no proposal NMS) ----
            objectness, pred_bbox_deltas = self.model.rpn.head(feature_list)
            anchors = self.model.rpn.anchor_generator(images, feature_list)  # list len 1
            objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
            proposals = self.model.rpn.box_coder.decode(pred_bbox_deltas, anchors)  # [A,1,4]
            proposals = proposals.reshape(-1, 4)                                    # [A,4]
            obj = objectness.reshape(-1)                                            # [A]

            # Pad by top_k low-score sentinels so topk(k=top_k) is valid for ANY input
            # size (a tiny image can have < top_k anchors). Sentinel score -1 < any
            # sigmoid prob, and sentinel boxes are degenerate; the box head's own
            # score-threshold in EfficientNMS drops the rare survivor.
            k = self.top_k
            pad_scores = torch.full((k,), -1.0, dtype=obj.dtype, device=obj.device)
            pad_boxes = torch.zeros((k, 4), dtype=proposals.dtype, device=proposals.device)
            obj = torch.cat([obj, pad_scores], dim=0)
            proposals = torch.cat([proposals, pad_boxes], dim=0)
            top_idx = torch.topk(obj, k=k, dim=0).indices  # [K]
            props = proposals[top_idx]                     # [K,4], resized frame
            props = box_ops.clip_boxes_to_image(props, images.image_sizes[0])

            # ---- RoIAlign + box head + per-class decode (all shape-dependent) ----
            box_features = self.model.roi_heads.box_roi_pool(features, [props], images.image_sizes)
            box_features = self.model.roi_heads.box_head(box_features)
            class_logits, box_regression = self.model.roi_heads.box_predictor(box_features)
            pred_boxes = self.model.roi_heads.box_coder.decode(box_regression, [props])  # [K,C+1,4]
            pred_scores = F.softmax(class_logits, -1)                                     # [K,C+1]

            # Drop the background class (index 0); torchvision does the same.
            boxes_fg = pred_boxes[:, 1:, :]   # [K, C, 4], resized frame
            scores_fg = pred_scores[:, 1:]    # [K, C]

            # resized -> input-pixel frame (per-axis scale; tensors so it tracks
            # dynamic input H/W under ONNX export), as in retinanet.py.
            resized_h, resized_w = images.image_sizes[0]
            sx = orig_w / torch.as_tensor(resized_w, dtype=boxes_fg.dtype, device=boxes_fg.device)
            sy = orig_h / torch.as_tensor(resized_h, dtype=boxes_fg.dtype, device=boxes_fg.device)
            scale = torch.stack([sx, sy, sx, sy]).reshape(1, 1, 4)
            boxes_fg = boxes_fg * scale

            # EfficientNMS per-class form: boxes [1,K,C,4], scores [1,K,C].
            return boxes_fg.unsqueeze(0), scores_fg.unsqueeze(0)

    # EfficientNMS doesn't clip to the image; torchvision's postprocess does, so the
    # runtime must (this engine's meta only — the ONNX path keeps its internal clip).
    meta["clip_boxes"] = True

    wrapper = FasterRCNNRaw(model, post_nms_top_n)
    raw_path = out_onnx_path.with_suffix(".raw.onnx")
    _export_raw_perclass(wrapper, raw_path)
    append_efficientnms(
        raw_path,
        out_onnx_path,
        score_threshold=score_thresh,
        iou_threshold=nms_thresh,
        max_output_boxes=detections_per_img,
        box_coding=0,          # xyxy corners
        background_class=-1,   # background already dropped
        score_activation=0,    # scores are already softmax probabilities
        class_agnostic=0,      # class-aware NMS (matches torchvision batched_nms)
    )


def _export_raw_perclass(wrapper, onnx_path, *, dummy_hw=(640, 640), opset: int = 17) -> None:
    """Export ``forward(pixel_values[1,3,H,W]) -> (boxes[1,K,C,4], scores[1,K,C])``.

    Unlike the single-stage ``export_raw_boxes_scores``, K (proposals) and C
    (classes) are fixed, so only the input H/W are dynamic.
    """
    import torch

    dummy = torch.rand(1, 3, dummy_hw[0], dummy_hw[1])
    wrapper.eval()
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(onnx_path),
            input_names=[INPUT_NAME],
            output_names=["boxes", "scores"],
            dynamic_axes={INPUT_NAME: {0: "batch", 2: "height", 3: "width"}},
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )
