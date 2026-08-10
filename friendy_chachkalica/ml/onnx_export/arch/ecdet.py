"""ECDet ONNX exporter.

ECDet's head is RT-DETR's: ``pred_logits`` per query plus ``pred_boxes`` as
normalized ``cxcywh`` in ``[0, 1]`` over the model input. Upstream's
``PostProcessor`` sigmoids the logits, ``topk``s the flattened
``[queries * classes]`` scores down to ``num_top_queries`` (300), derives
``label = idx % C`` / ``box = idx // C``, then scales the boxes by
``orig_target_sizes``. ``ECDetAdapter.predict`` re-normalizes by that same size,
so the scaling cancels — **the friendy box is the model's normalized box**.

So this wrapper bakes exactly the sigmoid + top-k and emits normalized ``[0,1]``
xyxy (``box_coords: "input_normalized"``). No NMS (ECDet is set-based and
NMS-free), and no threshold in the graph: the service applies it after top-k,
mirroring the adapter. Because the graph carries no data-dependent NMS,
TensorRT compiles this standard export directly — ecdet has **no**
``trt_export/arch/`` prep, which is what puts it on the passthrough side of the
split (and therefore on a batch-1 engine profile, like rtdetr/rfdetr).

The service replicates the adapter's input pipeline: stretch-resize to a square
``input_max_size`` canvas (aspect ratio *not* preserved, matching upstream
ECDet's own ``Resize [640, 640]`` and
``ECDetAdapter._resize_image_with_scale``) plus ImageNet normalize, no padding.

**Static input H/W.** Unlike rtdetr's export this is not dynamic in H/W.
``ECTransformer`` generates its anchors once at build time from
``eval_spatial_size`` and caches them (decoder.py's ``_generate_anchors`` under
``if self.eval_spatial_size``), so the graph is only correct at the size it was
built for. A dynamic-H/W export would trace those cached anchors as constants and
silently produce wrong boxes at every other size, so the axis is pinned instead.

**``deploy()`` is destructive.** It folds the reparameterizable conv blocks *and*
replaces the score/bbox heads past ``eval_idx`` with ``nn.Identity``
(decoder.py's ``convert_to_deploy``), which would corrupt the adapter's model for
any later training or torch-path eval. So it runs on a ``deepcopy``, the same way
``arch/rfdetr.py`` guards ``LWDETR.export()``.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

try:
    from ..common import build_meta, export_detection_wrapper
except ImportError:  # run as a flat script
    from common import build_meta, export_detection_wrapper


def export_ecdet(adapter, *, num_classes, params, class_map, onnx_path: str | Path) -> dict:
    import torch
    from torch import nn

    score_threshold = float(adapter.score_threshold)
    mean = [float(v) for v in adapter.image_mean]
    std = [float(v) for v in adapter.image_std]
    num_top_queries = int(getattr(adapter, "num_top_queries", 300))

    # The adapter stretches every image onto this square canvas, so the service
    # resizes the same way ("square") and the graph never sees padding. With
    # resizing disabled the adapter has no fixed canvas at all, which a static
    # ONNX graph cannot express — and here it is doubly impossible, since the
    # decoder's anchors are built for one specific size.
    canvas_size = adapter._fixed_canvas_size()
    if canvas_size is None:
        raise ValueError(
            "ONNX export needs a fixed input size, but this pipeline disables "
            "resizing (input_max_size is unset or <= 0). Set input_max_size to "
            "the square canvas the exported model should run at."
        )
    canvas_size = int(canvas_size)

    model = deepcopy(adapter.model).eval()
    model.deploy()

    class ECDetExport(nn.Module):
        def __init__(self, model: nn.Module) -> None:
            super().__init__()
            self.model = model

        def forward(self, pixel_values):
            out = self.model(pixel_values)
            logits = out["pred_logits"][0]    # [Q, C]
            boxes_n = out["pred_boxes"][0]    # [Q, 4] normalized cxcywh
            cx, cy, w, h = boxes_n[:, 0], boxes_n[:, 1], boxes_n[:, 2], boxes_n[:, 3]
            xyxy = torch.stack(
                [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1
            )  # [Q, 4] normalized xyxy

            num_cls = logits.shape[1]
            scores = torch.sigmoid(logits)  # focal-loss path (ECDet default)
            top_scores, top_idx = torch.topk(scores.flatten(), num_top_queries)
            labels = top_idx % num_cls
            box_idx = top_idx // num_cls
            return xyxy[box_idx], top_scores, labels.to(torch.int64)

    wrapper = ECDetExport(model)
    export_detection_wrapper(
        wrapper,
        onnx_path,
        dummy_hw=(canvas_size, canvas_size),
        dynamic_input_hw=False,
    )

    return build_meta(
        arch="ecdet",
        num_classes=num_classes,
        class_map=class_map,
        score_threshold=score_threshold,
        resize_mode="square",
        size=canvas_size,
        multiple=0,
        pad_value=0.0,
        input_scale="unit",
        normalize={"mean": mean, "std": std},
        box_coords="input_normalized",
        # The decoder emits normalized cxcywh through a sigmoid, so a box centred
        # near an edge can extend past the canvas (cx - w/2 < 0) with no padding
        # involved at all. The torch path clamps those (adapters/ecdet.py's
        # predict() calls clip_xyxy), so the exported path must too or the same
        # checkpoint scores differently per format: an unclipped box has a larger
        # area, so IoU against an edge-touching ground truth drops and near-
        # threshold matches flip to misses. tests_clip_parity pairs these two
        # files by name and asserts they agree.
        clip_boxes=True,
    )
