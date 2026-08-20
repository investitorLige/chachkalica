"""D-FINE ONNX exporter.

D-FINE's head is RT-DETR's: ``logits`` per query plus ``pred_boxes`` as
normalized ``cxcywh`` in ``[0, 1]`` over the model input — the FDR corner
distribution is decoded back to a box *inside* the model, so nothing about the
distribution refinement is visible at the output boundary. ``DFineAdapter.predict``
runs the very same ``RTDetrImageProcessor.post_process_object_detection`` RT-DETR's
adapter does (D-FINE ships no image processor of its own): sigmoid the logits,
``topk`` the flattened ``[queries*classes]`` scores down to ``num_queries``, derive
``label = idx % C`` / ``box = idx // C``, then scale by ``target_sizes``. The friendy
step re-normalizes by that same size, so the scaling cancels — **the friendy box is
the model's normalized box**.

So this wrapper bakes exactly the sigmoid + top-k and emits normalized ``[0,1]``
xyxy (``box_coords: "input_normalized"``). No NMS (D-FINE is set-based and
NMS-free), no threshold in the graph (the service applies it, mirroring
post_process filtering after top-k). Because the graph carries no data-dependent
NMS, TensorRT compiles this standard export directly — dfine has **no**
``trt_export/arch/`` prep.

The service replicates the adapter's input pipeline — stretch-resize to a square
``input_max_size`` canvas (aspect ratio not preserved, matching upstream D-FINE's
own ``Resize((640, 640))``, inherited from RT-DETR, and
``DFineAdapter._resize_image_with_scale``) plus ImageNet normalize. No padding:
the square side is already a multiple of 32, and a padded input would put the
decoder's reference points on dead pixels. ``pixel_mask`` is omitted: ``DFineModel.forward``
unpacks and discards the downsampled mask (see ``DFineAdapter._fixed_canvas_size``),
so it provably does not change the output.

**Batch-aware.** ``num_queries`` is a fixed constant per image, so the wrapper
does the sigmoid/top-k/gather per row of the batch instead of indexing batch
away, and emits ``(boxes[B,K,4], scores[B,K], labels[B,K])`` — a real batch axis
TensorRT can build a wider optimization profile around (see
``trt_export/arch/__init__.py``'s ``BATCH_AWARE_ARCHS``).

**float32 position embedding (export only).** D-FINE carries a byte-identical copy
of RT-DETR's ``build_2d_sinusoidal_position_embedding`` (verified: the two module
sources match exactly), so it inherits the same export defect — the frequency
arithmetic runs in ``float64`` and only casts to ``float32`` at the end, leaving
``Sin``/``Cos`` nodes typed ``double`` in the traced graph, which onnxruntime has
no CPU kernel for (``NOT_IMPLEMENTED : Could not find an implementation for
Cos(7)``). We reuse ``arch/rtdetr.py``'s float32 twin, swapped onto
``DFineSinePositionEmbedding`` for the duration of the export trace only.

**Dynamic input H/W**, like rtdetr's export and unlike ecdet's. The obvious worry
is D-FINE's encoder-level anchor grid: ``DFineModel.forward`` builds
``spatial_shapes_list`` from the feature maps and hands that tuple to an
``lru_cache``d ``generate_anchors``, which looks exactly like the build-time anchor
caching that forces ecdet's export static. It isn't — the cache key is the *traced*
spatial shape, and the ``torch.arange``/grid arithmetic inside ``_cached_generate_anchors``
traces symbolically, so the grid follows the input. (The genuinely static path,
``config.anchor_image_size`` pre-generating ``self.anchors`` at ``__init__``, is off:
that field defaults to ``None`` and the adapter never sets it.) Verified rather than
assumed: an export traced at 320 and then fed 256 matches torch-at-256 to 2.4e-7 on
box coordinates. The meta still says ``resize_mode: "square"``, so in practice the
service only ever feeds the canvas size — the dynamic axis is headroom, and
``profile.py`` still derives a fully-static TensorRT profile from that square mode.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

try:
    from ..common import build_meta, export_detection_wrapper
    from .rtdetr import _f32_sinusoidal_position_embedding
except ImportError:  # run as a flat script
    from common import build_meta, export_detection_wrapper
    from arch.rtdetr import _f32_sinusoidal_position_embedding


@contextmanager
def _float32_position_embedding():
    """Swap D-FINE's AIFI sine-embedding builder for RT-DETR's float32 twin during
    export, restoring the original afterwards so the torch reference path is
    unchanged. Safe to share the twin: the two ``build_2d_sinusoidal_position_embedding``
    implementations are byte-identical in transformers."""
    from transformers.models.d_fine.modeling_d_fine import DFineSinePositionEmbedding

    original = DFineSinePositionEmbedding.__dict__[
        "_cached_build_2d_sinusoidal_position_embedding"
    ]
    DFineSinePositionEmbedding._cached_build_2d_sinusoidal_position_embedding = (
        staticmethod(_f32_sinusoidal_position_embedding)
    )
    try:
        yield
    finally:
        DFineSinePositionEmbedding._cached_build_2d_sinusoidal_position_embedding = (
            original
        )


def export_dfine(adapter, *, num_classes, params, class_map, onnx_path: str | Path) -> dict:
    import torch
    from torch import nn

    model = adapter.model.eval()
    score_threshold = float(adapter.score_threshold)
    mean = [float(v) for v in adapter.image_mean]
    std = [float(v) for v in adapter.image_std]
    # The adapter stretches every image onto this square canvas, so the service
    # resizes the same way ("square") and the graph never sees padding. With
    # resizing disabled the adapter has no fixed canvas at all (it pads to the
    # batch max), which a static ONNX graph cannot express.
    canvas_size = adapter._fixed_canvas_size()
    if canvas_size is None:
        raise ValueError(
            "ONNX export needs a fixed input size, but this pipeline disables "
            "resizing (input_max_size is unset or <= 0). Set input_max_size to "
            "the square canvas the exported model should run at."
        )
    canvas_size = int(canvas_size)

    # The adapter's own post-processing branches on this (it forwards
    # `use_focal_loss` to post_process_object_detection); the softmax branch also
    # drops a trailing background class and picks an argmax rather than a
    # flattened top-k, so baking sigmoid here would silently mean something else.
    # Every shipped D-FINE config is focal (DFineConfig's default), so refuse
    # rather than guess if one ever isn't.
    if not getattr(model.config, "use_focal_loss", True):
        raise ValueError(
            "D-FINE ONNX export bakes the focal-loss (sigmoid + flattened top-k) "
            "decode path, but this checkpoint's config sets use_focal_loss=False, "
            "whose torch path is a softmax argmax over classes with a background "
            "class dropped. Exporting it would silently disagree with predict()."
        )

    class DFineExport(nn.Module):
        def __init__(self, model: nn.Module) -> None:
            super().__init__()
            self.model = model

        def forward(self, pixel_values):
            out = self.model(pixel_values=pixel_values)
            logits = out.logits          # [B, Q, C]
            boxes_n = out.pred_boxes     # [B, Q, 4] normalized cxcywh
            cx, cy, w, h = (
                boxes_n[..., 0], boxes_n[..., 1], boxes_n[..., 2], boxes_n[..., 3]
            )
            xyxy = torch.stack(
                [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1
            )  # [B, Q, 4] normalized xyxy

            num_queries, num_cls = logits.shape[1], logits.shape[2]
            scores = torch.sigmoid(logits)  # focal-loss path (D-FINE default)
            flat_scores = scores.reshape(scores.shape[0], -1)  # [B, Q*C]
            top_scores, top_idx = torch.topk(flat_scores, num_queries, dim=1)  # [B, K]
            labels = top_idx % num_cls
            box_idx = top_idx // num_cls
            # Per-batch-row gather: box_idx[b, k] indexes xyxy[b], not a shared
            # first dim, so this can't be a plain xyxy[box_idx] fancy-index
            # (that would index the batch axis itself once B > 1).
            boxes_out = torch.gather(
                xyxy, 1, box_idx.unsqueeze(-1).expand(-1, -1, 4)
            )  # [B, K, 4]
            return boxes_out, top_scores, labels.to(torch.int64)

    wrapper = DFineExport(model)
    with _float32_position_embedding():
        export_detection_wrapper(
            wrapper, onnx_path, dummy_hw=(canvas_size, canvas_size), batch_aware=True
        )

    return build_meta(
        arch="dfine",
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
        # involved at all. The torch path clamps those (adapters/dfine.py's
        # predict() calls clip_xyxy), so the exported path must too or the same
        # checkpoint scores differently per format: an unclipped box has a larger
        # area, so IoU against an edge-touching ground truth drops and near-
        # threshold matches flip to misses. tests_clip_parity pairs these two
        # files by name and asserts they agree.
        clip_boxes=True,
    )
