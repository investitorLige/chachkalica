"""RT-DETR ONNX exporter.

The HF RT-DETR head emits normalized ``cxcywh`` boxes (``pred_boxes``, in
``[0,1]`` over the model input) plus per-query class logits. The adapter's
``predict`` runs ``RTDetrImageProcessor.post_process_object_detection``
(``image_processing_rt_detr.py``): sigmoid the logits, ``topk`` the flattened
``[queries*classes]`` scores down to ``num_queries``, derive ``label = idx %
C`` / ``box = idx // C``, then scale boxes by ``target_sizes`` (the *original*
image size). The friendy step immediately re-normalizes by that same original
size, so the scaling cancels — **the friendy box is the model's normalized box**.

So the export wrapper bakes exactly the sigmoid + top-k selection and emits the
boxes in normalized ``[0,1]`` xyxy (``box_coords: "input_normalized"``). No NMS
(RT-DETR is NMS-free), no threshold in the graph (the service applies it, mirroring
post_process filtering after top-k). The service replicates the adapter's input
pipeline — longest-side resize to ``input_max_size``, ImageNet normalize, pad to a
multiple of 32 — which is mandatory: RT-DETR is *not* padding-invariant and rejects
non-multiple-of-32 inputs (verified). ``pixel_mask`` is omitted: it provably does
not change the output.

**float32 position embedding (export only).** HF's
``build_2d_sinusoidal_position_embedding`` does its sin/cos frequency arithmetic
in ``float64`` (see its docstring) and only casts to ``float32`` at the end. That
leaves ``Sin``/``Cos`` nodes typed ``double`` in the traced graph, and
onnxruntime has **no CPU kernel for double Sin/Cos** — the session fails to load
(``NOT_IMPLEMENTED : Could not find an implementation for Cos(7)``). We therefore
swap in a byte-identical float32 reimplementation *for the duration of the export
trace only*, then restore the original so the torch reference path is untouched.
The grid size stays dynamic (traced from the feature map's symbolic H/W), so
dynamic input H/W still works. The fp32-vs-fp64 embedding delta is ~1e-6 and well
inside the parity tolerance.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

try:
    from ..common import build_meta, export_detection_wrapper
except ImportError:  # run as a flat script
    from common import build_meta, export_detection_wrapper


def _f32_sinusoidal_position_embedding(
    height, width, embed_dim=256, temperature=10000.0, cls_token=False,
    device=None, dtype=None,
):
    """float32 twin of HF ``build_2d_sinusoidal_position_embedding`` — same layout
    ``[sin_h | cos_h | sin_w | cos_w]``, same row-major (H-outer) ordering, but the
    frequency arithmetic runs in float32 so the traced Sin/Cos nodes are float32
    (onnxruntime has no double Sin/Cos kernel)."""
    import torch

    if embed_dim % 4 != 0:
        raise ValueError(f"`embed_dim` must be divisible by 4, got {embed_dim}")
    pos_dim = embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32, device=device) / pos_dim
    omega = 1.0 / temperature**omega

    grid_h = torch.arange(height, dtype=torch.float32, device=device)
    grid_w = torch.arange(width, dtype=torch.float32, device=device)
    grid_h, grid_w = torch.meshgrid(grid_h, grid_w, indexing="ij")

    emb_h = grid_h.flatten().outer(omega)
    emb_w = grid_w.flatten().outer(omega)
    pos_embed = torch.cat([emb_h.sin(), emb_h.cos(), emb_w.sin(), emb_w.cos()], dim=1)
    if cls_token:
        pos_embed = torch.cat(
            [torch.zeros(1, embed_dim, dtype=torch.float32, device=device), pos_embed],
            dim=0,
        )
    return pos_embed.to(dtype if dtype is not None else torch.float32)


@contextmanager
def _float32_position_embedding():
    """Swap the AIFI sine-embedding builder for its float32 twin during export,
    restoring the original afterwards so the torch reference path is unchanged."""
    from transformers.models.rt_detr.modeling_rt_detr import RTDetrSinePositionEmbedding

    original = RTDetrSinePositionEmbedding.__dict__[
        "_cached_build_2d_sinusoidal_position_embedding"
    ]
    RTDetrSinePositionEmbedding._cached_build_2d_sinusoidal_position_embedding = (
        staticmethod(_f32_sinusoidal_position_embedding)
    )
    try:
        yield
    finally:
        RTDetrSinePositionEmbedding._cached_build_2d_sinusoidal_position_embedding = (
            original
        )


def export_rtdetr(adapter, *, num_classes, params, class_map, onnx_path: str | Path) -> dict:
    import torch
    from torch import nn

    model = adapter.model.eval()
    score_threshold = float(adapter.score_threshold)
    mean = [float(v) for v in adapter.image_mean]
    std = [float(v) for v in adapter.image_std]
    max_size = int(adapter.input_max_size)
    multiple = int(adapter.input_size_multiple)

    class RTDetrExport(nn.Module):
        def __init__(self, model: nn.Module) -> None:
            super().__init__()
            self.model = model

        def forward(self, pixel_values):
            out = self.model(pixel_values=pixel_values)
            logits = out.logits[0]        # [Q, C]
            boxes_n = out.pred_boxes[0]   # [Q, 4] normalized cxcywh
            cx, cy, w, h = boxes_n[:, 0], boxes_n[:, 1], boxes_n[:, 2], boxes_n[:, 3]
            xyxy = torch.stack(
                [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1
            )  # [Q, 4] normalized xyxy

            num_queries, num_cls = logits.shape[0], logits.shape[1]
            scores = torch.sigmoid(logits)  # focal-loss path (RT-DETR default)
            top_scores, top_idx = torch.topk(scores.flatten(), num_queries)
            labels = top_idx % num_cls
            box_idx = top_idx // num_cls
            return xyxy[box_idx], top_scores, labels.to(torch.int64)

    wrapper = RTDetrExport(model)
    with _float32_position_embedding():
        export_detection_wrapper(wrapper, onnx_path)

    return build_meta(
        arch="rtdetr",
        num_classes=num_classes,
        class_map=class_map,
        score_threshold=score_threshold,
        resize_mode="longest_side",
        max_size=max_size,
        multiple=multiple,
        pad_value=0.0,
        input_scale="unit",
        normalize={"mean": mean, "std": std},
        box_coords="input_normalized",
    )
