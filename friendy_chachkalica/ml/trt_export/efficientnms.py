"""Append a TensorRT ``EfficientNMS_TRT`` plugin node to a raw-output ONNX graph.

Some archs (torchvision RetinaNet, vendored YOLOX) bake decode + NMS into their
standard ONNX graph as *data-dependent* subgraphs (boolean-mask ``NonZero`` +
``NonMaxSuppression``) that TensorRT's optimizer refuses to compile ("data-
dependent shapes are only allowed in the top-level scope"). For those, we instead
export a graph that stops at **raw decoded boxes + per-class scores** (pure
tensor math, no data-dependent ops) and bolt on the fused ``EfficientNMS_TRT``
plugin, which does score-threshold + class-aware NMS + top-k in one op and emits
**fixed-size** outputs — exactly what TensorRT wants.

The plugin's four outputs (``num_detections``, ``detection_boxes``,
``detection_scores``, ``detection_classes``) are auto-detected and unpacked by
``trt_infer/session.py`` back into the canonical ``(boxes, scores, labels)``.
"""

from __future__ import annotations

from pathlib import Path

RAW_OUTPUTS = ["boxes", "scores"]  # [1, N, 4] and [1, N, num_classes]
# The plugin's fixed-size outputs, in the order EfficientNMS_TRT emits them.
NMS_OUTPUTS = ["num_detections", "detection_boxes", "detection_scores", "detection_classes"]


def export_raw_boxes_scores(wrapper, onnx_path, *, dummy_hw=(640, 640), opset: int = 17) -> None:
    """``torch.onnx.export`` a wrapper ``forward(pixel_values[B,3,H,W]) ->
    (boxes[B,N,4], scores[B,N,C])`` — the pre-NMS tensors EfficientNMS consumes.

    Dynamic axes on input H/W, on the box count ``N`` (which tracks H/W), and
    on batch — real batching only actually happens for a wrapper that carries
    the true batch dim through (see ``arch/yolox.py``'s fixed ``YOLOXRaw``);
    marking it dynamic here is harmless for a wrapper whose own output is
    always batch=1 (retinanet/fasterrcnn, unchanged) since a traced ONNX
    graph's real shape at export time doesn't depend on this metadata.
    """
    import torch

    dummy = torch.rand(1, 3, dummy_hw[0], dummy_hw[1])
    wrapper.eval()
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(onnx_path),
            input_names=["pixel_values"],
            output_names=RAW_OUTPUTS,
            dynamic_axes={
                "pixel_values": {0: "batch", 2: "height", 3: "width"},
                "boxes": {0: "batch", 1: "num_boxes"},
                "scores": {0: "batch", 1: "num_boxes"},
            },
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )


def append_efficientnms(
    raw_onnx_path,
    out_onnx_path,
    *,
    score_threshold: float,
    iou_threshold: float,
    max_output_boxes: int,
    box_coding: int = 0,          # 0 = xyxy corners, 1 = cxcywh
    background_class: int = -1,   # -1 = no background class
    score_activation: int = 0,    # 0 = scores already probabilities (we pre-sigmoid)
    class_agnostic: int = 0,      # 0 = class-aware NMS (matches torchvision batched_nms)
) -> None:
    """Read ``raw_onnx_path`` (outputs ``boxes``/``scores``), append an
    ``EfficientNMS_TRT`` node, and write ``out_onnx_path`` with the plugin's four
    fixed-size outputs.
    """
    import numpy as np
    import onnx
    import onnx_graphsurgeon as gs

    graph = gs.import_onnx(onnx.load(str(raw_onnx_path)))
    by_name = {o.name: o for o in graph.outputs}
    try:
        boxes, scores = by_name["boxes"], by_name["scores"]
    except KeyError as exc:  # pragma: no cover - guards a wiring mistake
        raise ValueError(
            f"raw ONNX must output 'boxes' and 'scores'; got {list(by_name)}"
        ) from exc

    # "batch" is a symbolic label, not a literal size — TensorRT resolves the
    # real per-call batch dim from the engine's optimization profile at build/
    # run time regardless of what's declared here (EfficientNMS_TRT natively
    # supports a batched input; its outputs, `[batch, ...]`, follow suit).
    num_det = gs.Variable("num_detections", dtype=np.int32, shape=("batch", 1))
    det_boxes = gs.Variable("detection_boxes", dtype=np.float32, shape=("batch", max_output_boxes, 4))
    det_scores = gs.Variable("detection_scores", dtype=np.float32, shape=("batch", max_output_boxes))
    det_classes = gs.Variable("detection_classes", dtype=np.int32, shape=("batch", max_output_boxes))

    node = gs.Node(
        op="EfficientNMS_TRT",
        name="EfficientNMS_TRT",
        inputs=[boxes, scores],
        outputs=[num_det, det_boxes, det_scores, det_classes],
        attrs={
            "plugin_version": "1",
            "background_class": int(background_class),
            "max_output_boxes": int(max_output_boxes),
            "score_threshold": float(score_threshold),
            "iou_threshold": float(iou_threshold),
            "box_coding": int(box_coding),
            "score_activation": int(score_activation),
            "class_agnostic": int(class_agnostic),
        },
    )
    graph.nodes.append(node)
    graph.outputs = [num_det, det_boxes, det_scores, det_classes]
    graph.cleanup().toposort()
    onnx.save(gs.export_onnx(graph), str(out_onnx_path))
