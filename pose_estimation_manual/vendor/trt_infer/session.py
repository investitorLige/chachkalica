"""TensorRT runtime session — the GPU counterpart to
``onnx_infer/session.py::OnnxModel``.

Owns the deserialized engine + execution context and runs one pre-processed
``[B,3,H,W]`` batch through it, returning the three Contract-A outputs
(``boxes``/``scores``/``labels``) in that canonical order — a single
``(boxes, scores, labels)`` triple for ``B == 1`` (unchanged from before, so
every existing single-image caller keeps working as-is), or a list of one
triple per image for ``B > 1``. One arch (rtmo) isn't a detection triple at
all — see ``RTMO_OUTPUTS`` below — and gets a same-shaped ``(dets, keypoints)``
pair instead, per image, for its ArchHandler to turn into Contract A itself.

Device memory and the CUDA stream are managed with **torch** (present wherever a
GPU runtime runs), so there is no pycuda / cuda-python dependency: input bytes go
in a torch CUDA tensor whose ``data_ptr()`` is handed to TensorRT, and outputs —
whose detection count is data-dependent (NMS / top-k) — are captured through a
single TensorRT output allocator, built once per execution context, whose torch
buffers grow on demand and are then reused for the process's lifetime.

A second arch (rfdetr_raw) also isn't a plain Contract-A triple — see
``RFDETR_RAW_OUTPUTS`` below — its graph was never run through the
``rfdetr``/``rtdetr`` export wrapper, so it still emits raw ``(dets, labels)``
for its ArchHandler to decode.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Contract-A output tensor names, in the order downstream expects (boxes, scores,
# labels). TensorRT preserves the ONNX graph's tensor names, so we map by name.
CANONICAL_OUTPUTS = ("boxes", "scores", "labels")

# EfficientNMS_TRT plugin outputs (retinanet / yolox engines). Fixed-size; the
# valid count is in ``num_detections``. Unpacked back into (boxes, scores, labels).
EFFICIENTNMS_OUTPUTS = ("num_detections", "detection_boxes", "detection_scores", "detection_classes")

# RTMO (pose) engine outputs. Not a detection triple at all — see
# onnx_infer/arch/rtmo.py — so this pair is handed to the ArchHandler as-is,
# per-image, rather than unpacked into (boxes, scores, labels) here. Dynamic
# per-image row count (NMS'd), same as EfficientNMS_TRT's num_detections case,
# but with no fixed-size padding to slice against: TensorRT reports each
# image's real count directly via the output allocator (see notify_shape).
RTMO_OUTPUTS = ("dets", "keypoints")

# RF-DETR raw-export engine outputs — see onnx_infer/arch/rfdetr_raw.py. A plain
# torch.onnx.export of the RF-DETR head, with none of the wrapped exporters'
# decode baked in: ``dets`` is boxes only (no score column, unlike rtmo's),
# ``labels`` is raw per-class logits including the background slot. Batch-first,
# like rtmo — this graph was never had its batch axis indexed away.
RFDETR_RAW_OUTPUTS = ("dets", "labels")


def _torch_dtype_for(trt, torch, trt_dtype):
    """Map a TensorRT ``DataType`` to the matching torch dtype (version-tolerant)."""
    mapping = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.INT32: torch.int32,
    }
    for name, tdtype in (("INT64", torch.int64), ("BOOL", torch.bool), ("INT8", torch.int8)):
        if hasattr(trt.DataType, name):
            mapping[getattr(trt.DataType, name)] = tdtype
    if trt_dtype not in mapping:
        raise RuntimeError(f"unsupported TensorRT output dtype: {trt_dtype}")
    return mapping[trt_dtype]


def _build_allocator_class(trt, torch):
    """A ``trt.IOutputAllocator`` that backs each output with a torch CUDA buffer.

    Defined lazily so importing this module doesn't require tensorrt. Handles both
    the TRT < 10 (``reallocate_output``) and TRT >= 10 (``*_async``) signatures.

    One instance serves an execution context for its whole life: TensorRT's contract
    is that an allocator resizes its own buffer in place across calls, and *only* the
    grow path may allocate. See :meth:`TrtModel.run` for what building one of these
    per call did to this service.
    """

    class _TorchOutputAllocator(trt.IOutputAllocator):
        def __init__(self):
            super().__init__()
            self.buffers = {}  # name -> uint8 torch CUDA tensor
            self.shapes = {}   # name -> resolved output shape (tuple)

        def _reallocate(self, tensor_name, size):
            # Grow-only: keep the buffer we already hold whenever it is big enough,
            # so a steady-state stream of same-shaped frames stops allocating after
            # its first call. Reallocating unconditionally here leaks the GPU, badly
            # — see run().
            size = int(size)
            buf = self.buffers.get(tensor_name)
            if buf is None or buf.numel() < size:
                # max(size, 1): a 0-byte torch tensor's data_ptr() is 0, and handing
                # TensorRT a null pointer is how it reports allocation failure.
                buf = torch.empty(max(size, 1), dtype=torch.uint8, device="cuda")
                self.buffers[tensor_name] = buf
            return int(buf.data_ptr())

        def reallocate_output(self, tensor_name, memory, size, alignment):  # TRT < 10
            return self._reallocate(tensor_name, size)

        def reallocate_output_async(self, tensor_name, memory, size, alignment, stream):  # TRT >= 10
            return self._reallocate(tensor_name, size)

        def notify_shape(self, tensor_name, shape):
            self.shapes[tensor_name] = tuple(int(d) for d in shape)

    return _TorchOutputAllocator


class TrtModel:
    """A loaded TensorRT engine + its meta, ready to run one ``[B,3,H,W]`` batch."""

    def __init__(self, engine_path, meta, device="cuda") -> None:
        import tensorrt as trt
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "TensorRT runtime requires a CUDA GPU, but torch.cuda.is_available() is False"
            )

        self._trt = trt
        self._torch = torch
        self.meta = meta
        self.path = Path(engine_path)
        # Engines only run on the GPU they were built for; coerce any non-cuda
        # request to cuda rather than silently mis-binding.
        name = str(device).lower()
        self._device = "cuda" if ("cuda" in name or "gpu" in name) else "cuda"

        self._logger = trt.Logger(trt.Logger.WARNING)
        # Register the standard TensorRT plugins (EfficientNMS_TRT et al.) before
        # deserializing: engines for the fused-NMS archs (retinanet, yolox,
        # fasterrcnn) embed that plugin, and deserialization fails if its creator
        # isn't registered in this process. The builder registers them too, so a
        # build+load in one process worked without this — but a fresh runtime
        # (the real trt_infer path) must register them itself.
        trt.init_libnvinfer_plugins(self._logger, "")
        runtime = trt.Runtime(self._logger)
        self.engine = runtime.deserialize_cuda_engine(self.path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize TensorRT engine: {self.path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(
                f"failed to create a TensorRT execution context for {self.path} — "
                "TensorRT logged the reason above (commonly a device-memory allocation "
                "failure sized far larger than the model should need, which usually means "
                "the engine's optimization profile is too wide for this graph; try a "
                "narrower --min-hw/--max-hw)"
            )

        self._input_names, self._output_names = [], []
        for i in range(self.engine.num_io_tensors):
            tname = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(tname) == trt.TensorIOMode.INPUT:
                self._input_names.append(tname)
            else:
                self._output_names.append(tname)
        self.input_name = self._input_names[0]
        # Four graph output layouts, auto-detected by tensor name:
        #   * EfficientNMS_TRT (retinanet/yolox): 4 fixed-size outputs; unpack to
        #     (boxes, scores, labels) by slicing to num_detections.
        #   * passthrough (rtdetr/rfdetr/scrfd): already (boxes, scores, labels)
        #     (or, for scrfd, raw outputs an ArchHandler still needs to NMS).
        #   * rtmo (pose): (dets, keypoints) — not a detection triple; handed to
        #     the ArchHandler as a per-image pair, unchanged.
        #   * rfdetr_raw: (dets, labels) — also not a detection triple; same idea,
        #     handed to its ArchHandler as a per-image pair.
        self._efficientnms = set(EFFICIENTNMS_OUTPUTS).issubset(set(self._output_names))
        self._rtmo = set(RTMO_OUTPUTS) == set(self._output_names)
        self._rfdetr_raw = set(RFDETR_RAW_OUTPUTS) == set(self._output_names)
        if self._efficientnms:
            self._emit_order = list(EFFICIENTNMS_OUTPUTS)
        elif self._rtmo:
            self._emit_order = list(RTMO_OUTPUTS)
        elif self._rfdetr_raw:
            self._emit_order = list(RFDETR_RAW_OUTPUTS)
        elif set(CANONICAL_OUTPUTS).issubset(set(self._output_names)):
            self._emit_order = list(CANONICAL_OUTPUTS)
        else:
            self._emit_order = list(self._output_names)

        # One allocator, registered once, for this context's whole life. TensorRT's
        # python bindings tie a registered allocator's lifetime to the execution
        # context, so a per-call allocator is never collected and neither are its
        # torch buffers. That is what wedged this service on 2026-08-07: ~17h45m of
        # continuous two-camera inference on a 117MB pair of engines climbed to
        # 9.45GiB of *live* (not cached) torch allocations, after which every call
        # failed forever on the 4-byte num_detections output, once every 2-5s, until
        # the container was restarted by hand.
        self._allocator = _build_allocator_class(trt, torch)()
        for tname in self._output_names:
            self.context.set_output_allocator(tname, self._allocator)

    def to(self, device) -> "TrtModel":
        # The engine is bound to the GPU it was built on; nothing to move. Kept for
        # interface parity with OnnxModel.to().
        return self

    def run(self, batched: np.ndarray) -> list:
        """``[B,3,H,W]`` numpy float32 -> host numpy detections.

        The output allocator's buffers are reused across calls, which is safe here
        only because every output is copied off the GPU below before this returns —
        nothing a caller holds points into a buffer the next call overwrites. Keep it
        that way: if this ever grows a torch-out path (as the upstream repo's
        ``run_torch`` did), that path has to clone its outputs.
        """
        trt, torch = self._trt, self._torch

        batched = np.ascontiguousarray(batched, dtype=np.float32)
        batch_size = int(batched.shape[0])
        input_gpu = torch.from_numpy(batched).to(self._device)  # keep alive through execute

        self.context.set_input_shape(self.input_name, tuple(int(d) for d in batched.shape))
        self.context.set_tensor_address(self.input_name, int(input_gpu.data_ptr()))

        # The allocator is registered once, in __init__, and lives as long as this
        # context — only its per-call shapes are stale. Drop them so an output whose
        # shape TensorRT doesn't notify this time falls back to the context's own
        # shape rather than silently reusing the previous call's.
        self._allocator.shapes.clear()

        stream = torch.cuda.current_stream()
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 returned False")
        stream.synchronize()

        outputs = {}
        for tname in self._output_names:
            shape = self._allocator.shapes.get(tname)
            if shape is None:  # static output: shape known from the context
                shape = tuple(int(d) for d in self.context.get_tensor_shape(tname))
            dtype = _torch_dtype_for(trt, torch, self.engine.get_tensor_dtype(tname))
            numel = int(np.prod(shape)) if len(shape) else 1
            typed = self._torch.empty(0, dtype=dtype)
            elem_size = typed.element_size()
            buf = self._allocator.buffers[tname]
            flat = buf[: numel * elem_size].view(dtype)
            # .cpu() is the copy that makes buffer reuse safe — see run()'s docstring.
            outputs[tname] = flat.reshape(shape).detach().cpu().numpy()

        ordered = [outputs[tname] for tname in self._emit_order]
        if self._efficientnms:
            triples = _unpack_efficientnms(ordered, batch_size)
        elif self._rtmo:
            triples = _split_rtmo(ordered, batch_size)
        elif self._rfdetr_raw:
            triples = _split_rfdetr_raw(ordered, batch_size)
        else:
            triples = _split_passthrough(ordered, batch_size)
        # B == 1: return the bare triple (or, for rtmo/rfdetr_raw, pair), exactly
        # what every caller written before batching existed still expects. B > 1:
        # return the list so a caller that actually submitted a real batch can
        # tell images apart.
        return triples[0] if batch_size == 1 else triples


def _unpack_efficientnms(ordered: list, batch_size: int) -> list:
    """[num_detections, boxes, scores, classes] (fixed-size *per image*) ->
    one ``[boxes[Ni,4], scores[Ni], labels[Ni]]`` triple per image, each sliced
    to that image's own valid detection count."""
    num_det, det_boxes, det_scores, det_classes = ordered
    num_det = np.asarray(num_det).reshape(batch_size, -1)[:, 0]
    det_boxes = np.asarray(det_boxes).reshape(batch_size, -1, 4)
    det_scores = np.asarray(det_scores).reshape(batch_size, -1)
    det_classes = np.asarray(det_classes).reshape(batch_size, -1)
    triples = []
    for i in range(batch_size):
        n = int(num_det[i])
        triples.append([det_boxes[i, :n], det_scores[i, :n], det_classes[i, :n]])
    return triples


def _split_passthrough(ordered: list, batch_size: int) -> list:
    """Passthrough outputs -> one ``[boxes, scores, labels]`` triple per image.

    **The passthrough graphs have no batch axis.** Both export wrappers
    (``onnx_export/arch/rfdetr.py``, ``.../rtdetr.py``) index the batch away
    (``boxes_n, logits = boxes_n[0], logits[0]``) before baking the head math,
    so the graph emits ``boxes[N,4] / scores[N] / labels[N]`` where **N is the
    detection count**, not the batch size. Their engines are built with a
    batch profile of exactly 1 to match.

    This used to read ``boxes[i], scores[i], labels[i]``, which at
    ``batch_size == 1`` returned detection *zero* — ``(4,) / () / ()`` — and
    silently discarded every other detection. ``postprocess.to_friendy`` then
    reshaped that lone box to a valid ``(1, 6)`` result, so nothing raised: an
    RF-DETR engine emitting 300 detections per crop delivered exactly one, and
    a person could only ever carry a single PPE label.

    The batch axis is therefore detected by rank rather than assumed: rank-2
    ``boxes`` means no batch axis (the current exporters), rank-3 means a
    genuinely batch-aware graph, should one ever be exported.
    """
    boxes, scores, labels = ordered
    if np.asarray(boxes).ndim <= 2:
        # No batch axis: the whole output *is* this one image's detections.
        if batch_size != 1:
            raise RuntimeError(
                f"passthrough engine returned un-batched outputs (boxes shape "
                f"{np.asarray(boxes).shape}) for a batch of {batch_size}. Its graph "
                "collapses the batch axis, so it must be run one image at a time — "
                "build the engine with a batch-1 profile, or lower infer_batch_size."
            )
        return [[boxes, scores, labels]]
    return [[boxes[i], scores[i], labels[i]] for i in range(batch_size)]


def _split_rtmo(ordered: list, batch_size: int) -> list:
    """``[dets[B,N,5], keypoints[B,N,17,3]]`` -> one ``[dets_i, keypoints_i]``
    pair per image. Always batch-first — the export never collapses the batch
    axis the way the passthrough archs' does (see ``_split_passthrough``) — so
    unlike that function this needs no rank-based fallback."""
    dets, keypoints = ordered
    dets = np.asarray(dets)
    keypoints = np.asarray(keypoints)
    return [[dets[i], keypoints[i]] for i in range(batch_size)]


def _split_rfdetr_raw(ordered: list, batch_size: int) -> list:
    """``[dets[B,300,4], labels[B,300,5]]`` -> one ``[dets_i, labels_i]`` pair
    per image. Batch-first like rtmo — this graph's batch axis was never
    indexed away (unlike the wrapped ``rfdetr``/``rtdetr`` exports)."""
    dets, labels = ordered
    dets = np.asarray(dets)
    labels = np.asarray(labels)
    return [[dets[i], labels[i]] for i in range(batch_size)]
