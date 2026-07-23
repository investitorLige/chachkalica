"""TensorRT runtime session — the GPU counterpart to
``onnx_infer/session.py::OnnxModel``.

Owns the deserialized engine + execution context and runs one pre-processed
``[B,3,H,W]`` batch through it, returning the three Contract-A outputs
(``boxes``/``scores``/``labels``) in that canonical order — a single
``(boxes, scores, labels)`` triple for ``B == 1`` (unchanged from before, so
every existing single-image caller keeps working as-is), or a list of one
triple per image for ``B > 1``.

Device memory and the CUDA stream are managed with **torch** (present wherever a
GPU runtime runs), so there is no pycuda / cuda-python dependency: input bytes go
in a torch CUDA tensor whose ``data_ptr()`` is handed to TensorRT, and outputs —
whose detection count is data-dependent (NMS / top-k) — are captured through a
TensorRT output allocator that (re)allocates a torch buffer on demand.
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
    """

    class _TorchOutputAllocator(trt.IOutputAllocator):
        def __init__(self):
            super().__init__()
            self.buffers = {}  # name -> uint8 torch CUDA tensor
            self.shapes = {}   # name -> resolved output shape (tuple)

        def _reallocate(self, tensor_name, size):
            buf = torch.empty(int(size), dtype=torch.uint8, device="cuda")
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
        # Two graph output layouts, auto-detected by tensor name:
        #   * EfficientNMS_TRT (retinanet/yolox): 4 fixed-size outputs; unpack to
        #     (boxes, scores, labels) by slicing to num_detections.
        #   * passthrough (rtdetr/rfdetr): already (boxes, scores, labels).
        self._efficientnms = set(EFFICIENTNMS_OUTPUTS).issubset(set(self._output_names))
        if self._efficientnms:
            self._emit_order = list(EFFICIENTNMS_OUTPUTS)
        elif set(CANONICAL_OUTPUTS).issubset(set(self._output_names)):
            self._emit_order = list(CANONICAL_OUTPUTS)
        else:
            self._emit_order = list(self._output_names)

        self._Allocator = _build_allocator_class(trt, torch)

    def to(self, device) -> "TrtModel":
        # The engine is bound to the GPU it was built on; nothing to move. Kept for
        # interface parity with OnnxModel.to().
        return self

    def run(self, batched: np.ndarray) -> list:
        trt, torch = self._trt, self._torch

        batched = np.ascontiguousarray(batched, dtype=np.float32)
        batch_size = int(batched.shape[0])
        input_gpu = torch.from_numpy(batched).to(self._device)  # keep alive through execute

        self.context.set_input_shape(self.input_name, tuple(int(d) for d in batched.shape))
        self.context.set_tensor_address(self.input_name, int(input_gpu.data_ptr()))

        allocator = self._Allocator()
        for tname in self._output_names:
            self.context.set_output_allocator(tname, allocator)

        stream = torch.cuda.current_stream()
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 returned False")
        stream.synchronize()

        outputs = {}
        for tname in self._output_names:
            shape = allocator.shapes.get(tname)
            if shape is None:  # static output: shape known from the context
                shape = tuple(int(d) for d in self.context.get_tensor_shape(tname))
            dtype = _torch_dtype_for(trt, torch, self.engine.get_tensor_dtype(tname))
            numel = int(np.prod(shape)) if len(shape) else 1
            typed = self._torch.empty(0, dtype=dtype)
            elem_size = typed.element_size()
            buf = allocator.buffers[tname]
            flat = buf[: numel * elem_size].view(dtype)
            outputs[tname] = flat.reshape(shape).detach().cpu().numpy()

        ordered = [outputs[tname] for tname in self._emit_order]
        if self._efficientnms:
            triples = _unpack_efficientnms(ordered, batch_size)
        else:
            triples = _split_passthrough(ordered, batch_size)
        # B == 1: return the bare triple, exactly what every caller written
        # before batching existed still expects. B > 1: return the list so a
        # caller that actually submitted a real batch can tell images apart.
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
    """Slice already-batched passthrough outputs (leading dim = batch_size,
    e.g. rtdetr/rfdetr, which compile straight from their own batch-aware
    ONNX graph) into one per-image ``[boxes, scores, labels]`` triple.

    Unlike the EfficientNMS path, this repo doesn't yet build any passthrough
    arch's engine with a batch profile wider than 1, so this branch is
    currently only exercised at ``batch_size == 1`` in practice; it's written
    to generalize correctly (assuming the graph's own batch axis matches the
    input batch) rather than to special-case that.
    """
    boxes, scores, labels = ordered
    return [[boxes[i], scores[i], labels[i]] for i in range(batch_size)]
