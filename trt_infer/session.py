"""TensorRT runtime session — the GPU counterpart to
``onnx_infer/session.py::OnnxModel``.

Owns the deserialized engine + execution context and runs one pre-processed
``[B,3,H,W]`` batch through it, returning the three Contract-A outputs
(``boxes``/``scores``/``labels``) in that canonical order. One arch (rtmo) isn't
a detection triple at all — see ``RTMO_OUTPUTS`` below — and gets a same-shaped
``(dets, keypoints)`` pair instead, per image, for its ArchHandler to turn into
Contract A itself.

Two entry points over one implementation:

* :meth:`TrtModel.run_torch` — torch in, torch out, everything staying in device
  memory. **Always** a list of ``B`` triples. This is the path
  :class:`~trt_infer.adapter.TrtAdapter` uses, so a person crop never round-trips
  through the host.
* :meth:`TrtModel.run` — the original numpy contract, kept verbatim for callers
  that predate ``run_torch`` (``person_model_test/run_yolox_export_eval.py``,
  ``inferlica/benchmark/kernel_util.py``): numpy in, numpy out, a bare
  ``(boxes, scores, labels)`` triple when ``B == 1`` and a list of triples
  otherwise. It is a thin wrapper over ``run_torch``, so the TensorRT plumbing
  exists once and the two paths cannot drift.

Device memory and the CUDA stream are managed with **torch** (present wherever a
GPU runtime runs), so there is no pycuda / cuda-python dependency: input bytes go
in a torch CUDA tensor whose ``data_ptr()`` is handed to TensorRT, and outputs —
whose detection count is data-dependent (NMS / top-k) — are captured through a
single TensorRT output allocator, built once per execution context, whose torch
buffers grow on demand and are then reused for the process's lifetime.
"""

from __future__ import annotations

import json
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
# per-image row count (the graph NMSes itself), same as EfficientNMS_TRT's
# num_detections case, but with no fixed-size padding to slice against:
# TensorRT reports each image's real count directly via the output allocator.
RTMO_OUTPUTS = ("dets", "keypoints")


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
    grow path may allocate. See :meth:`TrtModel.run_torch` for why building one of
    these per call is not a viable alternative.
    """

    class _TorchOutputAllocator(trt.IOutputAllocator):
        def __init__(self):
            super().__init__()
            self.buffers = {}  # name -> uint8 torch CUDA tensor
            self.shapes = {}   # name -> resolved output shape (tuple)

        def _reallocate(self, tensor_name, size):
            # Grow-only: keep the buffer we already hold whenever it is big enough,
            # so a steady-state stream of same-shaped batches stops allocating after
            # its first call. Reallocating unconditionally here leaks the GPU, badly
            # — see run_torch.
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
        # The execution context binds to whichever device is current at creation.
        # `run` always allocated its own input here so that was implicit; `run_torch`
        # accepts a caller-owned tensor, which could sit on a different GPU and would
        # otherwise be read as garbage. Recorded so it can be rejected instead.
        self._device_index = torch.cuda.current_device()

        self._logger = trt.Logger(trt.Logger.WARNING)
        # Register the standard TensorRT plugins (EfficientNMS_TRT et al.) before
        # deserializing: engines for the fused-NMS archs (retinanet, yolox,
        # fasterrcnn) embed that plugin, and deserialization fails if its creator
        # isn't registered in this process. The builder registers them too, so a
        # build+load in one process worked without this — but a fresh runtime
        # (the real trt_infer path) must register them itself.
        trt.init_libnvinfer_plugins(self._logger, "")
        _check_engine_version(self.path, trt.__version__)
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
        # Three graph output layouts, auto-detected by tensor name:
        #   * EfficientNMS_TRT (retinanet/yolox): 4 fixed-size outputs; unpack to
        #     (boxes, scores, labels) by slicing to num_detections.
        #   * rtmo (pose): (dets, keypoints) — not a detection triple; handed to
        #     the ArchHandler as that pair, one per image.
        #   * passthrough (rtdetr/rfdetr): already (boxes, scores, labels).
        self._efficientnms = set(EFFICIENTNMS_OUTPUTS).issubset(set(self._output_names))
        # Set EQUALITY, not issubset (which is right only for EfficientNMS's fixed
        # four): ``dets`` is a name other pose/DETR-family exports use too, so a
        # subset test would claim one of those as rtmo and hand keypoint code a
        # tensor that isn't keypoints — with nothing raising.
        self._rtmo = set(RTMO_OUTPUTS) == set(self._output_names)
        if self._efficientnms:
            self._emit_order = list(EFFICIENTNMS_OUTPUTS)
        elif self._rtmo:
            self._emit_order = list(RTMO_OUTPUTS)
        elif set(CANONICAL_OUTPUTS).issubset(set(self._output_names)):
            self._emit_order = list(CANONICAL_OUTPUTS)
        else:
            self._emit_order = list(self._output_names)

        # Largest batch this engine's optimization profile accepts. Submitting more
        # is not a soft failure: TensorRT rejects set_input_shape through its *logger*
        # rather than an exception, leaves the context bound to its previous shape, and
        # runs anyway — so every image past the first comes back holding a slice of the
        # first one's output. For a passthrough arch there isn't even a batch axis to
        # index, so `_split_passthrough` walks the query axis and `reshape(-1, 4)`
        # hides it. Checked up front instead.
        self._max_batch = _profile_max_batch(self.engine, self.input_name)

        # One allocator, registered once, for this context's whole life. TensorRT's
        # python bindings tie a registered allocator's lifetime to the execution
        # context, so a per-call allocator is never collected and neither are its
        # torch buffers: ~18h of continuous inference on a 117MB pair of engines
        # climbed to 9.45GiB of *live* (not cached) torch allocations and then wedged
        # permanently, every subsequent call failing on the 4-byte num_detections.
        self._allocator = _build_allocator_class(trt, torch)()
        for tname in self._output_names:
            self.context.set_output_allocator(tname, self._allocator)

    @property
    def device(self):
        """The device the execution context is bound to — where inputs must live."""
        return self._device

    @property
    def max_batch(self):
        """Largest batch this engine's optimization profile accepts, or ``None`` if
        unreadable. Public mirror of :attr:`_max_batch` — read by callers (e.g.
        ``chachak.infer.infer_in_chunks``) that want to chunk *before* handing this
        engine a same-shaped stack, rather than relying on :meth:`run_torch` to
        reject an oversized one after the fact."""
        return self._max_batch

    def to(self, device) -> "TrtModel":
        # The engine is bound to the GPU it was built on; nothing to move. Kept for
        # interface parity with OnnxModel.to().
        return self

    def run_torch(self, batched) -> list:
        """``[B,3,H,W]`` CUDA float32 -> a list of ``B`` CUDA ``[boxes, scores, labels]``.

        Always a list, never the bare-triple special case — a caller holding real
        torch tensors has no legacy shape to preserve. The returned tensors own their
        memory: the allocator's buffers are reused by the next call, so what comes
        back here is copied out of them rather than viewing them (see the copy below).
        """
        trt, torch = self._trt, self._torch

        batched = self._as_engine_input(batched)  # keep alive through execute
        batch_size = int(batched.shape[0])

        self.context.set_input_shape(self.input_name, tuple(int(d) for d in batched.shape))
        self.context.set_tensor_address(self.input_name, int(batched.data_ptr()))

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
            elem_size = torch.empty(0, dtype=dtype).element_size()
            buf = self._allocator.buffers[tname]
            # .clone() is what makes buffer reuse safe. Returning views would hand the
            # caller memory the *next* run_torch overwrites, and callers legitimately
            # hold results across calls: TrtAdapter.predict runs one batch per distinct
            # input shape and postprocesses only after every group has run, so views
            # would give the earlier groups the last group's detections. This is a
            # device-to-device copy of one batch of already-NMS'd outputs (a few KB) on
            # a synchronized stream, next to an engine execute.
            outputs[tname] = buf[: numel * elem_size].view(dtype).reshape(shape).clone()

        ordered = [outputs[tname] for tname in self._emit_order]
        if self._efficientnms:
            return _unpack_efficientnms(ordered, batch_size)
        if self._rtmo:
            return _split_rtmo(ordered, batch_size)
        return _split_passthrough(ordered, batch_size)

    def run(self, batched: np.ndarray) -> list:
        """numpy in, numpy out — the pre-``run_torch`` contract, unchanged.

        Output dtypes are whatever the engine declares (EfficientNMS labels stay
        int32, which is why callers here ``.astype(np.int64)``), and ``B == 1``
        returns the bare triple rather than a one-element list.
        """
        torch = self._torch
        array = np.ascontiguousarray(batched, dtype=np.float32)
        triples = self.run_torch(torch.from_numpy(array).to(self._device))
        out = [[tensor.detach().cpu().numpy() for tensor in triple] for triple in triples]
        # B == 1: the bare triple, exactly what every caller written before batching
        # existed still expects. B > 1: the list, so a caller that actually submitted
        # a real batch can tell images apart.
        return out[0] if len(out) == 1 else out

    def _as_engine_input(self, batched):
        """Coerce to the contiguous float32 CUDA tensor the context expects."""
        torch = self._torch
        if not torch.is_tensor(batched):
            raise TypeError(
                f"run_torch expects a torch tensor, got {type(batched).__name__}; "
                f"use run() for numpy input"
            )
        if batched.is_cuda and batched.device.index not in (None, self._device_index):
            raise ValueError(
                f"input is on cuda:{batched.device.index} but this engine's context "
                f"is on cuda:{self._device_index}; TensorRT would read the wrong "
                f"device memory"
            )
        batch = int(batched.shape[0])
        if self._max_batch is not None and batch > self._max_batch:
            raise ValueError(
                f"{self.path.name} was built with a batch profile of at most "
                f"{self._max_batch}, but {batch} images were submitted as one batch. "
                f"TensorRT would silently return the first image's detections for all "
                f"of them. Send at most {self._max_batch} at a time, or rebuild the "
                f"engine with a wider batch profile."
            )
        return batched.to(device=self._device, dtype=torch.float32).contiguous()


def _check_engine_version(engine_path: Path, runtime_version: str) -> None:
    """Fail loudly if ``engine_path`` was built by a different TensorRT than this one.

    An engine plan only deserializes on the exact TensorRT version that built it —
    a mismatch doesn't raise from ``deserialize_cuda_engine``, it just returns
    ``None`` (the ``RuntimeError`` right after this call), which reads as "this
    file is corrupt" with no hint that the real problem is a stale cached engine
    or a container image that drifted to a newer TensorRT. The builder
    (``friendy_chachkalica.ml.trt_export.cli.build_engine``) always stamps its own
    ``trt.__version__`` into ``<engine>.json`` beside the engine; check it up
    front instead of letting deserialization fail opaquely.
    """
    provenance_path = Path(str(engine_path) + ".json")
    if not provenance_path.exists():
        return  # no provenance sidecar (e.g. a hand-built engine) — nothing to check
    try:
        built_version = json.loads(provenance_path.read_text()).get("tensorrt_version")
    except (json.JSONDecodeError, OSError):
        return
    if built_version and built_version != runtime_version:
        raise RuntimeError(
            f"{engine_path} was built with TensorRT {built_version}, but this "
            f"runtime is TensorRT {runtime_version} — engine plans only load on "
            f"the exact version that built them. Rebuild the engine here, or run "
            f"it where TensorRT {built_version} is installed (see "
            f"{provenance_path.name})."
        )


def _profile_max_batch(engine, input_name):
    """Batch ceiling from the engine's first optimization profile, or None if unreadable."""
    try:
        return int(engine.get_tensor_profile_shape(input_name, 0)[-1][0])
    except (AttributeError, IndexError, TypeError, ValueError):
        return None


def _unpack_efficientnms(ordered: list, batch_size: int) -> list:
    """[num_detections, boxes, scores, classes] (fixed-size *per image*) ->
    one ``[boxes[Ni,4], scores[Ni], labels[Ni]]`` triple per image, each sliced
    to that image's own valid detection count.

    The counts are the one thing that genuinely has to reach the host — the plugin
    zero-pads the tail, and a score mask can't stand in for them because callers
    legitimately pass ``score_threshold=0.0``. That is a single ``B x 4``-byte copy
    per *batch*, on an already-synchronized stream, and the slices it drives are
    stride-preserving views rather than copies.
    """
    num_det, det_boxes, det_scores, det_classes = ordered
    counts = num_det.reshape(batch_size, -1)[:, 0].cpu().tolist()
    det_boxes = det_boxes.reshape(batch_size, -1, 4)
    det_scores = det_scores.reshape(batch_size, -1)
    det_classes = det_classes.reshape(batch_size, -1)
    return [
        [det_boxes[i, :n], det_scores[i, :n], det_classes[i, :n]]
        for i, n in enumerate(int(count) for count in counts)
    ]


def _split_passthrough(ordered: list, batch_size: int) -> list:
    """Passthrough outputs (ecdet/rtdetr/rfdetr/dfine) -> one ``[boxes, scores, labels]``
    triple per image.

    **Batch axis detected by rank, not assumed.** These archs' export wrappers used
    to index the batch axis away before baking the head math (a single ``[0]`` on
    the model output, then a flattened top-k), which emitted
    ``boxes[N,4] / scores[N] / labels[N]`` with N the detection count, not the batch
    size — and their engines were built with a batch profile of exactly 1 to match.
    The wrappers now carry a real batch dim through (see each
    ``onnx_export/arch/*.py`` module docstring and ``trt_export.arch.
    BATCH_AWARE_ARCHS``), emitting ``boxes[B,K,4] / scores[B,K] / labels[B,K]``
    instead — but an engine built from the OLD-style export (or from a fresh export
    with the default batch-1 profile) still only ever has rank-2 outputs, so both
    shapes have to keep working: rank-2 ``boxes`` means no batch axis (the whole
    output is one image's detections), rank-3 means a real batch axis to index.

    Getting this wrong at ``batch_size == 1`` on a rank-2 output used to read
    ``boxes[i], scores[i], labels[i]``, which returned detection *zero* —
    ``(4,) / () / ()`` — and silently discarded every other detection. ``to_friendy``
    then reshaped that lone box into a valid ``(1, 6)``, so nothing raised: an
    RF-DETR engine emitting 300 detections per crop delivered exactly one, and a
    person could only ever carry a single PPE label. The ONNX path was never
    affected — ``OnnxModel.run`` returns the graph's outputs untouched, which is
    what defines the handler contract these have to meet.
    """
    boxes, scores, labels = ordered
    if boxes.ndim <= 2:
        # No batch axis: the whole output *is* this one image's detections.
        if batch_size != 1:
            raise RuntimeError(
                f"passthrough engine returned un-batched outputs (boxes shape "
                f"{tuple(boxes.shape)}) for a batch of {batch_size}. Its graph collapses "
                f"the batch axis, so it must be run one image at a time — build the "
                f"engine with a batch-1 profile, or lower the inference batch size."
            )
        return [[boxes, scores, labels]]
    return [[boxes[i], scores[i], labels[i]] for i in range(batch_size)]


def _split_rtmo(ordered: list, batch_size: int) -> list:
    """``[dets[B,N,5], keypoints[B,N,17,3]]`` -> one ``[dets_i, keypoints_i]``
    pair per image — **not** a Contract-A triple; ``RTMOHandler`` makes one out
    of the pair (see ``onnx_infer/arch/rtmo.py``).

    Always batch-first, so unlike :func:`_split_passthrough` this needs no
    rank-based fallback: the mmdeploy export never indexes the batch axis away
    the way this repo's own export wrappers used to.
    """
    dets, keypoints = ordered
    return [[dets[i], keypoints[i]] for i in range(batch_size)]
