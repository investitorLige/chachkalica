"""``AsyncEngine`` — ``trt_infer.session.TrtModel`` without the per-call synchronization.

This is the mechanism the whole package is built around. ``TrtModel.run_torch`` ends every
execution with ``stream.synchronize()`` (``trt_infer/session.py:236``), which drains the CUDA
queue and blocks the host. With one engine call per person crop, that is a stall per crop, and
frame *k+1* cannot start while frame *k* is still finishing.

What is actually safe to remove, and why
----------------------------------------

``run_torch`` needs its synchronization for two reasons, and this class removes both *causes*
rather than ignoring the effect:

1. **It reads output shapes and copies buffers on the host side of the call.** The copy is the
   part that matters, and ``.clone()`` on a CUDA tensor is itself a *stream-ordered*
   device-to-device copy. Enqueued on the same stream as the execute, it cannot begin until the
   execute finishes, and the *next* execute cannot begin until it finishes. So
   ``execute_i -> copy_i -> execute_{i+1}`` is correctly ordered by the stream alone. The
   synchronization was never what made the copy safe; the copy's stream placement is.

2. **``_unpack_efficientnms`` reads ``num_detections`` to the host** (``session.py:352``) so it
   can slice ``det_boxes[i, :n]``. That read is a genuine synchronization. This class does not
   slice — it returns the fixed-size output plus the count *tensor*, and
   :func:`gpu_infer.tail.mask_from_num_detections` turns the count into a device-side mask. Same
   detections, no host read.

Output *shapes* still have to be known host-side, but they already are: TensorRT resolves them
during enqueue and reports them through the allocator's ``notify_shape`` callback, which runs
synchronously as part of ``execute_async_v3``. Nothing is read back from the device to learn
them. Where an output is fully static TensorRT may not notify at all, and the context's own
``get_tensor_shape`` answers — also host-side metadata. Both paths are exercised by
``tests/test_engine_gpu.py``, which is also where the ordering claim above is proven byte-for-byte
against ``run_torch``.

What is deliberately *not* changed
----------------------------------

Engine deserialization, plugin registration, the version check, the output allocator and its
grow-only buffer policy, and the batch-profile ceiling all come from ``TrtModel`` by inheritance.
In particular the ceiling is load-bearing: submitting more images than the profile allows makes
TensorRT report a rejected ``set_input_shape`` through its *logger* and then run at the previous
shape, so every image past the first comes back holding a slice of the first one's output.
``TrtModel._as_engine_input`` raises instead, and this class routes every submission through it.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Optional, Tuple

#: TensorRT requires tensor addresses to be aligned. A pointer into row ``i`` of a stacked
#: batch is only usable directly when the per-image byte stride is a multiple of this.
_ADDRESS_ALIGNMENT = 256

#: One non-default stream per device index, created on first use. See :func:`inference_stream`.
_STREAMS: dict = {}

#: Set once, so the default-stream warning is printed a single time rather than per submission.
_DEFAULT_STREAM_WARNED: set = set()


def inference_stream(device=None):
    """The package's dedicated non-default CUDA stream for ``device``.

    **Why not the default stream.** TensorRT warns on every ``enqueueV3`` submitted to the
    default stream: *"Using default stream in enqueueV3() may lead to performance issues due to
    additional calls to cudaStreamSynchronize() by TensorRT to ensure correct synchronization."*
    It inserts those synchronizations itself, which would give back exactly the overlap this
    package removes ``run_torch``'s ``stream.synchronize()`` to obtain — a silent partial
    defeat, visible only as a speedup that fails to materialize.

    **Why one stream for everything.** Preprocessing, every engine, and the whole masked tail
    run on this single stream, so all ordering is implicit: each operation sees the previous
    one's results because they are sequential on the same stream. Splitting engines across
    streams would need explicit events and ``record_stream`` calls to stay correct, and would buy
    nothing — the work is already a dependency chain, not independent parallel work.

    Cached per device index rather than per call, because a stream is a real resource and the
    tensors produced on it must all belong to the same one.

    Do not enter this stream directly with ``torch.cuda.stream(...)`` — use
    :func:`inference_scope`, which adds the cross-stream ordering that makes it safe.
    """
    import torch

    index = torch.cuda.current_device() if device is None else torch.device(device).index
    index = torch.cuda.current_device() if index is None else index
    if index not in _STREAMS:
        _STREAMS[index] = torch.cuda.Stream(device=index)
    return _STREAMS[index]


@contextlib.contextmanager
def inference_scope(device=None, *, inputs=()):
    """Enter :func:`inference_stream` with the cross-stream ordering that makes it correct.

    Entering a second stream is not free of consequences, and getting this wrong produces a
    ``CUDA error 700 (illegal address)`` that surfaces somewhere else entirely — which is exactly
    how it was found. Three hazards, all handled here:

    1. **Inputs produced on the caller's stream.** A frame decoded on the default stream is not
       necessarily *finished* being written when this stream starts reading it. ``wait_stream``
       makes this stream wait for everything already queued on the caller's.
    2. **Inputs freed too early.** torch's caching allocator is stream-aware: it may hand a
       tensor's memory to a new allocation on the *caller's* stream while this stream is still
       reading it. ``record_stream`` tells the allocator to hold that memory until this stream is
       done with it.
    3. **The caller's later work.** Anything queued on the caller's stream afterwards must not
       start before this stream's work finishes, so on exit the caller's stream waits on ours.

    An engine's output-allocator buffers are long-lived shared state, so *all* submissions to a
    given engine must be ordered with respect to each other. Routing every submission through
    this scope is what provides that; two unsynchronized streams submitting to one engine race on
    those buffers.
    """
    import torch

    stream = inference_stream(device)
    prior = torch.cuda.current_stream(device=stream.device)
    if stream.cuda_stream == prior.cuda_stream:  # already here (nested); nothing to order
        yield stream
        return

    stream.wait_stream(prior)
    for tensor in inputs:
        if tensor is not None and tensor.is_cuda:
            tensor.record_stream(stream)
    try:
        with torch.cuda.stream(stream):
            yield stream
    finally:
        prior.wait_stream(stream)


@dataclass
class EngineOutputs:
    """One submission's raw detections, padded and still on the device.

    Uniform across both graph layouts so the tail does not have to care which it got:

    * ``boxes`` ``[B, K, 4]``, ``scores`` ``[B, K]``, ``labels`` ``[B, K]``
    * ``num_detections`` ``[B]`` for EfficientNMS graphs, or ``None`` for passthrough graphs
      where every one of the ``K`` query slots is a real (if low-scoring) detection.

    ``K`` is the graph's static detection axis — ``max_output_boxes`` for EfficientNMS,
    ``num_queries`` for passthrough — so it is a compile-time constant, which is what lets the
    tail preallocate.
    """

    boxes: "object"
    scores: "object"
    labels: "object"
    num_detections: Optional["object"]

    @property
    def batch(self) -> int:
        return int(self.boxes.shape[0])

    @property
    def max_det(self) -> int:
        return int(self.boxes.shape[1])

    def valid_mask(self):
        """``[B, K]`` bool: which slots hold real detections.

        For EfficientNMS this is the ``arange`` comparison that replaces the host count read.
        For passthrough it is all-true — the graph emits a fixed query set with no notion of a
        count, exactly as ``_split_passthrough`` assumes.
        """
        import torch

        if self.num_detections is None:
            return torch.ones(
                (self.batch, self.max_det), dtype=torch.bool, device=self.boxes.device
            )
        from . import tail

        return tail.mask_from_num_detections(self.num_detections, self.max_det)


class AsyncEngine:
    """A loaded engine that submits work without blocking the host.

    Composes ``TrtModel`` rather than subclassing it: the synchronous ``run_torch`` / ``run``
    remain available and unmodified through :attr:`model`, which is what
    ``tests/test_engine_gpu.py`` differentials against, and what
    ``GpuOptions.async_engine=False`` falls back to when a suspected ordering bug needs
    bisecting.
    """

    def __init__(self, engine_path, meta, device="cuda", *, warmup: bool = True) -> None:
        from trt_infer.session import TrtModel

        self.meta = meta
        self.model = TrtModel(engine_path, meta, device=device)
        self._efficientnms = bool(getattr(self.model, "_efficientnms", False))
        self._scratch = None
        if warmup:
            self._warmup()

    # -- introspection the caller needs before it can size anything ----------------

    @property
    def device(self):
        return self.model.device

    @property
    def max_batch(self) -> Optional[int]:
        """Largest batch this engine's profile accepts, or ``None`` if unreadable."""
        return self.model.max_batch

    @property
    def efficientnms(self) -> bool:
        return self._efficientnms

    def _warmup(self) -> None:
        """Run one throwaway batch so the allocator's buffers already exist.

        The allocator is grow-only (``session.py:78-90``), so after this the steady-state path
        allocates nothing. Doing it through the *synchronous* ``run_torch`` is deliberate: it
        forces any engine or plugin problem to surface here, at load, with a clean stack, rather
        than inside the first un-synchronized submission where the failure would surface later
        and somewhere else.
        """
        import torch

        shape = self._warmup_shape()
        if shape is None:
            return
        self.model.run_torch(torch.zeros(shape, dtype=torch.float32, device=self.device))

    def _warmup_shape(self) -> Optional[Tuple[int, ...]]:
        """The engine's own optimum profile shape, or ``None`` if it cannot be read."""
        try:
            profile = self.model.engine.get_tensor_profile_shape(
                self.model.input_name, 0
            )
        except (AttributeError, IndexError, TypeError, ValueError):
            return None
        opt = tuple(int(dim) for dim in profile[1])
        return opt if len(opt) == 4 and all(dim > 0 for dim in opt) else None

    # -- submission ---------------------------------------------------------------

    def submit(self, batched) -> EngineOutputs:
        """Enqueue one ``[B, 3, H, W]`` batch and return its padded outputs. **No sync.**

        The returned tensors own their memory: they are stream-ordered copies out of the
        allocator's reusable buffers, so holding them across further submissions is safe — which
        callers do, since a multi-shape submission runs every group before postprocessing any of
        them.
        """
        import torch

        batched = self.model._as_engine_input(batched)  # keep alive through the execute
        context = self.model.context
        context.set_input_shape(
            self.model.input_name, tuple(int(dim) for dim in batched.shape)
        )
        context.set_tensor_address(self.model.input_name, int(batched.data_ptr()))

        # Only the per-call shapes are stale; the allocator itself lives as long as the context.
        # Clearing means an output TensorRT does not notify this time falls back to the
        # context's own shape rather than silently reusing the previous call's.
        self.model._allocator.shapes.clear()

        # Whatever stream is current -- the caller is expected to have entered
        # inference_stream()'s non-default one, and _warn_if_default_stream says so if not.
        stream = torch.cuda.current_stream()
        self._warn_if_default_stream(stream)
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 returned False")
        # Deliberately no stream.synchronize() -- see the module docstring.

        raw = self._collect(context)
        return self._shape_outputs(raw, int(batched.shape[0]))

    def submit_single(self, stacked, index: int) -> EngineOutputs:
        """Enqueue row ``index`` of a stacked batch as a batch of one, without copying it.

        This is the batch-1 passthrough path — rtdetr/rfdetr engines are built with a batch-1
        profile because their export wrappers index the batch axis away, so N crops must be N
        executions. Pointing TensorRT at row ``index`` of an existing contiguous buffer avoids
        re-materializing each crop.

        TensorRT requires an aligned address, and the per-image byte stride is not guaranteed to
        be a multiple of :data:`_ADDRESS_ALIGNMENT` for an arbitrary input size. When it is not,
        the row is copied into a persistent scratch buffer first — a stream-ordered
        device-to-device copy, so still no synchronization.
        """
        row_bytes = stacked[0].numel() * stacked.element_size()
        if row_bytes % _ADDRESS_ALIGNMENT == 0 and stacked.is_contiguous():
            return self.submit(stacked[index : index + 1])
        if self._scratch is None or self._scratch.shape != stacked[0:1].shape:
            import torch

            self._scratch = torch.empty_like(stacked[0:1])
        self._scratch.copy_(stacked[index : index + 1])
        return self.submit(self._scratch)

    @staticmethod
    def _warn_if_default_stream(stream) -> None:
        """Say so, once, if work is being submitted to the default stream.

        Submitting there is *correct* but slow: TensorRT adds its own synchronizations, undoing
        the overlap this class exists to create. Warning rather than raising, because a caller
        doing a one-off submission outside the pipeline is a legitimate use and does not care.
        """
        import torch

        if stream.cuda_stream == torch.cuda.default_stream(stream.device).cuda_stream:
            if not _DEFAULT_STREAM_WARNED:
                _DEFAULT_STREAM_WARNED.add(True)
                print(
                    "[gpu_infer] submitting on the default CUDA stream; TensorRT will insert "
                    "its own synchronizations and the async win will not materialize. Wrap the "
                    "call in `with gpu_infer.engine.inference_scope(inputs=(frame,)):` — the "
                    "scope, not inference_stream() directly, because it also supplies the "
                    "cross-stream ordering."
                )

    # -- output plumbing ----------------------------------------------------------

    def _collect(self, context) -> dict:
        """Stream-ordered copies of every output buffer, keyed by tensor name.

        Mirrors ``run_torch``'s reshape-and-clone block. ``.clone()`` is the copy that makes
        buffer reuse safe *and* is itself ordered on the stream, which is the whole reason the
        synchronization is unnecessary.
        """
        import numpy as np
        import torch

        from trt_infer.session import _torch_dtype_for

        outputs = {}
        for name in self.model._output_names:
            shape = self.model._allocator.shapes.get(name)
            if shape is None:  # static output: the context knows it, host-side
                shape = tuple(int(dim) for dim in context.get_tensor_shape(name))
            dtype = _torch_dtype_for(
                self.model._trt, torch, self.model.engine.get_tensor_dtype(name)
            )
            count = int(np.prod(shape)) if len(shape) else 1
            element_size = _ELEMENT_SIZES[dtype]
            buffer = self.model._allocator.buffers[name]
            outputs[name] = (
                buffer[: count * element_size].view(dtype).reshape(shape).clone()
            )
        return outputs

    def _shape_outputs(self, raw: dict, batch: int) -> EngineOutputs:
        """Normalize either graph layout into :class:`EngineOutputs`."""
        if self._efficientnms:
            return EngineOutputs(
                boxes=raw["detection_boxes"].reshape(batch, -1, 4),
                scores=raw["detection_scores"].reshape(batch, -1),
                labels=raw["detection_classes"].reshape(batch, -1),
                num_detections=raw["num_detections"].reshape(batch, -1)[:, 0],
            )
        boxes, scores, labels = (
            raw[name] for name in ("boxes", "scores", "labels")
        )
        if boxes.ndim <= 2:
            # No batch axis: both DETR export wrappers index it away before baking the head
            # math, so the graph emits [N_queries, 4] and its engine carries a batch-1 profile.
            # Reading boxes[i] here instead would return detection *zero* and silently discard
            # every other one -- the bug _split_passthrough documents at session.py:377.
            if batch != 1:
                raise RuntimeError(
                    f"passthrough engine returned un-batched outputs (boxes shape "
                    f"{tuple(boxes.shape)}) for a batch of {batch}. Its graph collapses the "
                    f"batch axis, so it must be submitted one image at a time."
                )
            return EngineOutputs(
                boxes=boxes.reshape(1, -1, 4),
                scores=scores.reshape(1, -1),
                labels=labels.reshape(1, -1),
                num_detections=None,
            )
        return EngineOutputs(
            boxes=boxes.reshape(batch, -1, 4),
            scores=scores.reshape(batch, -1),
            labels=labels.reshape(batch, -1),
            num_detections=None,
        )


def _element_sizes() -> dict:
    """Bytes per element, per torch dtype.

    A table rather than ``run_torch``'s ``torch.empty(0, dtype=dtype).element_size()``, which
    allocates a throwaway tensor per output per call just to read a constant.
    """
    import torch

    return {
        torch.float32: 4, torch.float16: 2, torch.bfloat16: 2,
        torch.int64: 8, torch.int32: 4, torch.int16: 2, torch.int8: 1,
        torch.uint8: 1, torch.bool: 1, torch.float64: 8,
    }


class _LazyElementSizes(dict):
    """``_ELEMENT_SIZES`` without importing torch at module scope.

    The table needs torch's dtype objects as keys, and this package's rule is that importing it
    must not import torch. Populating on first lookup keeps both true.
    """

    def __missing__(self, key):
        self.update(_element_sizes())
        try:
            return dict.__getitem__(self, key)
        except KeyError:
            raise RuntimeError(f"unsupported TensorRT output dtype: {key}") from None


_ELEMENT_SIZES = _LazyElementSizes()
