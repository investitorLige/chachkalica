"""The loaded models, and one chachak pipeline per camera.

Lives inside the GPU worker process and owns everything expensive: the TensorRT
engines, the optional person-detector engines, and the pipeline objects built
around them. Nothing here touches Redis or the database — the worker does that —
so this module is testable with a fake adapter and no GPU.

**One set of engines per distinct bundle in use, shared by every camera on it.**
Cameras pick their own model now, so this holds a pool keyed by bundle rather
than the single adapter it used to. What it deliberately does *not* do is hold
one per camera: five cameras on the same bundle share one copy of its engines, so
the cost is the number of *different* models an operator has asked for, not the
number of cameras. Only bundles some enabled camera actually resolves to are
loaded, and one nothing points at anymore is dropped on the next reconcile — that
combination is what keeps a per-camera choice affordable on a 16 GB card, and it
is why :meth:`ModelRuntime.sync_bundles` takes the whole desired set at once
rather than being told about one bundle at a time.

Per-camera pipelines remain cheap objects wrapping whichever bundle's adapter, so
a camera switching from raw to tiled still costs a rebuild of a few dataclasses
rather than a model reload.

A bundle that fails to load is remembered as a failure rather than raised past
the pool, so it takes down only the cameras pointing at it. With one global model
that distinction did not exist; with several it is the difference between one
misconfigured camera and a dead system.

Reloads are driven by fingerprints rather than by explicit invalidation. The
worker re-reads the config every reconcile and compares; anything that differs
gets rebuilt. That is why editing a model, re-pointing a camera at another
bundle, or retuning a camera's tiling in the admin takes effect without
restarting the process.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from vendor.chachak import Detector, build_pipeline, load_checkpoint_adapter
from vendor.chachak.detector import load_detector

logger = logging.getLogger(__name__)


class EngineNotReady(RuntimeError):
    """The active configuration cannot be run — reported against the camera or model."""


class LoadedBundle:
    """One bundle's engines, or the reason they could not be loaded.

    A failure is a *value* here rather than an exception: the pool loads several
    bundles in one pass, and one bad artifact must not stop the others from being
    loaded or keep being retried at full speed. The error is re-raised per camera
    by :meth:`ModelRuntime.sync_camera`, which is what puts it on the camera's
    admin page instead of only in a log.
    """

    __slots__ = ("name", "fingerprint", "adapter", "detector", "error")

    def __init__(self, name, fingerprint, adapter=None, detector=None, error=None):
        self.name = name
        self.fingerprint = fingerprint
        self.adapter = adapter
        self.detector: Optional[Detector] = detector
        self.error = error

    @property
    def ok(self) -> bool:
        return self.adapter is not None


class ModelRuntime:
    """Holds one set of engines per bundle in use, and each camera's pipeline."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        # bundle_pk -> LoadedBundle (which may be a recorded failure)
        self._bundles: dict[int, LoadedBundle] = {}
        # camera_id -> (fingerprint, pipeline_or_None, InferenceConfig, bundle_pk)
        self._pipelines: dict[int, tuple] = {}

    # ──────────────────────────────────────────────────── model loading ──

    @property
    def ready(self) -> bool:
        """Whether anything at all is loaded and usable."""
        return any(entry.ok for entry in self._bundles.values())

    @property
    def loaded_bundle_pks(self) -> set[int]:
        return {pk for pk, entry in self._bundles.items() if entry.ok}

    def sync_bundles(self, bundles: dict) -> bool:
        """Make the pool hold exactly ``bundles``. Returns whether anything changed.

        ``bundles`` is ``{pk: Bundle}`` — every distinct bundle some enabled
        camera resolves to. Passed whole rather than incrementally because
        "what should be resident" is a property of the camera table as a set:
        anything not in it is released here, which is the only thing that ever
        gives VRAM back.

        A bundle whose ``fingerprint()`` moved (re-registered after a re-export,
        thresholds retuned) is released and loaded again. Loading is best-effort
        per bundle: a failure is recorded on the entry and surfaces later against
        the cameras that use it.
        """
        changed = False
        released = False

        for pk in list(self._bundles):
            if pk not in bundles:
                logger.info("releasing bundle %s — no camera uses it", self._bundles[pk].name)
                self._release_bundle(pk)
                changed = released = True

        for pk, bundle in bundles.items():
            fingerprint = bundle.fingerprint()
            current = self._bundles.get(pk)
            if current is not None and current.fingerprint == fingerprint:
                continue
            if current is not None:
                # Freed before the replacement allocates: on a card that is
                # already holding several models, loading the new copy alongside
                # the old one is exactly the case that runs out of memory.
                self._release_bundle(pk)
                released = True
            self._bundles[pk] = self._load_bundle(bundle, fingerprint)
            changed = True

        if released:
            self._empty_cuda_cache()
        return changed

    def _load_bundle(self, bundle, fingerprint) -> LoadedBundle:
        try:
            path = bundle.model_path()
            logger.info("loading bundle %s (%s) from %s", bundle.name, bundle.arch, path)
            started = time.monotonic()
            adapter, info = load_checkpoint_adapter(path, self.device)
            logger.info(
                "bundle %s ready in %.1fs (arch=%s, classes=%s)",
                bundle.name, time.monotonic() - started,
                info.get("model_name"), info.get("num_classes"),
            )

            detector = None
            if bundle.has_detector:
                detector_path = bundle.detector_path()
                detector_config = (bundle.pipeline_config or {}).get("detector") or {}
                logger.info("loading detector for bundle %s from %s", bundle.name, detector_path)
                detector = load_detector(
                    detector_path, self.device,
                    person_class_name=detector_config.get("person_class_name"),
                    person_class_id=detector_config.get("person_class_id"),
                    score_threshold=detector_config.get("score_threshold", 0.5),
                    batch_size=detector_config.get("batch_size", 4),
                )
        except Exception as exc:  # noqa: BLE001 - recorded, not raised; see the class docstring
            logger.exception("bundle %s failed to load", bundle.name)
            return LoadedBundle(bundle.name, fingerprint, error=str(exc))

        return LoadedBundle(bundle.name, fingerprint, adapter=adapter, detector=detector)

    def _release_bundle(self, bundle_pk) -> None:
        """Drop one bundle's engines and every pipeline built against them.

        The pipelines have to go with it: they hold the adapter directly, and one
        left wrapping a freed engine fails in a way that reads like a model bug.
        """
        self._bundles.pop(bundle_pk, None)
        for camera_id, entry in list(self._pipelines.items()):
            if entry[3] == bundle_pk:
                del self._pipelines[camera_id]

    def release(self) -> None:
        """Drop every adapter, pipeline, and their VRAM."""
        self._pipelines.clear()
        self._bundles.clear()
        self._empty_cuda_cache()

    @staticmethod
    def _empty_cuda_cache() -> None:
        """Hand freed blocks back before the next allocation.

        Matters on a 16 GB card: torch keeps freed blocks in its allocator, so
        without this the *old* engine's memory is still reserved while the new one
        allocates, and swapping models could fail for want of memory that is
        notionally free. More so now that several models can be resident at once.
        """
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - torch may be absent (tests, CPU-only)
            pass

    # ───────────────────────────────────────────────── per-camera setup ──

    def sync_camera(self, camera, active) -> Any:
        """Return this camera's ``InferenceConfig``, rebuilding its pipeline if needed.

        Raises :class:`EngineNotReady` when the camera cannot run — no bundle
        resolved, its bundle failed to load or isn't in the pool, an invalid
        config, or a person-detector pipeline on a bundle with no detector engine.
        Each message names the bundle, because with several loaded "the model has
        no detector" no longer identifies which one. The worker records it against
        the camera so it shows up in the admin rather than only in a log.
        """
        bundle = camera.resolved_bundle(active)
        if bundle is None:
            raise EngineNotReady(
                "No bundle: this camera has none of its own and no default is set "
                "on the Active model page."
            )

        entry = self._bundles.get(bundle.pk)
        if entry is None:
            raise EngineNotReady(f"Bundle '{bundle.name}' is not loaded.")
        if not entry.ok:
            raise EngineNotReady(f"Bundle '{bundle.name}' failed to load: {entry.error}")

        config = camera.build_inference_config(active)
        try:
            config.validate()
        except ValueError as exc:
            raise EngineNotReady(str(exc)) from exc

        if config.needs_detector() and entry.detector is None:
            raise EngineNotReady(
                f"Pipeline '{config.pipeline}' needs a person detector, but bundle "
                f"'{bundle.name}' has none."
            )

        fingerprint = camera.inference_fingerprint(active)
        cached = self._pipelines.get(camera.pk)
        if cached and cached[0] == fingerprint:
            return cached[2]

        if config.is_raw():
            # No chachak pipeline at all — `run` calls the adapter directly.
            pipeline = None
        else:
            # A dict shared across the pipelines built from one config, so a
            # chained pipeline runs the frozen detector once per frame rather than
            # once per member. Per-camera and per-build, so it can't outlive the
            # frames it describes.
            pipeline = build_pipeline(
                config, entry.adapter, self.device,
                detector=entry.detector, box_cache={},
            )
        self._pipelines[camera.pk] = (fingerprint, pipeline, config, bundle.pk)
        logger.info(
            "camera %s: pipeline '%s' ready on bundle %s",
            camera.pk, config.pipeline, bundle.name,
        )
        return config

    def forget_camera(self, camera_id) -> None:
        self._pipelines.pop(camera_id, None)

    def config_for(self, camera_id) -> Optional[Any]:
        """The ``InferenceConfig`` cached by :meth:`sync_camera`, or None.

        Read per frame by the worker, so it reuses the resolved config rather than
        re-doing the three-layer merge (and its bundle lookup) thousands of times
        an hour for a value that only changes on reconcile.
        """
        entry = self._pipelines.get(camera_id)
        return entry[2] if entry else None

    def bundle_pk_for(self, camera_id) -> Optional[int]:
        """Which bundle this camera's prepared pipeline was built against.

        The worker labels boxes and stamps the ``Detection`` row from it. Read
        from the cache rather than resolved again so the bundle credited with a
        frame is the one that actually produced it, even if somebody re-pointed
        the camera between this frame and the next reconcile.
        """
        entry = self._pipelines.get(camera_id)
        return entry[3] if entry else None

    # ─────────────────────────────────────────────────────── inference ──

    def run(self, camera_id, frame_bgr, context=None):
        """Run one BGR frame through this camera's pipeline. Returns Friendy ``(N,6)``.

        Requires :meth:`sync_camera` to have been called for this camera.

        ``context``, when a dict is passed, is filled with what the pipeline knew
        about this frame beyond the detections themselves — currently
        ``"person_boxes"``, the people a person-first pipeline cropped around, in
        the same normalized ``{"cx","cy","w","h"}`` form a detection uses. Asking
        for it widens each returned row to seven columns, the last being the index
        into that list of the person the detection was found on. See
        ``vendor/chachak/pipeline.py::Pipeline.process_batch``.

        Stamped in place rather than returned alongside, matching how
        ``tracking.Tracker.update`` and ``zones.evaluate`` hand back their extras:
        the caller that wants only boxes keeps the call it already had.
        """
        entry = self._pipelines.get(camera_id)
        if entry is None:
            raise EngineNotReady(f"camera {camera_id} has no pipeline prepared")
        _fingerprint, pipeline, config, bundle_pk = entry

        chw = _bgr_to_chw(frame_bgr)
        if pipeline is None:
            # Raw: the whole frame straight to this camera's own model.
            loaded = self._bundles.get(bundle_pk)
            if loaded is None or not loaded.ok:
                raise EngineNotReady(f"camera {camera_id}'s bundle is no longer loaded")
            if context is not None:
                # No detector, no crops, so nobody to anchor anything to.
                context["person_boxes"] = []
            preds = loaded.adapter.predict([chw], score_threshold=config.score_threshold)
            return preds[0]
        # `targets` carries per-frame identity for the detector's box cache. A live
        # frame has no stable identity — it is seen once — so an empty dict is
        # correct here, and the cache degrades to "compute every time" as designed.
        frame_context = [{}] if context is not None else None
        preds = pipeline.process_batch([chw], [{}], context=frame_context)[0]
        if context is not None:
            context.update(frame_context[0])
        return preds


def _bgr_to_chw(frame_bgr):
    """OpenCV HWC BGR uint8 -> the CHW float tensor the adapters expect.

    Channel order is left as BGR: the model's ``meta.json`` declares its own
    ``layout``, and ``vendor/onnx_infer/preprocess`` performs the swap when the
    model wants RGB. Converting here as well would swap twice and silently feed
    every model wrong-ordered channels — a failure that shows up as mysteriously
    poor accuracy rather than an error.

    Scaling to [0, 1] matches what the adapters take; a model whose meta says
    ``input_scale: "byte"`` gets multiplied back up by ``preprocess``.
    """
    import numpy as np
    import torch

    array = np.ascontiguousarray(frame_bgr.transpose(2, 0, 1), dtype=np.float32) / 255.0
    return torch.from_numpy(array)
