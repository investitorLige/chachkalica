"""Per-stage timing for an exported bundle's *own* vendored runtime.

A bundle carries a frozen copy of ``chachak``/``onnx_infer``/``trt_infer``, taken at
export time. So this module cannot assume any instrumentation exists inside it:
stages are obtained by **wrapping the bundle's runtime at benchmark time**, and every
stage records *how* it was obtained (``direct`` / ``residual`` / ``harness`` /
``unavailable``). A bundle whose vendored code predates a probe reports that stage
``unavailable`` — never a silent zero.

Attribution is a tree, so the arithmetic is checkable (siblings + residual = parent):

    run_batch                              (harness-timed total)
      ├─ pipeline            process_batch
      │    ├─ person         the person detector, inclusive
      │    │    ├─ person.preprocess / person.forward / person.postprocess
      │    │    └─ person.other          (residual)
      │    ├─ model          the trained model adapter, inclusive
      │    │    ├─ model.preprocess / model.forward / model.postprocess
      │    │    └─ model.other           (residual)
      │    └─ assemble       expand / crop / pad / tile  (residual)
      ├─ merge_nms
      ├─ remap
      └─ unattributed                     (residual)

The same adapter *classes* serve both the person detector and the trained model, so a
probe is **role-aware**: the recorder keeps a call stack and files a hit under
``person.*`` when a ``person`` frame is open and ``model.*`` otherwise. That is what
keeps "stage 2 = person" and "stage 4 = other engine" separable at all.

**Forward compatibility.** A bundle may declare its own stages instead of being
probed, by shipping ``runtime/benchmark_stages.py`` with a ``BENCHMARK_STAGES`` list
(or a ``benchmark_stages`` key in ``pipeline.json``); see :func:`load_stage_plan`.
That declaration wins, which is how a new bundle version adds or renames stages
without a change here and without a change to the report template — the report
renders whatever stages the result JSON contains, in the JSON's own order.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

# ── the taxonomy ────────────────────────────────────────────────────────────────
# id, label, parent, kind. Order here is the report's order for probed bundles; a
# bundle that declares its own plan supplies its own order instead.
STAGE_TAXONOMY: Tuple[Tuple[str, str, Optional[str], str], ...] = (
    ("decode",            "Image decode",            None,     "harness"),
    ("pipeline",          "Pipeline",                 None,     "direct"),
    ("person",            "Person detector",          "pipeline", "direct"),
    ("person.preprocess", "· preprocess",             "person", "direct"),
    ("person.forward",    "· engine forward",         "person", "direct"),
    ("person.postprocess", "· postprocess",           "person", "direct"),
    ("person.other",      "· other",                  "person", "residual"),
    ("assemble",          "In-between (crop/pad/tile)", "pipeline", "residual"),
    ("model",             "Model engine",             "pipeline", "direct"),
    ("model.preprocess",  "· preprocess",             "model",  "direct"),
    ("model.forward",     "· engine forward",         "model",  "direct"),
    ("model.postprocess", "· postprocess",            "model",  "direct"),
    ("model.other",       "· other",                  "model",  "residual"),
    ("merge_nms",         "Merge + class-aware NMS",  None,     "direct"),
    ("remap",             "Class remap",              None,     "direct"),
    ("unattributed",      "Unattributed",             None,     "residual"),
)

STAGE_LABELS: Dict[str, str] = {sid: label for sid, label, _p, _k in STAGE_TAXONOMY}
STAGE_PARENTS: Dict[str, Optional[str]] = {sid: p for sid, _l, p, _k in STAGE_TAXONOMY}
STAGE_KINDS: Dict[str, str] = {sid: k for sid, _l, _p, k in STAGE_TAXONOMY}


@dataclass(frozen=True)
class Probe:
    """One wrap site inside the bundle's runtime.

    ``target`` is ``"module.path:Attr"`` or ``"module.path:Class.method"``. Several
    probes may share a ``stage`` — they are alternative bindings for the two runtime
    flavours (onnx vs engine), and only one flavour is ever live in a given bundle,
    so the other simply contributes nothing.

    ``role_aware`` resolves ``stage`` to ``person.<stage>`` when a ``person`` frame is
    open, ``model.<stage>`` otherwise.
    """

    stage: str
    target: str
    role_aware: bool = False


# The module name :func:`inferlica.benchmark.bundle_bench._import_bundle_infer` gives
# the bundle's own ``infer.py``. Named here because two probes target it.
BUNDLE_INFER_MODULE = "_benchmarked_bundle_infer"

# Binding sites, not definition sites. ``onnx_infer/adapter.py`` does
# ``from .preprocess import preprocess`` at module scope, so patching
# ``onnx_infer.preprocess.preprocess`` would never be seen — the adapter holds its own
# reference. ``trt_infer``'s adapter imports inside ``predict``, so its module-level
# name is the right target there. Getting this wrong is silent: the stage reads 0.
PROBES: Tuple[Probe, ...] = (
    Probe("pipeline", "chachak.pipeline:Pipeline.process_batch"),
    Probe("pipeline", "chachak.pipeline:BatchDetectPipeline.process_batch"),
    Probe("pipeline", "chachak.pipeline:PeopleDetectFirstPipeline.process_batch"),
    Probe("pipeline", "chachak.pipeline:BatchPeoplePipeline.process_batch"),
    Probe("pipeline", "chachak.pipeline:ChainedPipeline.process_batch"),
    # The detector is the one unambiguous person boundary: the ``_person_boxes``
    # overrides call each other through super(), which would double-count.
    Probe("person", "chachak.detector:Detector.predict"),
    Probe("model", "onnx_infer.adapter:OnnxAdapter.predict", role_aware=True),
    Probe("model", "trt_infer.adapter:TrtAdapter.predict", role_aware=True),
    Probe("preprocess", "onnx_infer.adapter:preprocess", role_aware=True),
    Probe("preprocess", "trt_infer.preprocess_torch:preprocess_torch", role_aware=True),
    # A bundle vendored before the GPU-resident path existed has no
    # preprocess_torch/postprocess_torch at all: its TrtAdapter binds onnx_infer's
    # numpy pair at module scope instead. Real bundles on disk here are that old, and
    # without these two the whole pre/post cost lands in "model.other".
    Probe("preprocess", "trt_infer.adapter:preprocess", role_aware=True),
    Probe("forward", "onnx_infer.session:OnnxModel.run", role_aware=True),
    Probe("forward", "trt_infer.session:TrtModel.run_torch", role_aware=True),
    Probe("forward", "trt_infer.session:TrtModel.run", role_aware=True),
    Probe("postprocess", "onnx_infer.adapter:to_friendy", role_aware=True),
    Probe("postprocess", "trt_infer.postprocess_torch:to_friendy_torch", role_aware=True),
    Probe("postprocess", "trt_infer.adapter:to_friendy", role_aware=True),
    Probe("merge_nms", "chachak.boxes:_class_aware_overlap_nms"),
    # ``run_batch`` calls the names it imported into the bundle's own ``infer``
    # module, so the binding to patch for those two lives there. Patching
    # ``chachak.boxes.merge_predictions`` instead resolves fine and then never fires
    # — a probe that "installs" and reports zero, which is the failure mode this
    # whole binding-site distinction exists to avoid.
    Probe("merge_nms", f"{BUNDLE_INFER_MODULE}:merge_predictions"),
    Probe("remap", f"{BUNDLE_INFER_MODULE}:remap_raw_predictions_to_eval_classes"),
)

# Stages whose id is built by the role prefix rather than named outright.
_ROLE_STAGES = ("preprocess", "forward", "postprocess")


@dataclass
class StagePlan:
    """What will be measured, and where the plan came from."""

    probes: Tuple[Probe, ...]
    taxonomy: Tuple[Tuple[str, str, Optional[str], str], ...]
    source: str  # "probe-table" | "bundle-declaration"


def load_stage_plan(bundle_dir, manifest: Optional[dict] = None) -> StagePlan:
    """The stage plan for one bundle: its own declaration if it has one, else the
    built-in probe table.

    A declaration is a list of ``{id, label, target, parent, role_aware}`` dicts under
    ``BENCHMARK_STAGES`` in ``runtime/benchmark_stages.py``, or under the
    ``benchmark_stages`` key of ``pipeline.json``. Anything malformed falls back to the
    probe table rather than failing the run — a benchmark that refuses to start is
    worse than one that reports ``unavailable``.
    """
    from pathlib import Path

    declared = None
    if manifest:
        declared = manifest.get("benchmark_stages")
    if declared is None:
        decl_path = Path(bundle_dir) / "runtime" / "benchmark_stages.py"
        if decl_path.is_file():
            namespace: Dict[str, Any] = {}
            try:
                exec(compile(decl_path.read_text(), str(decl_path), "exec"), namespace)
                declared = namespace.get("BENCHMARK_STAGES")
            except Exception:  # noqa: BLE001 - a bad declaration must not fail the run
                declared = None

    if not isinstance(declared, list) or not declared:
        return StagePlan(PROBES, STAGE_TAXONOMY, "probe-table")

    probes: List[Probe] = []
    taxonomy: List[Tuple[str, str, Optional[str], str]] = []
    try:
        for entry in declared:
            stage_id = str(entry["id"])
            label = str(entry.get("label") or stage_id)
            parent = entry.get("parent") or None
            target = entry.get("target")
            kind = "direct" if target else str(entry.get("kind") or "residual")
            taxonomy.append((stage_id, label, parent, kind))
            if target:
                probes.append(
                    Probe(stage_id, str(target), bool(entry.get("role_aware", False)))
                )
    except (KeyError, TypeError, ValueError):
        return StagePlan(PROBES, STAGE_TAXONOMY, "probe-table")

    return StagePlan(tuple(probes), tuple(taxonomy), "bundle-declaration")


# ── the recorder ────────────────────────────────────────────────────────────────


@dataclass
class _Frame:
    stage: str
    started: float
    child_time: float = 0.0


@dataclass
class _Totals:
    inclusive_s: float = 0.0
    child_s: float = 0.0
    calls: int = 0


class StageRecorder:
    """Wraps a bundle's runtime and accumulates per-stage inclusive/exclusive time.

    ``sync`` inserts a ``torch.cuda.synchronize()`` at every stage boundary, without
    which per-stage wall time on CUDA is meaningless (the work is still queued). It
    also perturbs the total — which is precisely why this recorder runs in its own
    pass and never in the pass that produces the headline latency.
    """

    def __init__(self, plan: StagePlan, *, sync: bool = True, device: str = "cpu"):
        self.plan = plan
        self.sync = sync and str(device).startswith("cuda")
        self._totals: Dict[str, _Totals] = {}
        self._local = threading.local()
        self._patched: List[Tuple[Any, str, Any]] = []
        self._resolved: set[str] = set()
        self._unresolved: List[str] = []
        self._lock = threading.Lock()

    # -- stack helpers (thread-local: the concurrency sweep runs one runtime per
    # -- worker thread, and a frame opened on one thread must not tag another's) --
    @property
    def _stack(self) -> List[_Frame]:
        stack = getattr(self._local, "stack", None)
        if stack is None:
            stack = []
            self._local.stack = stack
        return stack

    def _resolve_stage(self, probe: Probe) -> Optional[str]:
        """Which stage this probe's hit belongs to, or ``None`` to pass through.

        ``None`` is not an error case. The person detector runs the *same* adapter
        class as the trained model, so inside a ``person`` frame the adapter-predict
        probe would open a second frame that is, in time terms, the detector all over
        again — and both would accumulate under ``person``, making it larger than the
        pipeline that contains it. Passing through leaves the detector's inclusive time
        measured once, by the detector probe, while the nested preprocess/forward/
        postprocess probes still attribute inside it.
        """
        if not probe.role_aware:
            return probe.stage
        inside_person = any(frame.stage.startswith("person") for frame in self._stack)
        role = "person" if inside_person else "model"
        if probe.stage in _ROLE_STAGES:
            return f"{role}.{probe.stage}"
        if probe.stage == "model":
            return None if inside_person else "model"
        return f"{role}.{probe.stage}"

    def _torch_sync(self) -> None:
        if not self.sync:
            return
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:  # noqa: BLE001 - a sync failure must not kill the run
            pass

    def _wrap(self, probe: Probe, fn: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            stage = self._resolve_stage(probe)
            if stage is None:
                return fn(*args, **kwargs)
            stack = self._stack
            self._torch_sync()
            frame = _Frame(stage, time.perf_counter())
            stack.append(frame)
            try:
                return fn(*args, **kwargs)
            finally:
                stack.pop()
                self._torch_sync()
                elapsed = time.perf_counter() - frame.started
                if stack:
                    stack[-1].child_time += elapsed
                with self._lock:
                    totals = self._totals.setdefault(stage, _Totals())
                    totals.inclusive_s += elapsed
                    totals.child_s += frame.child_time
                    totals.calls += 1

        wrapper.__name__ = getattr(fn, "__name__", "wrapped")
        wrapper.__doc__ = getattr(fn, "__doc__", None)
        wrapper._stage_probe = probe  # type: ignore[attr-defined]
        return wrapper

    def _owner_and_attr(self, target: str) -> Optional[Tuple[Any, str]]:
        """Resolve ``"module.path:Class.method"`` to the object to setattr on."""
        import importlib

        module_path, _, attr_path = target.partition(":")
        try:
            owner: Any = importlib.import_module(module_path)
        except Exception:  # noqa: BLE001 - this flavour isn't in this bundle
            return None
        parts = attr_path.split(".")
        for part in parts[:-1]:
            owner = getattr(owner, part, None)
            if owner is None:
                return None
        if not hasattr(owner, parts[-1]):
            return None
        return owner, parts[-1]

    def install(self) -> None:
        """Patch every probe whose target exists in this bundle's runtime."""
        for probe in self.plan.probes:
            found = self._owner_and_attr(probe.target)
            if found is None:
                self._unresolved.append(probe.target)
                continue
            owner, attr = found
            original = getattr(owner, attr)
            if getattr(original, "_stage_probe", None) is not None:
                continue  # already wrapped (a shared base method reached twice)
            setattr(owner, attr, self._wrap(probe, original))
            self._patched.append((owner, attr, original))
            self._resolved.add(probe.stage)

    def uninstall(self) -> None:
        for owner, attr, original in reversed(self._patched):
            setattr(owner, attr, original)
        self._patched.clear()

    def __enter__(self) -> "StageRecorder":
        self.install()
        return self

    def __exit__(self, *exc_info) -> None:
        self.uninstall()

    def reset(self) -> None:
        with self._lock:
            self._totals.clear()

    # -- reporting ---------------------------------------------------------------
    def _incl_ms(self, stage: str, divisor: float) -> Optional[float]:
        totals = self._totals.get(stage)
        if totals is None:
            return None
        return totals.inclusive_s * 1000.0 / divisor

    def stages(self, total_s: float, frames: int,
               decode_ms_per_frame: Optional[float] = None) -> List[dict]:
        """Per-frame stage table for the result JSON.

        ``total_s`` is the harness-measured ``run_batch`` total over ``frames`` frames;
        residual stages are derived from it so that siblings plus residual equal the
        parent at every level.
        """
        divisor = max(1, frames)
        total_ms = total_s * 1000.0 / divisor

        def incl(stage: str) -> Optional[float]:
            return self._incl_ms(stage, divisor)

        def residual(parent_ms: Optional[float], children: List[str]) -> Optional[float]:
            if parent_ms is None:
                return None
            taken = sum(incl(child) or 0.0 for child in children)
            return max(0.0, parent_ms - taken)

        computed: Dict[str, Optional[float]] = {}
        for stage_id, _label, _parent, kind in self.plan.taxonomy:
            if kind == "direct":
                computed[stage_id] = incl(stage_id)

        # Decode happens outside run_batch (the harness owns it), so it is handed in
        # already reduced to a per-frame figure rather than probed.
        computed["decode"] = decode_ms_per_frame
        for role in ("person", "model"):
            computed[f"{role}.other"] = residual(
                computed.get(role),
                [f"{role}.preprocess", f"{role}.forward", f"{role}.postprocess"],
            )
        computed["assemble"] = residual(computed.get("pipeline"), ["person", "model"])
        computed["unattributed"] = max(
            0.0,
            total_ms
            - sum(
                computed.get(stage) or 0.0
                for stage in ("pipeline", "merge_nms", "remap")
            ),
        )

        rows: List[dict] = []
        for stage_id, label, parent, kind in self.plan.taxonomy:
            ms = computed.get(stage_id)
            totals = self._totals.get(stage_id)
            if ms is None:
                coverage = "unavailable"
            elif kind == "residual":
                coverage = "residual"
            elif kind == "harness":
                coverage = "harness"
            else:
                coverage = "direct"
            rows.append({
                "id": stage_id,
                "label": label,
                "parent": parent,
                "depth": 0 if parent is None else (1 if "." not in stage_id else 2),
                "ms": None if ms is None else round(ms, 4),
                "pct": None if (ms is None or total_ms <= 0) else round(ms / total_ms * 100, 2),
                "calls_per_frame": None if totals is None else round(totals.calls / divisor, 3),
                "coverage": coverage,
            })
        return rows

    def diagnostics(self) -> dict:
        return {
            "plan_source": self.plan.source,
            "synchronized": self.sync,
            "probes_resolved": sorted(self._resolved),
            # Overwhelmingly the *other* runtime flavour: an ONNX bundle vendors no
            # trt_infer and vice versa, so these are expected absences, not faults.
            "probes_absent": sorted(set(self._unresolved)),
        }
