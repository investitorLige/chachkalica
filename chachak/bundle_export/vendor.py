"""Assemble the bundle's ``runtime/`` — the code that runs the pipeline.

A bundle has to execute a chachak pipeline on a machine that does not have this
repo, so the runtime is copied in. Every file is copied **verbatim at export
time** rather than reimplemented, so a bundle can never drift from the pipeline
the exporter actually ran: change ``chachak/pipeline.py`` and the next bundle
carries the change.

Only two files are generated (from ``templates/``), and only because the
originals reach into the training toolkit:

* ``chachak/_friendy.py`` — the real shim imports all of ``friendy_chachkalica``
  (torchvision detection heads, transformers, the dataset loader). The bundle's
  version re-exports the handful of pure functions chachak's inference path
  touches from ``_vendor/`` and stubs the training-only names.
* ``chachak/__init__.py`` — the real one imports ``run.py``, whose batch-eval
  entrypoint needs the Friendy dataloader. The bundle's entrypoint is
  ``infer.py`` at the bundle root instead.

Nothing else is edited, and ``chachak``'s modules already dual-import
(``from ._friendy`` / ``from _friendy``), so they load unchanged.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TEMPLATES = Path(__file__).resolve().parent / "templates"

# chachak modules the inference path needs. `run.py` (batch eval over a Friendy
# dataloader) and `_friendy.py` (the training-toolkit bridge) are deliberately out.
_CHACHAK_MODULES = (
    "boxes.py",
    "config.py",
    "detector.py",
    "infer.py",
    "pipeline.py",
    "preview.py",
    "registry.py",
)

# friendy_chachkalica modules vendored under `chachak/_vendor/`, reachable through
# the generated `_friendy.py` shim. Each imports nothing beyond stdlib, torch, and
# the others listed here; anything heavier they touch is a function-local import
# the inference path never reaches. `ml/val.py` is deliberately absent — its
# module-level imports pull in the trainer, so the two serialization helpers
# chachak takes from it are stubbed in the shim instead (only `Pipeline.run()`,
# the batch-eval entrypoint, ever calls them).
_FRIENDY_MODULES = (
    ("formats.py", "formats.py"),        # box-format conversions      <- boxes.py
    ("metrics.py", "metrics.py"),        # eval + class remap          <- pipeline.py
    ("postprocess.py", "postprocess.py"),  # class-aware NMS           <- metrics.py
    ("device.py", "device.py"),          # resolve_device              <- infer.py (bundle)
)


def _copy_package(source: Path, destination: Path) -> None:
    """Copy a python package, skipping tests and caches."""
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns("__pycache__", "tests", "*.pyc", "PLAN.md"),
    )


def vendor_runtime(
    runtime_dir: Path, *, include_trt: bool, include_gpu: bool = False
) -> List[str]:
    """Populate ``runtime_dir``; returns the top-level entries written.

    ``include_gpu`` adds the ``gpu_infer`` package, which backs the optional
    ``infer_gpu.py`` entrypoint. Defaults off, so a bundle exported without the option has a
    byte-for-byte unchanged ``runtime/``.
    """
    runtime_dir.mkdir(parents=True, exist_ok=True)

    chachak_dir = runtime_dir / "chachak"
    vendor_dir = chachak_dir / "_vendor"
    vendor_dir.mkdir(parents=True, exist_ok=True)

    for name in _CHACHAK_MODULES:
        shutil.copy2(_REPO_ROOT / "chachak" / name, chachak_dir / name)
    for source, name in _FRIENDY_MODULES:
        shutil.copy2(_REPO_ROOT / "friendy_chachkalica" / source, vendor_dir / name)

    shutil.copy2(_TEMPLATES / "chachak_init.py", chachak_dir / "__init__.py")
    shutil.copy2(_TEMPLATES / "friendy_shim.py", chachak_dir / "_friendy.py")
    (vendor_dir / "__init__.py").write_text(
        '"""Verbatim copies of the self-contained friendy_chachkalica modules that\n'
        'chachak\'s inference path imports. Do not edit — regenerate the bundle."""\n'
    )

    # onnx_infer runs the .onnx artifacts (numpy + onnxruntime; torch only at the
    # adapter boundary). trt_infer reuses its whole numpy core, so it is never
    # bundled alone.
    _copy_package(_REPO_ROOT / "onnx_infer", runtime_dir / "onnx_infer")
    entries = ["chachak", "onnx_infer"]
    if include_trt:
        _copy_package(_REPO_ROOT / "trt_infer", runtime_dir / "trt_infer")
        entries.append("trt_infer")

    # gpu_infer runs the same pipelines fully GPU-resident (infer_gpu.py). Opt-in, and only
    # useful alongside trt_infer since it is engine-only. _copy_package strips its tests/,
    # which matters: those import chachak and would fail the bundle's import-closure check.
    if include_gpu:
        _copy_package(_REPO_ROOT / "gpu_infer", runtime_dir / "gpu_infer")
        entries.append("gpu_infer")

    # The manifest contract, shared by the exporter and the bundle's infer.py so
    # pipeline.json has exactly one definition.
    shutil.copy2(Path(__file__).resolve().parent / "manifest.py", runtime_dir / "bundle_manifest.py")
    entries.append("bundle_manifest.py")
    return sorted(entries)
