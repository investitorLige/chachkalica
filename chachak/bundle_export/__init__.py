"""Export a whole chachak pipeline as a self-contained, runnable bundle.

``friendy_chachkalica/ml/onnx_export`` makes one *model* portable: a ``.onnx``
graph plus a ``meta.json`` describing how to pre-process for it and read its
outputs. That is everything a single model needs — and nothing about the pipeline
wrapped around it. Tiling geometry, the person detector and its crop expansion,
the NMS thresholds that merge overlapping sub-regions: those live only in a
chachak request YAML full of paths to the exporter's own filesystem.

This package closes that gap. :func:`cli.export_bundle` takes a request YAML and
writes a directory containing:

* ``models/`` — every checkpoint the request names (model, extra models, person
  detector), exported to ONNX (and optionally TensorRT) with their sidecars;
* ``pipeline.json`` — Contract C (:mod:`.manifest`): every non-path field of the
  request, bundle-relative artifact paths, and the confidence defaults;
* ``runtime/`` — the chachak + onnx_infer code, copied verbatim at export time
  (:mod:`.vendor`) so a bundle can't drift from the pipeline that produced it;
* ``infer.py`` — the entrypoint, whose only routine argument is ``--conf``;
* ``README.md`` — generated from the manifest (:mod:`.readme`).

The recipient needs torch, numpy, pillow, pyyaml and onnxruntime — nothing from
this repo, and no architecture packages.
"""

from .manifest import SCHEMA_VERSION, build_manifest, load_manifest

__all__ = ["SCHEMA_VERSION", "build_manifest", "load_manifest"]
