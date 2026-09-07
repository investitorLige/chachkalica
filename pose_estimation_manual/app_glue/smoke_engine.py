#!/usr/bin/env python
"""Load a model artifact and run one image through it. No Django, no cameras.

**Run this first**, inside the GPU image, before registering anything in the admin:

    docker compose run --rm gpu-inference \\
        python scripts/smoke_engine.py /data/bundles/ppe-v0.1
    docker compose run --rm gpu-inference \\
        python scripts/smoke_engine.py /data/bundles/ppe-v0.1/models/model.engine

It answers the one question that everything else depends on and that nothing else
answers cleanly: *does this artifact deserialize under this container's TensorRT?*
An engine only loads under the version it was built with, and the failure otherwise
happens inside the worker's reconcile loop where it looks like a configuration
problem rather than a version problem.

Point it at a whole bundle directory to resolve ``pipeline.json`` and check the
model artifact it names, or straight at a ``.engine``/``.onnx`` file (inside or
outside a bundle) for a quicker single-artifact check.

With no image given it runs a synthetic frame, which is enough to prove the
artifact loads, binds, and produces output of the right shape. Pass a real image
to also sanity-check that the detections are plausible.
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Run from the repo root without installing anything.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _resolve_target(target: Path) -> Path:
    """A bundle directory -> its pipeline.json's model artifact; a file -> itself."""
    if target.is_dir():
        manifest_path = target / "pipeline.json"
        if not manifest_path.exists():
            print(f"FAIL: not a bundle — no pipeline.json in {target}", file=sys.stderr)
            raise SystemExit(2)
        manifest = json.loads(manifest_path.read_text())
        model_relative = (manifest.get("artifacts") or {}).get("model")
        if not model_relative:
            print(f"FAIL: {manifest_path} has no artifacts.model", file=sys.stderr)
            raise SystemExit(2)
        print(f"bundle       {manifest.get('name') or target.name} "
              f"pipeline={manifest.get('pipeline')!r} -> {model_relative}")
        return target / model_relative
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("engine", help="Path to a bundle directory, or a .engine/.onnx artifact.")
    parser.add_argument("--image", help="Optional image to run instead of a synthetic frame.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--repeat", type=int, default=3,
                        help="Inferences to time. The first includes warm-up.")
    parser.add_argument("--save", help="Write an annotated copy here.")
    args = parser.parse_args()

    target = Path(args.engine)
    if not target.exists():
        print(f"FAIL: no such path: {target}", file=sys.stderr)
        return 2
    engine_path = _resolve_target(target)
    if not engine_path.exists():
        print(f"FAIL: no such artifact: {engine_path}", file=sys.stderr)
        return 2

    # ── 1. the sidecars, before touching the GPU ──
    meta_path = engine_path.with_suffix(".meta.json")
    if not meta_path.exists():
        print(f"FAIL: Contract-B sidecar missing: {meta_path}", file=sys.stderr)
        return 2

    from vendor.onnx_infer.meta import ModelMeta

    meta = ModelMeta.load(meta_path)
    print(f"meta.json    arch={meta.arch} classes={meta.class_map} "
          f"resize={meta.input.resize_mode}/{meta.input.size} "
          f"scale={meta.input.input_scale} layout={meta.layout}")

    provenance_path = Path(str(engine_path) + ".json")
    built_with = None
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text())
        built_with = provenance.get("tensorrt_version")
        print(f"engine.json  precision={provenance.get('precision')} "
              f"built_with_tensorrt={built_with} batch={provenance.get('batch')}")

    # ── 2. the version question ──
    if engine_path.suffix.lower() == ".engine":
        try:
            import tensorrt as trt

            print(f"runtime      tensorrt={trt.__version__}")
            if built_with and trt.__version__.split(".")[0] != built_with.split(".")[0]:
                print(f"WARN: engine was built with TensorRT {built_with} but this "
                      f"container has {trt.__version__}. A major-version mismatch "
                      f"will fail to deserialize — see docker/Dockerfile.gpu.")
        except ImportError:
            print("FAIL: tensorrt is not installed in this environment. Run this "
                  "inside the gpu-inference container.", file=sys.stderr)
            return 2

    # ── 3. load it ──
    from vendor.chachak import load_checkpoint_adapter

    started = time.monotonic()
    try:
        adapter, info = load_checkpoint_adapter(engine_path, args.device)
    except Exception as exc:  # noqa: BLE001 - this is the whole point of the script
        print(f"\nFAIL: could not load the engine: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        print("\nIf this mentions a serialization or version mismatch, the engine was "
              "built with a different TensorRT than this container has. Either rebuild "
              "the engine, or pin TENSORRT_IMAGE to match.", file=sys.stderr)
        return 1
    print(f"loaded       in {time.monotonic() - started:.1f}s "
          f"({info.get('model_name')}, {info.get('num_classes')} classes)")

    # ── 4. run it ──
    import numpy as np

    if args.image:
        import cv2

        bgr = cv2.imread(args.image)
        if bgr is None:
            print(f"FAIL: could not read image: {args.image}", file=sys.stderr)
            return 2
        print(f"image        {args.image} {bgr.shape[1]}x{bgr.shape[0]}")
    else:
        bgr = np.random.default_rng(0).integers(
            0, 255, size=(720, 1280, 3), dtype=np.uint8
        )
        print("image        synthetic 1280x720 (pass --image for a real one)")

    from detections.services.engine_runtime import _bgr_to_chw

    chw = _bgr_to_chw(bgr)

    timings = []
    preds = None
    for index in range(max(1, args.repeat)):
        started = time.monotonic()
        preds = adapter.predict([chw], score_threshold=args.threshold)[0]
        timings.append((time.monotonic() - started) * 1000)
        label = " (includes warm-up)" if index == 0 else ""
        print(f"inference {index + 1}    {timings[-1]:.0f} ms{label}")
    if len(timings) > 1:
        print(f"steady state {min(timings[1:]):.0f} ms best of {len(timings) - 1}")

    # ── 5. report what came out ──
    from detections.services import annotate

    class _Names:
        def class_name(self, class_id):
            return meta.class_map.get(int(class_id), f"class_{int(class_id)}")

    boxes = annotate.friendy_to_dicts(preds, engine=_Names(),
                                      score_threshold=args.threshold)
    print(f"\ndetections   {len(boxes)} above {args.threshold}")
    for box in boxes[:10]:
        print(f"  {box['class_name']:<16} {box['confidence']:.3f}  "
              f"centre=({box['cx']:.3f}, {box['cy']:.3f}) "
              f"size=({box['w']:.3f} x {box['h']:.3f})")
    if len(boxes) > 10:
        print(f"  … and {len(boxes) - 10} more")

    if args.save:
        jpeg, width, height = annotate.annotate(bgr, boxes)
        Path(args.save).write_bytes(jpeg)
        print(f"\nwrote        {args.save} ({width}x{height})")

    print("\nOK: the engine loads and runs in this environment.")
    if not boxes and args.image:
        print("Note: nothing was detected. On a real image that may mean the "
              "threshold is too high, or the model needs a pipeline (tiling / person "
              "crops) rather than the whole frame — see the bundle's pipeline.json.")
    return 0


if __name__ == "__main__":
    import os

    # detections.services imports Django settings.
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "watchroom.settings")
    os.environ.setdefault("DJANGO_DB", "sqlite")
    import django

    django.setup()
    sys.exit(main())
