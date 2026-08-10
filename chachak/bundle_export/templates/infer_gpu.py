#!/usr/bin/env python3
"""Run this bundle's detection pipeline entirely on the GPU.

    python infer_gpu.py path/to/images
    python infer_gpu.py path/to/images --conf 0.4 --save-images

A **sibling** of ``infer.py``, not a replacement. Both run the same pipeline, on the same
artifacts, with the same settings from ``pipeline.json``, and write the same
``detections.json``. The difference is where the work happens: this one keeps every frame in
device memory from decode to detections, and submits engine work without blocking the host
between calls.

Use ``infer.py`` when you want the reference path or have no GPU. Use this one for throughput.
By default the two produce **byte-identical** output — that is enforced by a differential test
against the reference pipeline, so switching costs no re-validation.

Requires a CUDA GPU and this bundle's ``.engine`` artifacts. An ONNX bundle cannot use it:
onnxruntime feeds numpy, so a frame cannot stay in device memory, which is this script's whole
point. It says so and exits rather than silently running the ordinary path.

Arguments
    images              Required, positional. A single image file or a directory, searched
                        recursively for .jpg/.jpeg/.png/.bmp/.webp/.tif/.tiff.

    --out DIR           Where results are written (default: "results").
    --conf FLOAT        Detection confidence threshold (default: the bundle's exported value).
    --detector-conf F   Person-detector threshold, for pipelines that crop people first.
    --save-images       Also write annotated copies under "<out>/annotated/".
    --device STR        "cuda" (default) or "cuda:N". CPU is not supported — see above.
    --batch-size INT    Regions (crops or tiles) per engine call. Frames are always processed
                        one at a time; batching frames buys nothing, since one frame's regions
                        already fill the engine.

    --gpu-decode        Decode JPEGs with nvJPEG on the device instead of PIL on the host.
                        OFF by default, deliberately: nvJPEG is not byte-identical to
                        libjpeg-turbo. Measured on 4:2:0 content the median per-pixel
                        difference is 0, but ~12% of pixels differ by >=1/255 and ~0.6% by
                        >=8/255, which has been observed to move a box. It is a real throughput
                        win (PIL decode can be most of a 4K frame's cost) but it is a different
                        input, so turn it on deliberately and compare.
    --strict-parity     Force the configuration proven byte-identical to infer.py, overriding
                        any flag above that would deviate. Use when a number has to match one
                        the training repo produced.

Output is identical in shape to infer.py's — see that script for the detections.json schema.

Requires: torch, numpy, pillow, pyyaml, tensorrt (see requirements.txt + requirements-gpu.txt).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

BUNDLE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BUNDLE_DIR / "runtime"))

import bundle_manifest as manifest_contract  # noqa: E402
from chachak.config import pipeline_config_from_dict  # noqa: E402


def _sibling_infer():
    """Load ``infer.py`` from this bundle by explicit path, for the helpers both scripts share.

    Imported rather than duplicated so the two entry points cannot drift on the output format —
    a consumer must not see different rounding or a different JSON shape depending on which
    script produced it. Loaded by *path* rather than by name so it can only ever resolve to this
    bundle's own ``infer.py``, whatever the working directory is.
    """
    path = BUNDLE_DIR / "infer.py"
    if not path.exists():
        raise SystemExit(
            f"[infer_gpu] {path.name} is missing from this bundle; infer_gpu.py shares its "
            f"output formatting and cannot run without it."
        )
    spec = importlib.util.spec_from_file_location("_bundle_infer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_infer = _sibling_infer()


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run this bundle's detection pipeline entirely on the GPU.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("images", help="Image file, or a directory searched recursively")
    parser.add_argument("--out", default="results", help="Directory for detections.json")
    parser.add_argument("--conf", type=float, default=None,
                        help="Detection threshold (default: the bundle's exported value)")
    parser.add_argument("--detector-conf", type=float, default=None,
                        help="Person-detector threshold, for pipelines that crop people")
    parser.add_argument("--save-images", action="store_true",
                        help="Also write annotated copies of each image")
    parser.add_argument("--device", default="cuda", help="cuda, cuda:1, ...")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Regions per engine call (frames are always one at a time)")
    parser.add_argument("--gpu-decode", action="store_true",
                        help="Decode JPEGs with nvJPEG; NOT byte-identical to PIL")
    parser.add_argument("--strict-parity", action="store_true",
                        help="Force the configuration proven identical to infer.py")
    return parser.parse_args(argv)


def _require_cuda(device_arg: str):
    import torch

    if not str(device_arg).lower().startswith("cuda"):
        raise SystemExit(
            f"[infer_gpu] --device {device_arg!r} is not supported. This script is GPU-only "
            f"by design; run infer.py for CPU or ONNX execution."
        )
    if not torch.cuda.is_available():
        raise SystemExit(
            "[infer_gpu] no CUDA GPU is visible to torch. Run infer.py instead, or check the "
            "driver / container GPU access."
        )
    return torch.device(device_arg)


def _resolve_engine_format(manifest) -> str:
    """Assert this bundle carries engines, with a message that says what to do instead."""
    try:
        return manifest_contract.resolve_format(manifest, "engine")
    except Exception:
        actual = manifest_contract.resolve_format(manifest, None)
        raise SystemExit(
            f"[infer_gpu] this bundle carries {actual!r} artifacts, not TensorRT engines. "
            f"A GPU-resident run needs engines — onnxruntime feeds numpy, so a frame cannot "
            f"stay in device memory. Use `python infer.py` for this bundle, or re-export it "
            f"with --format engine."
        ) from None


def _build_options(args):
    from gpu_infer.config import GpuOptions

    if args.strict_parity:
        options = GpuOptions.strict_parity()
        if args.gpu_decode:
            print(
                "[infer_gpu] --strict-parity overrides --gpu-decode: nvJPEG is not "
                "byte-identical to PIL, so it cannot be part of a parity run."
            )
        return options
    return GpuOptions(gpu_decode=args.gpu_decode)


def main(argv=None) -> None:
    args = _parse_args(argv)
    device = _require_cuda(args.device)

    manifest = manifest_contract.load_manifest(BUNDLE_DIR / "pipeline.json")
    fmt = _resolve_engine_format(manifest)
    paths = manifest_contract.artifact_paths(manifest, BUNDLE_DIR, fmt)
    if not _infer._needs_detector(manifest.get("pipeline"), manifest.get("chain") or []):
        paths["detector"] = None

    output_dir = Path(args.out).resolve()
    request = manifest_contract.pipeline_request_from_manifest(
        manifest,
        paths,
        fmt=fmt,
        images=args.images,
        output_dir=output_dir,
        conf=args.conf,
        detector_conf=args.detector_conf,
        device=str(device),
        batch_size=args.batch_size,
    )
    config = pipeline_config_from_dict(request, BUNDLE_DIR)

    from gpu_infer.decode import describe_decoder, load_frames
    from gpu_infer.loader import build_gpu_pipeline

    options = _build_options(args)
    for line in manifest_contract.describe(manifest):
        print(f"[infer_gpu] {line}")
    print(f"[infer_gpu] format={fmt} conf={config.score_threshold} device={device}")
    print(f"[infer_gpu] {describe_decoder(options.gpu_decode)}")

    pipeline = build_gpu_pipeline(
        config,
        paths["model"],
        device,
        detector_path=paths["detector"],
        extra_model_paths=paths.get("extra_models") or (),
        options=options,
        eval_classes=config.classes,
    )

    source = Path(args.images).resolve()
    image_paths = _infer._list_images(source)
    print(f"[infer_gpu] {len(image_paths)} image(s) from {source}")

    output_dir.mkdir(parents=True, exist_ok=True)
    class_map = manifest_contract.class_map(manifest)
    records: List[Dict[str, Any]] = []
    total = 0
    # Frames are processed one at a time; the chunk only batches *decoding*, which is where a
    # batch actually helps (nvJPEG takes a list).
    chunk_size = max(1, config.infer_batch_size)
    for start in range(0, len(image_paths), chunk_size):
        chunk = image_paths[start : start + chunk_size]
        frames = load_frames(chunk, device, gpu_decode=options.gpu_decode)
        for path, frame in zip(chunk, frames):
            prediction = pipeline.process_frame(frame).detach().cpu()
            frame_w, frame_h = int(frame.shape[2]), int(frame.shape[1])
            detections = _infer._detections(prediction, class_map, frame_w, frame_h)
            total += len(detections)
            relative = path.relative_to(source) if source.is_dir() else Path(path.name)
            records.append(
                {
                    "image": str(relative),
                    "width": frame_w,
                    "height": frame_h,
                    "detections": detections,
                }
            )
            if args.save_images:
                # Deliberately re-reads the source file and rasterizes on the CPU. Drawing from
                # the device tensor would need a full-frame copy back plus a synchronization,
                # which breaks the frame-to-frame overlap in exactly the mode that cares least.
                _infer._annotate(path, detections, output_dir / "annotated" / relative)
        print(
            f"[infer_gpu] {min(start + chunk_size, len(image_paths))}/{len(image_paths)} "
            f"images, {total} detections"
        )

    result_path = output_dir / "detections.json"
    result_path.write_text(
        json.dumps(
            {
                "bundle": manifest.get("bundle"),
                "format": fmt,
                "runtime": "gpu_infer",
                "conf": config.score_threshold,
                "detector_conf": (
                    config.detector.score_threshold if paths["detector"] is not None else None
                ),
                "device": str(device),
                "gpu_decode": options.gpu_decode,
                "classes": {str(key): value for key, value in class_map.items()},
                "images": records,
            },
            indent=2,
        )
    )
    print(f"[infer_gpu] Wrote {result_path} ({len(records)} images, {total} detections)")
    if args.save_images:
        print(f"[infer_gpu] Wrote annotated images to {output_dir / 'annotated'}")


if __name__ == "__main__":
    main()
