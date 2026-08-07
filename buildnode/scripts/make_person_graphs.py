"""Regenerate the person-detector graphs a build node bakes into its image.

Writes two files next to the shipped person artifacts, under ``models/people/``:

    detector.onnx      + detector.meta.json       standard graph, for ONNX bundles
    detector.trt.onnx  + detector.trt.meta.json   EfficientNMS graph, compile input

Run it in the TRAINER container, which has torch and the checkpoint. It needs no
GPU — both outputs are torch ONNX exports:

    docker compose run --rm --no-deps -w /app trainer \\
        python buildnode/scripts/make_person_graphs.py

Why this is a script and not a step in the node's Dockerfile: producing the
EfficientNMS graph means loading the YOLOX ``.pt`` and re-exporting it, which
needs the whole training stack. Baking that into the slim image would defeat the
point of the slim image. So it runs once, here, and the node image just COPYs the
result. ``models/`` is gitignored, so the outputs live alongside the existing
``best_ckpt.engine`` as out-of-band deployment artifacts.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = REPO_ROOT / "person_model_test/export_test/best_ckpt.friendy.pt"
DEFAULT_META = REPO_ROOT / "person_model_test/export_test/best_ckpt.meta.json"
DEFAULT_OUT = REPO_ROOT / "models/people"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--meta",
        type=Path,
        default=DEFAULT_META,
        help="Contract-B sidecar to ship verbatim. This is the HAND-PATCHED Megvii "
        "meta (bgr/byte/pad=114); do not substitute a freshly exported one.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        print(f"error: no checkpoint at {args.checkpoint}", file=sys.stderr)
        return 1
    if not args.meta.is_file():
        print(f"error: no meta sidecar at {args.meta}", file=sys.stderr)
        return 1

    sys.path.insert(0, str(REPO_ROOT))
    from friendy_chachkalica.ml.onnx_export.cli import export_checkpoint
    from friendy_chachkalica.ml.trt_export.arch import get_trt_prep, has_trt_prep
    from friendy_chachkalica.ml.trt_export.cli import _load_adapter

    meta = json.loads(args.meta.read_text())
    arch = meta.get("arch")
    args.out.mkdir(parents=True, exist_ok=True)

    # 1. Standard graph. Export to a temp name, then overwrite the meta with the
    #    hand-patched one — the exporter would write rgb/unit for a YOLOX and
    #    silently mis-preprocess every frame.
    standard = args.out / "detector.onnx"
    print(f"[person] exporting standard graph -> {standard}")
    export_checkpoint(args.checkpoint, standard)
    (args.out / "detector.meta.json").write_text(json.dumps(meta, indent=2))

    # 2. EfficientNMS graph, for archs whose baked NMS TensorRT cannot compile.
    if has_trt_prep(arch):
        prepared = args.out / "detector.trt.onnx"
        print(f"[person] re-exporting {arch} with EfficientNMS -> {prepared}")
        adapter = _load_adapter(args.checkpoint)
        get_trt_prep(arch)(adapter, meta, prepared)
        (args.out / "detector.trt.meta.json").write_text(json.dumps(meta, indent=2))
        # prep writes a `.raw.onnx` intermediate beside its output; it is not needed.
        raw = prepared.with_name(prepared.name.replace(".onnx", ".raw.onnx"))
        if raw.exists():
            raw.unlink()
    else:
        print(f"[person] {arch} compiles from its standard graph; no EfficientNMS export")
        shutil.copy2(standard, args.out / "detector.trt.onnx")
        shutil.copy2(args.out / "detector.meta.json", args.out / "detector.trt.meta.json")

    for path in sorted(args.out.glob("detector.*")):
        print(f"  {path.name:28} {path.stat().st_size:>12,} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
