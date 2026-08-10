"""Generate the bundle's ``README.md``.

Written from the finished manifest rather than a static template, so the document
describes the bundle that actually exists — its real classes, its real pipeline
stages, its real confidence default — instead of a generic placeholder the reader
has to translate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .manifest import class_map

# What each pipeline does, in the terms a consumer cares about: how many times the
# model sees the frame and why.
_PIPELINE_BLURB = {
    "batch_detect": (
        "The frame is cut into overlapping tiles, the model runs on each tile, and the "
        "per-tile detections are mapped back onto the full frame and merged with NMS. "
        "Tiling exists to keep small objects large enough for the model to see them."
    ),
    "people_detect_first": (
        "A person detector runs on the full frame, each person box is expanded and "
        "cropped out, the model runs on each crop, and the detections are mapped back "
        "onto the full frame. The model therefore only ever looks at people."
    ),
    "batch_people": (
        "The frame is tiled, the person detector runs per tile (so distant people stay "
        "large enough to detect), the person boxes are merged back onto the frame, and "
        "the model runs on each person crop."
    ),
    "chain": (
        "Several pipelines run over the same frame and their detections are pooled and "
        "merged with NMS."
    ),
}


def render_readme(manifest: Dict[str, Any], exported: Dict[str, Dict[str, Any]]) -> str:
    name = manifest["pipeline"]
    chain = manifest.get("chain") or []
    bundle = manifest["bundle"]
    defaults = manifest["defaults"]
    classes = class_map(manifest)
    artifacts = manifest["artifacts"]
    fmt = bundle["default_format"]
    # Set only when the bundle was exported with the GPU entrypoint (see build_manifest).
    gpu_infer = bool(bundle.get("gpu_infer"))

    lines = [
        f"# {bundle['name']}",
        "",
        f"A self-contained object-detection pipeline, exported {bundle['created_utc']}.",
        "Everything needed to run it is in this directory.",
        "",
        "## Run it",
        "",
        "```sh",
        "pip install -r requirements.txt",
        f"python infer.py path/to/images --conf {defaults['conf']}",
        "```",
        "",
        "`images` can be a single image or a directory (searched recursively). Results go",
        "to `results/detections.json`; add `--save-images` to also get annotated copies.",
        "",
        "```sh",
        "python infer.py --help          # every option",
        "```",
        "",
        "## The one setting you should tune",
        "",
        f"`--conf` is the confidence threshold, defaulting to **{defaults['conf']}** —",
        "the value this pipeline was exported with. Raise it for fewer, surer detections;",
        "lower it to catch more and accept false positives.",
        "",
        "Everything else in `pipeline.json` — tile geometry, person-crop expansion, the",
        "NMS thresholds that merge overlapping regions — was tuned together with the",
        "weights. Changing those changes what the pipeline detects, so they are not",
        "command-line options.",
        "",
        "## Classes",
        "",
        "| id | name |",
        "| --- | --- |",
    ]
    lines += [f"| {class_id} | {class_name} |" for class_id, class_name in sorted(classes.items())]
    lines += [
        "",
        "## What runs",
        "",
        f"**Pipeline: `{name}`"
        + (f" ({' + '.join(chain)})`" if chain else "")
        + "**",
        "",
        _PIPELINE_BLURB.get(name, "A chachak detection pipeline."),
        "",
        "| role | file | architecture | classes |",
        "| --- | --- | --- | --- |",
    ]
    role_files = {"model": artifacts["model"], "detector": artifacts.get("detector")}
    for index, extra in enumerate(artifacts.get("extra_models") or [], start=1):
        role_files[f"extra_{index}"] = extra
    for role, info in exported.items():
        lines.append(
            f"| {role} | `{role_files.get(role, '-')}` | {info['arch']} | "
            f"{', '.join(str(v) for v in (info['classes'] or {}).values()) or '-'} |"
        )

    lines += [
        "",
        "## Format",
        "",
        f"This bundle carries **{fmt}** artifacts; `infer.py` runs them by default.",
        "",
    ]
    # A role whose artifact couldn't be produced in the bundle's own format (a
    # prebuilt engine can't be turned back into ONNX) is carried in the format it
    # exists in — each role loads by its own suffix. Say so, because it changes
    # where the bundle can run: one `.engine` locks the whole pipeline to that GPU.
    other_format = sorted(
        f"`{role}` ({Path(str(path)).suffix.lstrip('.')})"
        for role, path in role_files.items()
        if path and not str(path).endswith(f".{fmt}")
    )
    if other_format:
        lines += [
            "Except " + ", ".join(other_format) + ": that artifact was supplied "
            "prebuilt and cannot be converted, so this bundle is mixed-format. It runs "
            "as-is, but anything carrying a `.engine` only runs on the GPU model and "
            "TensorRT version that engine was built on.",
            "",
        ]
    if fmt == "engine":
        lines += [
            "The `.engine` files are TensorRT plans, tied to the exact GPU model and",
            "TensorRT version they were built on. They will not load elsewhere. `models/`",
            "also has the portable `.onnx` this bundle was compiled from, in case you want",
            "to rebuild an engine for different hardware or run on CPU — but `pipeline.json`",
            "points at the `.engine` files, so that requires pointing `infer.py`'s runtime",
            "at the `.onnx` paths yourself.",
            "",
        ]
    else:
        lines += [
            "The `.onnx` graphs run anywhere onnxruntime does, on CPU or GPU. Install",
            "`onnxruntime-gpu` instead of `onnxruntime` and pass `--device cuda` for GPU.",
            "",
        ]

    lines += [
        "## Output",
        "",
        "`results/detections.json`:",
        "",
        "```json",
        "{",
        '  "conf": ' + str(defaults["conf"]) + ",",
        '  "classes": {"0": "' + (classes.get(0) or "class_a") + '"},',
        '  "images": [',
        "    {",
        '      "image": "frame_001.jpg",',
        '      "width": 1920, "height": 1080,',
        '      "detections": [',
        "        {",
        '          "class_id": 0, "class_name": "'
        + (classes.get(0) or "class_a")
        + '", "confidence": 0.87,',
        '          "box_xyxy": [812.4, 233.1, 902.0, 361.7],',
        '          "box_xywhn": [0.446, 0.275, 0.047, 0.119]',
        "        }",
        "      ]",
        "    }",
        "  ]",
        "}",
        "```",
        "",
        "`box_xyxy` is in that image's own pixels (top-left, bottom-right).",
        "`box_xywhn` is centre-x, centre-y, width, height normalized to 0..1.",
        "Detections are sorted by confidence, highest first.",
        "",
        "## What's in here",
        "",
        "```",
        "infer.py          the entrypoint",
        "pipeline.json     the pipeline: stages, geometry, thresholds, class space",
        "models/           the exported weights + their preprocessing sidecars",
        "runtime/          the inference code (vendored; no install needed)",
        "requirements.txt",
    ] + ([
        "infer_gpu.py      GPU-only entrypoint (optional, see below)",
        "requirements-gpu.txt",
    ] if gpu_infer else []) + [
        "```",
        "",
        "`pipeline.json` also records, under `provenance`, the checkpoints and request",
        "this bundle was built from.",
        "",
    ] + ([
        "## GPU inference",
        "",
        "`infer_gpu.py` runs this same pipeline entirely on the GPU: frames stay in device",
        "memory from decode to detections, and engine work is submitted without blocking",
        "between calls. It needs a CUDA GPU and the `.engine` artifacts.",
        "",
        "```",
        "pip install -r requirements.txt -r requirements-gpu.txt",
        f"python infer_gpu.py path/to/images --conf {defaults['conf']}",
        "```",
        "",
        "Output is identical to `infer.py`'s, byte for byte, so switching needs no",
        "re-validation. `infer.py` still works exactly as documented above and is the one",
        "to use without a GPU.",
        "",
        "One flag deviates deliberately: `--gpu-decode` decodes JPEGs with nvJPEG, which is",
        "faster but *not* byte-identical to PIL (on 4:2:0 content the median pixel difference",
        "is 0, but ~12% of pixels differ by at least 1/255). It is off unless asked for.",
        "",
    ] if gpu_infer else [])
    return "\n".join(lines)
