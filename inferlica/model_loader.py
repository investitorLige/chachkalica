"""Load a trained detector for inference regardless of on-disk format.

Dispatches by file extension to the matching adapter loader, all of which
return the same ``(adapter, info)`` shape and expose the same
``adapter.predict(images, score_threshold=...) -> list[Tensor(N, 6)]``
("Friendy" format) surface:

- ``.pt`` / ``.pth``           -> ``chachak.infer.load_checkpoint_adapter``
- ``.onnx``                    -> ``onnx_infer.load_onnx_adapter``
- ``.engine`` / ``.plan`` / ``.trt`` -> ``trt_infer.load_trt_adapter``
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

PT_EXTENSIONS = {".pt", ".pth"}
ONNX_EXTENSIONS = {".onnx"}
TRT_EXTENSIONS = {".engine", ".plan", ".trt"}


def load_adapter(model_path: str | Path, device) -> Tuple[Any, Dict[str, Any]]:
    model_path = Path(model_path)
    suffix = model_path.suffix.lower()

    if suffix in PT_EXTENSIONS:
        from chachak.infer import load_checkpoint_adapter

        return load_checkpoint_adapter(model_path, device)

    if suffix in ONNX_EXTENSIONS:
        from onnx_infer import load_onnx_adapter

        return load_onnx_adapter(model_path, device)

    if suffix in TRT_EXTENSIONS:
        from trt_infer import load_trt_adapter

        return load_trt_adapter(model_path, device)

    supported = sorted(PT_EXTENSIONS | ONNX_EXTENSIONS | TRT_EXTENSIONS)
    raise ValueError(f"Unsupported model extension {suffix!r} ({model_path}). Supported: {supported}")
