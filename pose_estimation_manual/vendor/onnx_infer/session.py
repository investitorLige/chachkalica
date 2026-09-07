"""onnxruntime session wrapper — owns the ``InferenceSession`` and provider
selection, and runs a single pre-processed batch through the graph.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .meta import ModelMeta


def providers_for(device) -> list[str]:
    """ORT execution providers for a torch-style device string/object."""
    name = str(device).lower()
    if "cuda" in name or "gpu" in name:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


class OnnxModel:
    """A loaded ONNX graph + its meta, ready to run one ``[1,3,H,W]`` batch."""

    def __init__(self, onnx_path: str | Path, meta: ModelMeta, device="cpu") -> None:
        import onnxruntime as ort  # local import: keeps the module import cheap

        self.meta = meta
        self.path = Path(onnx_path)
        self._ort = ort
        self._device = str(device)
        self.session = ort.InferenceSession(str(self.path), providers=providers_for(device))
        self.input_name = self.session.get_inputs()[0].name

    def to(self, device) -> "OnnxModel":
        """Re-create the session on a new device only if the provider set changes."""
        if providers_for(device) != providers_for(self._device):
            self._device = str(device)
            self.session = self._ort.InferenceSession(
                str(self.path), providers=providers_for(device)
            )
            self.input_name = self.session.get_inputs()[0].name
        return self

    def run(self, batched: np.ndarray) -> list[np.ndarray]:
        return self.session.run(None, {self.input_name: batched})
