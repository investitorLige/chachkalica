"""An arch's exported ``clip_boxes`` must match whether its torch path clips.

``onnx_infer``/``trt_infer`` clamp predictions to the image only when the artifact's
``meta.json`` says ``clip_boxes: true`` (``onnx_infer/postprocess.py``). The flag's
whole job is parity with the adapter the artifact was exported from: an unclipped
export has larger boxes than its own torch path wherever a prediction crosses a
frame edge, so IoU against edge-touching ground truth drops and near-threshold
matches flip to misses — the exported model then scores worse than the checkpoint
for postprocessing reasons alone.

Both sides are one-line decisions in two files that nothing links, and they *have*
drifted: RT-DETR's adapter started clipping while its exporter kept the default
``False``, and the contract comment in ``onnx_infer/meta.py`` documented the stale
state as intentional. This is a source-level check (in the spirit of
``chachak/tests/test_bundle_export.py``'s vendored-import scan) because the real
export path needs a checkpoint and the training stack to run.
"""

import unittest
from pathlib import Path

_ML = Path(__file__).resolve().parent.parent / "ml"
_ADAPTERS = _ML / "adapters"
_EXPORTS = _ML / "onnx_export" / "arch"


def _arch_modules():
    """Arch names that have both an adapter and an ONNX exporter."""
    for export in sorted(_EXPORTS.glob("*.py")):
        if export.name == "__init__.py":
            continue
        adapter = _ADAPTERS / export.name
        if adapter.is_file():
            yield export.stem, adapter, export


class ClipBoxesParityTests(unittest.TestCase):
    def test_every_arch_clips_on_both_paths_or_neither(self):
        for arch, adapter, export in _arch_modules():
            with self.subTest(arch=arch):
                torch_clips = "clip_xyxy(" in adapter.read_text()
                export_clips = "clip_boxes=True" in export.read_text()
                self.assertEqual(
                    torch_clips, export_clips,
                    f"{arch}: adapter clips={torch_clips} but exported "
                    f"clip_boxes={export_clips}. Whichever side is right, both have to "
                    f"agree — see onnx_infer/meta.py's clip_boxes contract.",
                )

    def test_the_check_sees_the_archs_it_should(self):
        # A rename that silently emptied the scan would make the test above vacuous.
        found = {arch for arch, _, _ in _arch_modules()}
        self.assertLessEqual({"rtdetr", "rfdetr", "yolox", "retinanet"}, found)


if __name__ == "__main__":
    unittest.main()
