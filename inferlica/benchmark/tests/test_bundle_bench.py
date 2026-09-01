"""Unit tests for the bundle benchmark's own logic.

Deliberately torch-free and GPU-free: everything here is the *bookkeeping* that
decides whether a number in the report is trustworthy — stage arithmetic, the
``unavailable`` path, the declaration override, the consensus verdicts. Run them
anywhere:

    python -m unittest discover -s inferlica/benchmark/tests -t .

The end-to-end run against a real bundle is a separate, GPU-container exercise (see
the module docstring of ``bundle_bench``); pinning the arithmetic here is what stops a
refactor from quietly turning a residual into a wrong number.
"""

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark import bundle_stages  # noqa: E402
from benchmark.bundle_bench import (  # noqa: E402
    NSYS_TOP_KERNELS,
    _bare_input_shape,
    _kernel_rows,
    _parse_int_list,
    consensus,
    list_images,
    nsys_pass,
    resolve_nsys,
)
from benchmark.bundle_stages import Probe, StagePlan, StageRecorder, load_stage_plan  # noqa: E402


# A stand-in runtime: two modules whose functions nest the way a real bundle's do.
def _fake_runtime():
    session = types.ModuleType("fake_session")
    adapter = types.ModuleType("fake_adapter")

    def forward(work):
        _spin(work)

    def preprocess(work):
        _spin(work)

    def predict(work):
        preprocess(work)
        session.forward(work)

    def pipeline(work):
        adapter.predict(work)
        _spin(work)  # the crop/pad work "assemble" is meant to absorb

    session.forward = forward
    adapter.preprocess = preprocess
    adapter.predict = predict
    adapter.pipeline = pipeline
    sys.modules["fake_session"] = session
    sys.modules["fake_adapter"] = adapter
    return session, adapter


def _spin(seconds):
    import time

    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        pass


_FAKE_TAXONOMY = (
    ("pipeline", "Pipeline", None, "direct"),
    ("model", "Model", "pipeline", "direct"),
    ("model.preprocess", "· preprocess", "model", "direct"),
    ("model.forward", "· forward", "model", "direct"),
    ("model.postprocess", "· postprocess", "model", "direct"),
    ("model.other", "· other", "model", "residual"),
    ("assemble", "In-between", "pipeline", "residual"),
    ("person", "Person detector", "pipeline", "direct"),
    ("unattributed", "Unattributed", None, "residual"),
)

_FAKE_PROBES = (
    Probe("pipeline", "fake_adapter:pipeline"),
    Probe("model", "fake_adapter:predict", role_aware=True),
    Probe("preprocess", "fake_adapter:preprocess", role_aware=True),
    Probe("forward", "fake_session:forward", role_aware=True),
)


class StageArithmeticTests(unittest.TestCase):
    """Siblings plus residual must equal the parent, at every level."""

    def setUp(self):
        _fake_runtime()
        self.plan = StagePlan(_FAKE_PROBES, _FAKE_TAXONOMY, "probe-table")

    def _run(self, calls=3, work=0.004):
        recorder = StageRecorder(self.plan, sync=False, device="cpu")
        import time

        with recorder:
            started = time.perf_counter()
            for _ in range(calls):
                sys.modules["fake_adapter"].pipeline(work)
            total_s = time.perf_counter() - started
        rows = {row["id"]: row for row in recorder.stages(total_s, calls)}
        return recorder, rows, total_s

    def test_children_plus_residual_equal_parent(self):
        _recorder, rows, _total = self._run()
        model = rows["model"]["ms"]
        children = sum(
            rows[key]["ms"] or 0.0
            for key in ("model.preprocess", "model.forward", "model.postprocess", "model.other")
        )
        self.assertIsNotNone(model)
        # places=3: stages() rounds each figure to 4 decimals, so a sum of four
        # rounded children can differ from the rounded parent in the last digit.
        self.assertAlmostEqual(model, children, places=3)

        pipeline = rows["pipeline"]["ms"]
        self.assertAlmostEqual(
            pipeline,
            (rows["model"]["ms"] or 0.0) + (rows["assemble"]["ms"] or 0.0),
            places=3,
        )

    def test_assemble_absorbs_the_work_no_probe_covers(self):
        """The crop/pad step has no function of its own to wrap, so it can only ever
        be a residual — and it must be a *positive* one, not silently zero."""
        _recorder, rows, _total = self._run(work=0.01)
        self.assertEqual(rows["assemble"]["coverage"], "residual")
        self.assertGreater(rows["assemble"]["ms"], 0.0)

    def test_absent_probe_reports_unavailable_not_zero(self):
        """A bundle whose vendored code predates a probe must say so. Reporting 0 ms
        would read as "this stage is free", which is the opposite of the truth."""
        _recorder, rows, _total = self._run()
        self.assertIsNone(rows["person"]["ms"])
        self.assertEqual(rows["person"]["coverage"], "unavailable")

    def test_uninstall_restores_the_runtime(self):
        adapter = sys.modules["fake_adapter"]
        original = adapter.predict
        recorder = StageRecorder(self.plan, sync=False, device="cpu")
        with recorder:
            self.assertIsNot(adapter.predict, original)
        self.assertIs(adapter.predict, original)

    def test_calls_per_frame_is_recorded(self):
        """A tiling pipeline runs the model many times per frame; without this count
        the bare-model comparison reads as a 10x wrapper overhead that isn't real."""
        recorder = StageRecorder(self.plan, sync=False, device="cpu")
        with recorder:
            for _ in range(2):
                sys.modules["fake_adapter"].pipeline(0.001)
                sys.modules["fake_adapter"].predict(0.001)
        rows = {row["id"]: row for row in recorder.stages(1.0, 2)}
        self.assertEqual(rows["model.forward"]["calls_per_frame"], 2.0)


class RoleAwarenessTests(unittest.TestCase):
    """The detector and the trained model share adapter classes, so the same probe
    must file under ``person.*`` inside the detector and ``model.*`` outside it."""

    def setUp(self):
        _fake_runtime()

    def test_nested_person_frame_reassigns_the_role(self):
        adapter = sys.modules["fake_adapter"]

        def detector(work):
            adapter.predict(work)

        detector_module = types.ModuleType("fake_detector")
        detector_module.predict = detector
        sys.modules["fake_detector"] = detector_module

        probes = _FAKE_PROBES + (Probe("person", "fake_detector:predict"),)
        taxonomy = _FAKE_TAXONOMY + (
            ("person.preprocess", "· preprocess", "person", "direct"),
            ("person.forward", "· forward", "person", "direct"),
            ("person.other", "· other", "person", "residual"),
        )
        recorder = StageRecorder(StagePlan(probes, taxonomy, "probe-table"),
                                 sync=False, device="cpu")
        with recorder:
            sys.modules["fake_detector"].predict(0.002)
            adapter.predict(0.002)

        rows = {row["id"]: row for row in recorder.stages(1.0, 1)}
        self.assertIsNotNone(rows["person.forward"]["ms"])
        self.assertIsNotNone(rows["model.forward"]["ms"])
        self.assertEqual(rows["person.forward"]["calls_per_frame"], 1.0)
        self.assertEqual(rows["model.forward"]["calls_per_frame"], 1.0)

    def test_the_detector_is_not_counted_twice(self):
        """The detector runs the same adapter class as the model, so a naive
        role-aware mapping files ``Detector.predict`` and the adapter call it makes
        under the same ``person`` bucket — making the child larger than the pipeline
        that contains it. Observed on a real bundle before this was fixed:
        person 19.7 + model 124.8 against a 135.5 ms pipeline."""
        adapter = sys.modules["fake_adapter"]

        def pipeline_with_detector(work):
            sys.modules["fake_detector"].predict(work)
            adapter.predict(work)

        detector_module = types.ModuleType("fake_detector")
        detector_module.predict = lambda work: adapter.predict(work)
        sys.modules["fake_detector"] = detector_module
        holder = types.ModuleType("fake_holder")
        holder.pipeline = pipeline_with_detector
        sys.modules["fake_holder"] = holder

        probes = (
            Probe("pipeline", "fake_holder:pipeline"),
            Probe("person", "fake_detector:predict"),
            Probe("model", "fake_adapter:predict", role_aware=True),
            Probe("forward", "fake_session:forward", role_aware=True),
        )
        taxonomy = _FAKE_TAXONOMY + (
            ("person.forward", "· forward", "person", "direct"),
            ("person.other", "· other", "person", "residual"),
        )
        recorder = StageRecorder(StagePlan(probes, taxonomy, "probe-table"),
                                 sync=False, device="cpu")
        import time

        with recorder:
            started = time.perf_counter()
            sys.modules["fake_holder"].pipeline(0.004)
            total_s = time.perf_counter() - started

        rows = {row["id"]: row for row in recorder.stages(total_s, 1)}
        self.assertLessEqual(rows["person"]["ms"] + rows["model"]["ms"],
                             rows["pipeline"]["ms"] + 1e-3)
        self.assertGreater(rows["person"]["ms"], 0.0)
        self.assertGreater(rows["model"]["ms"], 0.0)


class StagePlanTests(unittest.TestCase):
    def test_probe_table_is_the_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = load_stage_plan(Path(tmp), {})
        self.assertEqual(plan.source, "probe-table")
        self.assertEqual(plan.probes, bundle_stages.PROBES)

    def test_bundle_declaration_wins(self):
        """This is the whole forward-compatibility story: a newer bundle version
        declares its own stages and neither the harness nor the template changes."""
        manifest = {"benchmark_stages": [
            {"id": "stage1", "label": "Preprocess", "target": "fake_adapter:preprocess"},
            {"id": "stage9", "label": "Something new", "kind": "residual"},
        ]}
        with tempfile.TemporaryDirectory() as tmp:
            plan = load_stage_plan(Path(tmp), manifest)
        self.assertEqual(plan.source, "bundle-declaration")
        self.assertEqual([spec[0] for spec in plan.taxonomy], ["stage1", "stage9"])
        self.assertEqual([probe.stage for probe in plan.probes], ["stage1"])

    def test_declaration_from_a_runtime_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = Path(tmp) / "runtime"
            runtime.mkdir()
            (runtime / "benchmark_stages.py").write_text(
                'BENCHMARK_STAGES = [{"id": "custom", "target": "fake_adapter:preprocess"}]\n'
            )
            plan = load_stage_plan(Path(tmp), {})
        self.assertEqual(plan.source, "bundle-declaration")
        self.assertEqual(plan.taxonomy[0][0], "custom")

    def test_malformed_declaration_falls_back_rather_than_failing(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = load_stage_plan(Path(tmp), {"benchmark_stages": [{"no_id": True}]})
        self.assertEqual(plan.source, "probe-table")


class ConsensusTests(unittest.TestCase):
    def test_host_bound_is_named(self):
        verdict = consensus(20.0, 10.0, 9.0, None)
        self.assertTrue(any("host-bound" in note for note in verdict["notes"]))
        self.assertTrue(verdict["disagree"])

    def test_agreement_is_named(self):
        verdict = consensus(10.2, 10.0, 9.8, 9.9)
        self.assertTrue(any("gpu-bound" in note for note in verdict["notes"]))
        self.assertFalse(verdict["disagree"])

    def test_idle_device_inside_the_call_is_named(self):
        verdict = consensus(10.0, 10.0, 1.0, None)
        self.assertTrue(any("device idle" in note for note in verdict["notes"]))

    def test_bare_model_is_compared_per_model_call(self):
        """9 tiles per frame must not read as a 9x wrapper overhead."""
        verdict = consensus(100.0, 100.0, 95.0, 10.0, model_calls_per_frame=9)
        self.assertTrue(any("adds little" in note for note in verdict["notes"]))
        verdict = consensus(100.0, 100.0, 95.0, 10.0, model_calls_per_frame=1)
        self.assertTrue(any("costs more than the model" in note for note in verdict["notes"]))


class NsightTests(unittest.TestCase):
    """The Nsight pass is opt-in and file-producing, so what matters here is that it
    parses nsys's tables correctly and explains itself when it cannot run."""

    def test_missing_nsys_is_reported_not_raised(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = nsys_pass(
                Path(tmp), Path(tmp), Path(tmp) / "nsys", batch=1, calls=4, warmup=2,
                max_images=2, device=None, fmt=None, conf=None, detector_conf=None,
                nsys_path="/nonexistent/nsys",
            )
        # resolve_nsys falls back to PATH, so this asserts the shape either way.
        self.assertIn("available", result)
        if not result["available"]:
            self.assertTrue(result["reason"])

    def test_resolve_prefers_an_explicit_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "nsys"
            fake.write_text("#!/bin/sh\n")
            fake.chmod(0o755)
            self.assertEqual(resolve_nsys(str(fake)), str(fake))

    def test_kernel_rows_read_every_report_spelling(self):
        """The three nsys reports name their columns differently — kernels use
        Name/Instances, the API report Name/Num Calls, and the memory report
        Operation/Count. Reading only the kernel spelling leaves two tables blank."""
        kernels = _kernel_rows([{
            "Time (%)": "27.0", "Total Time (ns)": "121,034,800", "Instances": "639",
            "Avg (ns)": "189,412", "Med (ns)": "188,000", "Name": "_gemm_mha_v2",
        }], 5)
        self.assertEqual(kernels[0]["instances"], 639)
        self.assertEqual(kernels[0]["name"], "_gemm_mha_v2")
        self.assertEqual(kernels[0]["total_ms"], 121.0348)
        self.assertEqual(kernels[0]["avg_us"], 189.412)

        api = _kernel_rows([{
            "Time (%)": "52.6", "Total Time (ns)": "499,712,000", "Num Calls": "761",
            "Name": "cudaMemcpyAsync",
        }], 5)
        self.assertEqual(api[0]["instances"], 761)
        self.assertEqual(api[0]["name"], "cudaMemcpyAsync")

        memory = _kernel_rows([{
            "Time (%)": "88.8", "Total Time (ns)": "82,175,300", "Count": "92",
            "Operation": "[CUDA memcpy Host-to-Device]",
        }], 5)
        self.assertEqual(memory[0]["instances"], 92)
        self.assertEqual(memory[0]["name"], "[CUDA memcpy Host-to-Device]")

    def test_kernel_rows_sort_by_total_and_truncate_long_names(self):
        rows = _kernel_rows([
            {"Total Time (ns)": "1000000", "Name": "small"},
            {"Total Time (ns)": "9000000", "Name": "T" * 200},
        ], 5)
        self.assertEqual(rows[0]["name"][:5], "TTTTT")
        self.assertEqual(len(rows[0]["name"]), 88)
        self.assertEqual(len(rows[0]["full_name"]), 200)
        self.assertEqual(rows[1]["name"], "small")

    def test_kernel_rows_survive_unparseable_numbers(self):
        rows = _kernel_rows([{"Total Time (ns)": "n/a", "Name": "x",
                             "Instances": "", "Time (%)": None}], 5)
        self.assertIsNone(rows[0]["total_ms"])
        self.assertIsNone(rows[0]["instances"])
        self.assertIsNone(rows[0]["time_pct"])

    def test_top_n_is_a_share_of_the_whole_not_of_itself(self):
        """kernel_total_ms must be summed over every kernel; summing the displayed
        top-N would understate the total the percentages refer to."""
        rows = [{"Total Time (ns)": str(i * 1_000_000), "Name": f"k{i}"}
                for i in range(1, 40)]
        top = _kernel_rows(rows, NSYS_TOP_KERNELS)
        every = _kernel_rows(rows, 10_000)
        self.assertEqual(len(top), NSYS_TOP_KERNELS)
        self.assertLess(sum(r["total_ms"] for r in top),
                        sum(r["total_ms"] for r in every))


class InputTests(unittest.TestCase):
    def test_images_are_sorted_then_truncated(self):
        """Two runs on the same directory must see the same frames in the same order,
        or their numbers are not comparable."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("c.jpg", "a.jpg", "b.png", "notes.txt"):
                (root / name).write_bytes(b"x")
            self.assertEqual([p.name for p in list_images(root, 2)], ["a.jpg", "b.png"])
            self.assertEqual(len(list_images(root, None)), 3)

    def test_empty_directory_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                list_images(Path(tmp), None)

    def test_bare_input_shape_uses_the_declared_size_not_one(self):
        """Filling a detector graph's dynamic spatial dims with 1 is the documented
        way to make it die inside its own backbone."""
        with tempfile.TemporaryDirectory() as tmp:
            onnx_path = Path(tmp) / "model.onnx"
            onnx_path.write_bytes(b"x")
            onnx_path.with_suffix(".meta.json").write_text(
                json.dumps({"input": {"size": 640}}))
            self.assertEqual(
                _bare_input_shape(["batch", 3, "height", "width"], onnx_path),
                [1, 3, 640, 640],
            )

    def test_bare_input_shape_without_meta(self):
        with tempfile.TemporaryDirectory() as tmp:
            onnx_path = Path(tmp) / "model.onnx"
            onnx_path.write_bytes(b"x")
            self.assertEqual(
                _bare_input_shape([1, 3, "h", "w"], onnx_path), [1, 3, 640, 640])

    def test_parse_int_list(self):
        self.assertEqual(_parse_int_list("1, 2,4", "batch-sizes"), [1, 2, 4])
        for bad in ("", "0", "-1", "two"):
            with self.assertRaises(SystemExit):
                _parse_int_list(bad, "batch-sizes")


if __name__ == "__main__":
    unittest.main()
