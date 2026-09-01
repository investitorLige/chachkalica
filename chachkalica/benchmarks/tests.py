"""Tests for the bundle benchmark admin section.

Reuses ``training.tests_bundles.BundleTestCase``, which points ``bundles_root`` at a
tmpdir and can write a real bundle tree into it — the benchmark's inputs are
filesystem objects, so a fixture that produces them is worth more than a mock.

The trainer is always stubbed: these tests are about the row, the form, the job's
bookkeeping and the rendered pages. What the harness itself measures is pinned in
``inferlica/benchmark/tests/test_bundle_bench.py``.
"""

import json
from pathlib import Path
from unittest import mock

from django.contrib.auth import get_user_model
from django.test import Client
from django.urls import reverse

from benchmarks import admin as benchmarks_admin
from benchmarks import jobs
from benchmarks.models import BundleBenchmark
from fleet.models import Dataset, FleetSettings
from training.models import TrainingSettings
from training.tests_bundles import BundleTestCase


def _result(*, p50=12.5, fps=80.0, mem=512.0, contended=False, cells=None,
            warnings=None, nsys=False):
    """A harness result envelope, shaped like the real thing."""
    return {
        "schema": 1,
        "generated": "2026-08-26T10:00:00",
        "duration_s": 91.2,
        "bundle": {"name": "ppe", "fmt": "engine", "pipeline": "people_detect_first",
                   "classes": {"0": "helmet"}, "max_batch": 4, "size_mb": 120.0,
                   "artifacts": []},
        "env": {"device": "cuda:0", "gpu": "RTX 4080 SUPER", "torch": "2.13.0"},
        "config": {"image_count": 24, "warmup": 30, "calls": 200,
                   "batch_sizes": [1], "concurrency": [1], "images": "/app/data/x"},
        "trust": {"contended": contended, "cotenants": []},
        "bare_model": {"ms": 4.2, "tool": "polygraphy", "error": None},
        "cells": cells if cells is not None else [{
            "batch": 1, "concurrency": 1,
            "latency_ms": {"p50": p50, "p90": p50 * 1.1, "p99": p50 * 1.2,
                           "mean": p50, "std": 0.4, "min": p50 * 0.9,
                           "max": p50 * 1.3, "samples": 200},
            "latency_source": "cuda_event",
            "fps": fps, "fps_per_stream": fps, "frames": 200,
            "gpu_busy_pct": 71.4, "util_method": "cupti",
            "mem": {"load_delta_mb": mem, "torch_peak_mb": 8.0,
                    "nvml_peak_mb": 900.0, "host_rss_peak_mb": 1100.0},
            "power": {"mean_w": 160.0, "peak_w": 190.0, "mean_sm_clock_mhz": 2700.0,
                      "throttle_reasons": []},
            "coldstart": {"build_runtime_ms": 1500.0, "detector_load_ms": 700.0,
                          "model_load_ms": 800.0, "first_call_ms": 2400.0},
            "instruments": {"t1_wall_ms": p50 + 0.1, "t2_cuda_event_ms": p50,
                            "t3_cupti_device_ms": p50 * 0.9, "t4_bare_model_ms": 4.2,
                            "spread_pct": 9.0, "disagree": False,
                            "notes": ["gpu-bound: wall and device latency agree"]},
            "model_calls_per_frame": 2.0,
            "stage_pass_total_ms": p50 + 3.0,
            "instrumentation_overhead_ms": 3.0,
            "stages": [
                {"id": "pipeline", "label": "Pipeline", "parent": None, "depth": 0,
                 "ms": p50, "pct": 100.0, "calls_per_frame": 1.0, "coverage": "direct"},
                {"id": "person", "label": "Person detector", "parent": "pipeline",
                 "depth": 1, "ms": 4.0, "pct": 32.0, "calls_per_frame": 1.0,
                 "coverage": "direct"},
                {"id": "model", "label": "Model engine", "parent": "pipeline",
                 "depth": 1, "ms": 7.0, "pct": 56.0, "calls_per_frame": 2.0,
                 "coverage": "direct"},
                {"id": "assemble", "label": "In-between", "parent": "pipeline",
                 "depth": 1, "ms": 1.5, "pct": 12.0, "calls_per_frame": None,
                 "coverage": "residual"},
                {"id": "remap", "label": "Class remap", "parent": None, "depth": 0,
                 "ms": None, "pct": None, "calls_per_frame": None,
                 "coverage": "unavailable"},
            ],
            "stage_diagnostics": {"plan_source": "probe-table", "synchronized": True,
                                  "probes_resolved": ["model"], "probes_absent": []},
            "detections_mean": 3.5,
            "cupti": {"available": True},
            "warnings": [],
        }],
        "nsys": {
            "available": True,
            "version": "NVIDIA Nsight Systems version 2025.5.2",
            "report_file": "profile.nsys-rep",
            "files": ["profile.nsys-rep", "cuda_gpu_kern_sum.csv"],
            "directory": "/app/data/training/runs/benchmarks/1/nsys",
            "profiled_cell": {"batch": 1, "calls": 10, "warmup": 3},
            "kernel_count": 146,
            "kernel_total_ms": 447.78,
            "kernels": [{"name": "_gemm_mha_v2", "full_name": "_gemm_mha_v2",
                         "time_pct": 27.0, "total_ms": 121.03, "instances": 639,
                         "avg_us": 189.4, "med_us": 188.0}],
            "api": [{"name": "cudaMemcpyAsync", "full_name": "cudaMemcpyAsync",
                     "time_pct": 52.6, "total_ms": 499.7, "instances": 761,
                     "avg_us": 656.6, "med_us": 11.4}],
            "memory": [{"name": "[CUDA memcpy Host-to-Device]",
                        "full_name": "[CUDA memcpy Host-to-Device]",
                        "time_pct": 88.8, "total_ms": 82.17, "instances": 92,
                        "avg_us": 893.2, "med_us": 757.5}],
            "counters": "not requested: GPU performance counters need CAP_SYS_ADMIN",
        } if nsys else {"available": False, "reason": "not requested"},
        "warnings": warnings or [],
    }


class BenchmarkAdminTestCase(BundleTestCase):
    def setUp(self):
        super().setUp()
        self.bundle = self.write_bundle()
        User = get_user_model()
        self.user = User.objects.create_superuser("bench", "b@x.io", "pw")
        self.client = Client()
        self.client.force_login(self.user)

    def _add(self, **extra):
        payload = {
            "bundle_path": "ppe-bundle",
            "images_path": str(self.root),
            "max_images": "24",
            "batch_sizes": "1",
            "concurrency": "1",
            "warmup": "30",
            "calls": "200",
            "min_duration_s": "2.0",
            "measure_stages": "on",
        }
        payload.update(extra)
        return self.client.post("/admin/benchmarks/bundlebenchmark/add/", payload)


class FormTests(BenchmarkAdminTestCase):
    def test_bundle_choices_come_from_the_filesystem_scan(self):
        """Bundles have no DB rows, so the dropdown is a scan of the bundle root."""
        form = benchmarks_admin.BundleBenchmarkForm()
        values = [value for value, _label in form.fields["bundle_path"].choices]
        self.assertEqual(values, ["ppe-bundle"])
        self.assertIn("ppe", dict(form.fields["bundle_path"].choices)["ppe-bundle"])

    def test_launch_records_the_name_and_format_from_the_manifest(self):
        with mock.patch.object(benchmarks_admin, "_queue"):
            resp = self._add()
        self.assertEqual(resp.status_code, 302)
        row = BundleBenchmark.objects.get()
        self.assertEqual(row.bundle_name, "ppe")
        self.assertEqual(row.fmt, "engine")
        self.assertEqual(row.status, BundleBenchmark.QUEUED)

    def test_no_image_source_is_rejected(self):
        """A bundle benchmark runs on real frames; without them there is nothing to
        measure, so this is caught at the form rather than as a failed job."""
        with mock.patch.object(benchmarks_admin, "_queue"):
            resp = self._add(images_path="")
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "Pick a dataset or give an explicit images path")
        self.assertFalse(BundleBenchmark.objects.exists())

    def test_bad_sweep_lists_are_rejected(self):
        for field, value in (("batch_sizes", "1,two"), ("concurrency", "0"),
                             ("batch_sizes", "")):
            with mock.patch.object(benchmarks_admin, "_queue"):
                resp = self._add(**{field: value})
            self.assertEqual(resp.status_code, 200, f"{field}={value!r}")
            self.assertFalse(BundleBenchmark.objects.exists())

    def test_a_broken_bundle_is_rejected_before_the_job(self):
        self.write_bundle(name="broken-bundle", model=False)
        with mock.patch.object(benchmarks_admin, "_queue"):
            resp = self._add(bundle_path="broken-bundle")
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "will not run")


class LaunchTests(BenchmarkAdminTestCase):
    def setUp(self):
        super().setUp()
        ts = TrainingSettings.load()
        ts.runs_root = str(self.root / "runs")
        ts.save()

    def test_saving_the_add_form_enqueues_the_job(self):
        with mock.patch.object(benchmarks_admin, "_queue") as queue:
            self._add()
        queue.return_value.enqueue.assert_called_once()
        args, kwargs = queue.return_value.enqueue.call_args
        self.assertEqual(args[0], jobs.run_bundle_benchmark)
        self.assertEqual(kwargs.get("job_timeout"), jobs.JOB_TIMEOUT)
        self.assertEqual(args[1], BundleBenchmark.objects.get().pk)

    def test_output_dir_is_set_at_launch(self):
        with mock.patch.object(benchmarks_admin, "_queue"):
            self._add()
        row = BundleBenchmark.objects.get()
        self.assertTrue(row.output_dir.endswith(f"benchmarks/{row.pk}"))

    def test_rerun_clones_rather_than_resetting(self):
        """Keeping the old row is the point: a history is what shows a bundle got
        slower."""
        original = BundleBenchmark.objects.create(
            bundle_path="ppe-bundle", bundle_name="ppe", fmt="engine",
            images_path=str(self.root), batch_sizes="1,2", concurrency="1",
            status=BundleBenchmark.OK, results=_result())
        with mock.patch.object(benchmarks_admin, "_queue") as queue:
            resp = self.client.post("/admin/benchmarks/bundlebenchmark/", {
                "action": "rerun_selected",
                "_selected_action": [str(original.pk)],
                "index": "0",
            })
        self.assertEqual(resp.status_code, 302)
        self.assertEqual(BundleBenchmark.objects.count(), 2)
        clone = BundleBenchmark.objects.exclude(pk=original.pk).get()
        self.assertEqual(clone.batch_sizes, "1,2")
        self.assertEqual(clone.status, BundleBenchmark.QUEUED)
        self.assertIsNone(clone.results)
        original.refresh_from_db()
        self.assertEqual(original.status, BundleBenchmark.OK)
        queue.return_value.enqueue.assert_called_once()

    def test_a_finished_row_cannot_be_edited(self):
        """A row whose parameters were changed after the run would describe a
        benchmark that never happened."""
        row = BundleBenchmark.objects.create(
            bundle_path="ppe-bundle", images_path=str(self.root),
            status=BundleBenchmark.OK, results=_result())
        resp = self.client.get(f"/admin/benchmarks/bundlebenchmark/{row.pk}/change/")
        self.assertEqual(resp.status_code, 200)
        self.assertNotContains(resp, 'name="batch_sizes"')


class PathTests(BundleTestCase):
    def test_dataset_supplies_the_frames_when_no_path_is_given(self):
        source = self.root / "src"
        (source / "ds1" / "images").mkdir(parents=True)
        (source / "ds1" / "images" / "a.jpg").write_bytes(b"x")
        fs = FleetSettings.load()
        fs.source_dir = str(source)
        fs.save()
        dataset = Dataset.objects.create(name="ds1")

        row = BundleBenchmark(bundle_path="ppe-bundle", dataset=dataset)
        self.assertEqual(row.resolved_images_path(), str(source / "ds1" / "images"))

    def test_an_explicit_path_wins_over_the_dataset(self):
        dataset = Dataset.objects.create(name="ds1")
        row = BundleBenchmark(bundle_path="ppe-bundle", dataset=dataset,
                              images_path="/somewhere/else")
        self.assertEqual(row.resolved_images_path(), "/somewhere/else")

    def test_neither_is_an_error_not_an_empty_path(self):
        """A re-run of a row whose dataset was since deleted must not reach the
        trainer with an empty images argument."""
        row = BundleBenchmark(bundle_path="ppe-bundle")
        with self.assertRaises(ValueError):
            row.resolved_images_path()

    def test_a_bundle_path_escaping_the_root_is_refused(self):
        from training.services.bundles import BundleError

        row = BundleBenchmark(bundle_path="../outside")
        with self.assertRaises(BundleError):
            row.absolute_bundle_dir()


class IngestTests(BundleTestCase):
    def test_headline_numbers_are_lifted_onto_the_row(self):
        row = BundleBenchmark(bundle_path="ppe-bundle")
        row.ingest(_result(p50=12.5, fps=80.0, mem=512.0))
        self.assertEqual(row.p50_ms, 12.5)
        self.assertEqual(row.fps, 80.0)
        self.assertEqual(row.gpu_mem_mb, 512.0)
        self.assertFalse(row.contended)
        self.assertEqual(row.bundle_name, "ppe")
        self.assertEqual(row.fmt, "engine")

    def test_contention_is_carried_onto_the_row(self):
        """A contended run is not a failure, but it must be visible without opening
        the report."""
        row = BundleBenchmark(bundle_path="ppe-bundle")
        row.ingest(_result(contended=True))
        self.assertTrue(row.contended)

    def test_a_failed_first_cell_is_skipped_for_the_headline(self):
        result = _result()
        result["cells"] = [{"batch": 1, "concurrency": 1, "error": "boom"},
                           *result["cells"]]
        row = BundleBenchmark(bundle_path="ppe-bundle")
        row.ingest(result)
        self.assertEqual(row.p50_ms, 12.5)

    def test_no_usable_cell_leaves_the_headline_empty(self):
        result = _result(cells=[{"batch": 1, "concurrency": 1, "error": "boom"}])
        row = BundleBenchmark(bundle_path="ppe-bundle")
        row.ingest(result)
        self.assertIsNone(row.p50_ms)
        self.assertIsNone(row.fps)


class JobTests(BundleTestCase):
    def setUp(self):
        super().setUp()
        # BundleTestCase redirects only bundles_root. runs_root has to be redirected
        # too, or resolved_output_dir() points at the real shared data mount and these
        # tests write directories into it.
        ts = TrainingSettings.load()
        ts.runs_root = str(self.root / "runs")
        ts.save()
        self.row = BundleBenchmark.objects.create(
            bundle_path="ppe-bundle", images_path=str(self.root))

    def _write_result(self, payload):
        out = Path(self.row.resolved_output_dir())
        out.mkdir(parents=True, exist_ok=True)
        (out / jobs.RESULT_NAME).write_text(json.dumps(payload))

    def test_a_finished_run_is_ingested_and_marked_ok(self):
        self._write_result(_result())
        with mock.patch.object(jobs.runner, "launch_benchmark") as launch, \
             mock.patch.object(jobs.runner, "fetch_benchmark_status",
                               return_value={"status": "ok"}), \
             mock.patch.object(jobs.time, "sleep"):
            summary = jobs.run_bundle_benchmark(self.row.pk)
        launch.assert_called_once()
        self.row.refresh_from_db()
        self.assertEqual(self.row.status, BundleBenchmark.OK)
        self.assertEqual(self.row.p50_ms, 12.5)
        self.assertIsNotNone(self.row.started_at)
        self.assertIsNotNone(self.row.finished_at)
        self.assertEqual(summary["cells"], 1)

    def test_a_launch_failure_is_recorded_and_reraised(self):
        with mock.patch.object(jobs.runner, "launch_benchmark",
                               side_effect=RuntimeError("trainer busy")):
            with self.assertRaises(RuntimeError):
                jobs.run_bundle_benchmark(self.row.pk)
        self.row.refresh_from_db()
        self.assertEqual(self.row.status, BundleBenchmark.ERROR)
        self.assertIn("trainer busy", self.row.last_error)

    def test_a_run_that_wrote_no_result_fails_with_the_trainer_log(self):
        with mock.patch.object(jobs.runner, "launch_benchmark"), \
             mock.patch.object(jobs.runner, "fetch_benchmark_status",
                               return_value={"status": "error",
                                             "log_tail": "Traceback: boom"}), \
             mock.patch.object(jobs.time, "sleep"):
            with self.assertRaises(RuntimeError):
                jobs.run_bundle_benchmark(self.row.pk)
        self.row.refresh_from_db()
        self.assertEqual(self.row.status, BundleBenchmark.ERROR)
        self.assertIn("wrote no benchmark.json", self.row.last_error)
        self.assertIn("Traceback: boom", self.row.last_error)

    def test_a_restarted_trainer_still_ingests_a_written_result(self):
        """The subprocess dies with the service, but a result already on disk is not
        worth throwing away."""
        self._write_result(_result())
        with mock.patch.object(jobs.runner, "launch_benchmark"), \
             mock.patch.object(jobs.runner, "fetch_benchmark_status",
                               return_value={"status": "unknown"}), \
             mock.patch.object(jobs.time, "sleep"):
            jobs.run_bundle_benchmark(self.row.pk)
        self.row.refresh_from_db()
        self.assertEqual(self.row.status, BundleBenchmark.OK)

    def test_a_restarted_trainer_with_no_result_fails_rather_than_hanging(self):
        with mock.patch.object(jobs.runner, "launch_benchmark"), \
             mock.patch.object(jobs.runner, "fetch_benchmark_status",
                               return_value={"status": "unknown"}), \
             mock.patch.object(jobs.time, "sleep"):
            with self.assertRaises(RuntimeError):
                jobs.run_bundle_benchmark(self.row.pk)
        self.row.refresh_from_db()
        self.assertEqual(self.row.status, BundleBenchmark.ERROR)
        self.assertIn("lost track", self.row.last_error)


class PageTests(BenchmarkAdminTestCase):
    def setUp(self):
        super().setUp()
        ts = TrainingSettings.load()
        ts.runs_root = str(self.root / "runs")
        ts.save()

    def _row(self, **kwargs):
        defaults = dict(bundle_path="ppe-bundle", bundle_name="ppe", fmt="engine",
                        images_path=str(self.root), status=BundleBenchmark.OK)
        defaults.update(kwargs)
        row = BundleBenchmark.objects.create(**defaults)
        if row.results is None:
            row.results = _result()
            row.save(update_fields=["results"])
        return row

    def test_the_report_page_renders_the_stored_result(self):
        row = self._row()
        url = reverse("admin:benchmarks_bundlebenchmark_report")
        resp = self.client.get(url, {"run": row.pk})
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "Four instruments")
        self.assertContains(resp, "Where the frame budget goes")
        self.assertContains(resp, "bb-data")
        # The result travels to the page as JSON, so the stage set is whatever the
        # harness reported rather than a list baked into the template.
        self.assertContains(resp, "Person detector")

    def test_the_report_page_survives_a_missing_run(self):
        url = reverse("admin:benchmarks_bundlebenchmark_report")
        resp = self.client.get(url, {"run": "99999"})
        self.assertEqual(resp.status_code, 302)

    def test_the_changelist_shows_the_headline_and_the_caveats(self):
        self._row(p50_ms=12.5, fps=80.0, contended=True)
        resp = self.client.get("/admin/benchmarks/bundlebenchmark/")
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "12.50 ms")
        self.assertContains(resp, "contended")

    def test_compare_needs_two_finished_runs(self):
        row = self._row()
        resp = self.client.post("/admin/benchmarks/bundlebenchmark/", {
            "action": "compare_selected",
            "_selected_action": [str(row.pk)],
            "index": "0",
        })
        self.assertEqual(resp.status_code, 302)
        self.assertIn("/admin/benchmarks/bundlebenchmark/", resp["Location"])

    def test_compare_renders_two_runs_with_deltas(self):
        first = self._row(results=_result(p50=10.0, fps=100.0))
        second = self._row(results=_result(p50=20.0, fps=50.0))
        resp = self.client.post("/admin/benchmarks/bundlebenchmark/", {
            "action": "compare_selected",
            "_selected_action": [str(first.pk), str(second.pk)],
            "index": "0",
        })
        self.assertEqual(resp.status_code, 302)
        page = self.client.get(resp["Location"])
        self.assertEqual(page.status_code, 200)
        self.assertContains(page, "Stage shares")
        self.assertContains(page, "bc-data")

    def test_the_report_renders_the_kernel_profile_when_present(self):
        row = self._row(results=_result(nsys=True))
        url = reverse("admin:benchmarks_bundlebenchmark_report")
        resp = self.client.get(url, {"run": row.pk})
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "Kernel profile (Nsight Systems)")
        self.assertContains(resp, "_gemm_mha_v2")
        self.assertContains(resp, "cudaMemcpyAsync")

    def test_the_download_link_is_absent_without_a_report_on_disk(self):
        """The result JSON can say a profile was captured while the file is gone (a
        pruned output dir). The link must follow the file, not the JSON."""
        row = self._row(results=_result(nsys=True))
        url = reverse("admin:benchmarks_bundlebenchmark_report")
        resp = self.client.get(url, {"run": row.pk})
        self.assertNotContains(resp, "benchmarks_bundlebenchmark_nsys")
        self.assertEqual(resp.context["nsys_url"], "")

    def test_downloading_serves_the_captured_report(self):
        row = self._row(results=_result(nsys=True))
        nsys_dir = Path(row.resolved_output_dir()) / "nsys"
        nsys_dir.mkdir(parents=True, exist_ok=True)
        (nsys_dir / "profile.nsys-rep").write_bytes(b"nsys-rep-bytes")
        row.output_dir = row.resolved_output_dir()
        row.save(update_fields=["output_dir"])

        url = reverse("admin:benchmarks_bundlebenchmark_nsys")
        resp = self.client.get(url, {"run": row.pk})
        self.assertEqual(resp.status_code, 200)
        self.assertIn("attachment", resp["Content-Disposition"])
        self.assertEqual(b"".join(resp.streaming_content), b"nsys-rep-bytes")

    def test_downloading_a_run_without_a_profile_redirects(self):
        row = self._row()
        url = reverse("admin:benchmarks_bundlebenchmark_nsys")
        resp = self.client.get(url, {"run": row.pk})
        self.assertEqual(resp.status_code, 302)

    def test_the_report_path_is_confined_to_the_runs_own_output_dir(self):
        """The request carries a row id, never a path, so a crafted report_file cannot
        make the download view serve something outside this run's directory."""
        row = self._row(results=_result(nsys=True))
        row.results["nsys"]["report_file"] = "../../../../etc/passwd"
        row.output_dir = row.resolved_output_dir()
        row.save(update_fields=["results", "output_dir"])
        self.assertIsNone(row.nsys_report_path())

    def test_the_nav_names_the_arch_console_for_what_it_measures(self):
        """Two different things are called "benchmark" in this admin. The header keeps
        only one button, relabelled: the arch console compares architecture variants on
        synthetic tensors, and this section is reached from the index sidebar instead."""
        resp = self.client.get("/admin/benchmarks/bundlebenchmark/")
        self.assertContains(resp, "Arch console")
        self.assertNotContains(resp, ">\n  Benchmarks\n</a>")
