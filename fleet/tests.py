import tempfile
from pathlib import Path
from unittest import mock

from django.test import TestCase

from fleet.models import Annotator, Dataset, FleetSettings, Project
from fleet.reconcile import txt_format
from fleet.services import analytics as analytics_svc
from fleet.services import lsapi
from fleet.services import merge as merge_svc

_PPE_NAMES = [
    "gloves", "goggles", "helmet", "no_gloves",
    "no_goggles", "no_helmet", "no_vest", "vest",
]


def _make_dataset(
    src: Path,
    name: str,
    classes_header: str,
    classes: list[str],
    images: list[str],
    labels: dict[str, str] | None = None,
) -> Dataset:
    ds_dir = src / name
    ds_dir.mkdir(parents=True)
    (ds_dir / "classes.txt").write_text(classes_header + "\n".join(classes) + "\n", encoding="utf-8")
    for img in images:
        (ds_dir / img).write_bytes(b"fake-jpeg")
    if labels:
        labels_dir = ds_dir / "labels"
        labels_dir.mkdir()
        for label_name, content in labels.items():
            (labels_dir / label_name).write_text(content, encoding="utf-8")
    return Dataset.objects.create(name=name, storage_type=Dataset.LOCAL)


class MergeDatasetsTests(TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.src = Path(self.tmp.name)
        fs = FleetSettings.load()
        fs.source_dir = str(self.src)  # absolute -> used verbatim by source_root()
        fs.save()

    def tearDown(self):
        self.tmp.cleanup()

    def test_merge_keeps_intersection_and_copies_prefixed_images(self):
        a = _make_dataset(
            self.src, "alpha", "# tools: bbox\n",
            ["person", "helmet", "vest"], ["a1.jpg", "a2.png"],
        )
        b = _make_dataset(
            self.src, "beta", "# tools: bbox, polygon\n",
            ["vest", "person", "dog"], ["b1.jpg"],
        )

        result = merge_svc.merge_datasets([a, b], "merged")

        # Intersection by name, ordered by the first dataset's order.
        self.assertEqual(result["kept"], ["person", "vest"])
        self.assertCountEqual(result["dropped"], ["helmet", "dog"])
        self.assertEqual(result["images"], 3)

        merged_dir = self.src / "merged"
        classes = (merged_dir / "classes.txt").read_text(encoding="utf-8")
        self.assertEqual(classes, "# tools: bbox\nperson\nvest\n")

        copied = sorted(p.name for p in (merged_dir / "images").iterdir() if p.suffix in {".jpg", ".png"})
        self.assertEqual(copied, ["alpha__a1.jpg", "alpha__a2.png", "beta__b1.jpg"])

        self.assertTrue(Dataset.objects.filter(name="merged").exists())

    def test_empty_intersection_raises_and_leaves_no_dir(self):
        a = _make_dataset(self.src, "alpha", "", ["cat"], ["a1.jpg"])
        b = _make_dataset(self.src, "beta", "", ["dog"], ["b1.jpg"])
        with self.assertRaises(RuntimeError):
            merge_svc.merge_datasets([a, b], "merged")
        self.assertFalse((self.src / "merged").exists())
        self.assertFalse(Dataset.objects.filter(name="merged").exists())

    def test_empty_tool_intersection_raises_and_leaves_no_dir(self):
        a = _make_dataset(self.src, "alpha", "# tools: bbox\n", ["cat"], ["a1.jpg"])
        b = _make_dataset(self.src, "beta", "# tools: polygon\n", ["cat"], ["b1.jpg"])
        with self.assertRaises(RuntimeError):
            merge_svc.merge_datasets([a, b], "merged")
        self.assertFalse((self.src / "merged").exists())
        self.assertFalse(Dataset.objects.filter(name="merged").exists())

    def test_tool_aliases_count_as_common_tools(self):
        a = _make_dataset(self.src, "alpha", "# tools: bbox\n", ["cat"], ["a1.jpg"])
        b = _make_dataset(self.src, "beta", "# tools: rectangle\n", ["cat"], ["b1.jpg"])

        merge_svc.merge_datasets([a, b], "merged")

        classes = (self.src / "merged" / "classes.txt").read_text(encoding="utf-8")
        self.assertEqual(classes, "# tools: bbox\ncat\n")

    def test_cloud_dataset_aborts(self):
        a = _make_dataset(self.src, "alpha", "", ["cat", "dog"], ["a1.jpg"])
        b = _make_dataset(self.src, "beta", "", ["cat", "dog"], ["b1.jpg"])
        b.storage_type = Dataset.CLOUD
        b.save()
        with self.assertRaises(RuntimeError):
            merge_svc.merge_datasets([a, b], "merged")

    def test_duplicate_name_aborts(self):
        a = _make_dataset(self.src, "alpha", "", ["cat", "dog"], ["a1.jpg"])
        b = _make_dataset(self.src, "beta", "", ["cat", "dog"], ["b1.jpg"])
        Dataset.objects.create(name="taken", storage_type=Dataset.LOCAL)
        with self.assertRaises(RuntimeError):
            merge_svc.merge_datasets([a, b], "taken")

    def test_merge_reads_images_from_images_subdir(self):
        # alpha uses the YOLO layout (images/ subdir); beta uses the flat layout.
        a = _make_dataset(self.src, "alpha", "", ["cat", "dog"], [])
        imgs = self.src / "alpha" / "images"
        imgs.mkdir()
        (imgs / "a1.jpg").write_bytes(b"fake-jpeg")
        (imgs / "a2.jpg").write_bytes(b"fake-jpeg")
        b = _make_dataset(self.src, "beta", "", ["cat", "dog"], ["b1.jpg"])

        result = merge_svc.merge_datasets([a, b], "merged")

        self.assertEqual(result["images"], 3)
        copied = sorted(p.name for p in (self.src / "merged" / "images").iterdir() if p.suffix == ".jpg")
        self.assertEqual(copied, ["alpha__a1.jpg", "alpha__a2.jpg", "beta__b1.jpg"])

    def test_merge_carries_and_remaps_labels(self):
        # alpha order: [person, helmet, vest]; beta order: [vest, person, dog].
        # Intersection kept order (from alpha): [person, vest] -> new idx person=0, vest=1.
        a = _make_dataset(
            self.src, "alpha", "# tools: bbox\n",
            ["person", "helmet", "vest"], ["a1.jpg"],
            labels={"a1.txt": "0 0.1 0.1 0.2 0.2\n1 0.3 0.3 0.2 0.2\n2 0.5 0.5 0.2 0.2\n"},
        )
        b = _make_dataset(
            self.src, "beta", "# tools: bbox\n",
            ["vest", "person", "dog"], ["b1.jpg"],
            labels={"b1.txt": "0 0.4 0.4 0.1 0.1\n1 0.6 0.6 0.1 0.1\n2 0.7 0.7 0.1 0.1\n"},
        )

        result = merge_svc.merge_datasets([a, b], "merged")

        self.assertEqual(result["kept"], ["person", "vest"])
        self.assertEqual(result["labels"], 2)

        labels_dir = self.src / "merged" / "labels"
        # alpha: person(0->0) kept, helmet(1) dropped, vest(2->1) kept.
        self.assertEqual(
            (labels_dir / "alpha__a1.txt").read_text(encoding="utf-8"),
            "0 0.1 0.1 0.2 0.2\n1 0.5 0.5 0.2 0.2\n",
        )
        # beta: vest(0->1) kept, person(1->0) kept, dog(2) dropped.
        self.assertEqual(
            (labels_dir / "beta__b1.txt").read_text(encoding="utf-8"),
            "1 0.4 0.4 0.1 0.1\n0 0.6 0.6 0.1 0.1\n",
        )
        self.assertTrue(Dataset.objects.get(name="merged").has_labels)

    def test_merge_mixes_labeled_and_unlabeled(self):
        a = _make_dataset(
            self.src, "alpha", "", ["cat", "dog"], ["a1.jpg"],
            labels={"a1.txt": "0 0.1 0.1 0.2 0.2\n"},
        )
        b = _make_dataset(self.src, "beta", "", ["cat", "dog"], ["b1.jpg"])  # no labels/

        result = merge_svc.merge_datasets([a, b], "merged")

        self.assertEqual(result["images"], 2)
        self.assertEqual(result["labels"], 1)
        labels = sorted(p.name for p in (self.src / "merged" / "labels").iterdir())
        self.assertEqual(labels, ["alpha__a1.txt"])
        self.assertTrue(Dataset.objects.get(name="merged").has_labels)

    def test_image_with_only_dropped_classes_gets_no_label_file(self):
        # alpha's a1 is annotated solely with `helmet`, which is dropped.
        a = _make_dataset(
            self.src, "alpha", "", ["person", "helmet"], ["a1.jpg"],
            labels={"a1.txt": "1 0.3 0.3 0.2 0.2\n"},
        )
        b = _make_dataset(self.src, "beta", "", ["person", "dog"], ["b1.jpg"])

        result = merge_svc.merge_datasets([a, b], "merged")

        self.assertEqual(result["kept"], ["person"])
        self.assertEqual(result["labels"], 0)
        self.assertFalse((self.src / "merged" / "labels" / "alpha__a1.txt").exists())
        self.assertFalse(Dataset.objects.get(name="merged").has_labels)

    def test_merge_preserves_app_format_header(self):
        a = _make_dataset(
            self.src, "alpha", "", ["person", "helmet"], ["a1.jpg"],
            labels={"a1.txt": "640 480\n0 0.1 0.1 0.2 0.2\n1 0.3 0.3 0.2 0.2\n"},
        )
        b = _make_dataset(self.src, "beta", "", ["person", "dog"], ["b1.jpg"])

        merge_svc.merge_datasets([a, b], "merged")

        # helmet(1) dropped; person(0->0) kept; "640 480" header preserved.
        self.assertEqual(
            (self.src / "merged" / "labels" / "alpha__a1.txt").read_text(encoding="utf-8"),
            "640 480\n0 0.1 0.1 0.2 0.2\n",
        )


class DatasetAnalyticsTests(TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.src = Path(self.tmp.name)
        fs = FleetSettings.load()
        fs.source_dir = str(self.src)
        fs.save()

    def tearDown(self):
        self.tmp.cleanup()

    def test_counts_regions_images_and_distributions(self):
        # img1: two cats + one dog; img2: one cat; img3: image with no label file.
        ds = _make_dataset(
            self.src, "zoo", "", ["cat", "dog", "bird"],
            ["img1.jpg", "img2.jpg", "img3.jpg"],
            labels={
                "img1.txt": "0 0.1 0.1 0.2 0.2\n0 0.3 0.3 0.2 0.2\n1 0.5 0.5 0.2 0.2\n",
                "img2.txt": "0 0.4 0.4 0.1 0.1\n",
            },
        )

        report = analytics_svc.analyze_dataset(ds)
        s = report["summary"]

        self.assertEqual(s["image_count"], 3)
        self.assertEqual(s["labeled_images"], 2)
        self.assertEqual(s["unlabeled_images"], 1)
        self.assertEqual(s["labeled_pct"], 66.7)
        self.assertEqual(s["total_regions"], 4)  # 3 in img1 + 1 in img2
        self.assertEqual(s["class_count"], 3)
        self.assertEqual(s["unused_classes"], ["bird"])

        by_name = {r["name"]: r for r in report["rows"]}
        # cat: 3 of 4 regions; present in 2 of 3 class-image occurrences.
        self.assertEqual(by_name["cat"]["region_count"], 3)
        self.assertEqual(by_name["cat"]["region_pct"], 75.0)
        self.assertEqual(by_name["cat"]["image_count"], 2)  # both labeled images
        # dog: 1 region, present in 1 image. image_pct over presence total (2+1=3).
        self.assertEqual(by_name["dog"]["region_count"], 1)
        self.assertEqual(by_name["dog"]["image_count"], 1)
        self.assertEqual(by_name["dog"]["image_pct"], 33.3)

    def test_multiple_same_class_in_image_counts_once_per_image(self):
        ds = _make_dataset(
            self.src, "many", "", ["cat"], ["a.jpg"],
            labels={"a.txt": "0 0.1 0.1 0.1 0.1\n0 0.2 0.2 0.1 0.1\n0 0.3 0.3 0.1 0.1\n"},
        )
        report = analytics_svc.analyze_dataset(ds)
        cat = report["rows"][0]
        self.assertEqual(cat["region_count"], 3)
        self.assertEqual(cat["image_count"], 1)

    def test_bbox_and_polygon_split(self):
        ds = _make_dataset(
            self.src, "shapes", "", ["thing"], ["a.jpg"],
            # a box, then a triangle (genuine polygon).
            labels={"a.txt": "0 0.5 0.5 0.2 0.2\n0 0.1 0.1 0.5 0.2 0.3 0.6\n"},
        )
        s = analytics_svc.analyze_dataset(ds)["summary"]
        self.assertEqual(s["bbox_regions"], 1)
        self.assertEqual(s["polygon_regions"], 1)

    def test_gradient_closes_at_full_circle(self):
        ds = _make_dataset(
            self.src, "two", "", ["a", "b"], ["i.jpg"],
            labels={"i.txt": "0 0.1 0.1 0.1 0.1\n1 0.2 0.2 0.1 0.1\n"},
        )
        report = analytics_svc.analyze_dataset(ds)
        self.assertTrue(report["label_gradient"].endswith("100.000%"))

    def test_out_of_range_class_index_is_ignored(self):
        ds = _make_dataset(
            self.src, "oob", "", ["cat"], ["a.jpg"],
            labels={"a.txt": "0 0.1 0.1 0.1 0.1\n9 0.2 0.2 0.1 0.1\n"},
        )
        report = analytics_svc.analyze_dataset(ds)
        self.assertEqual(report["summary"]["total_regions"], 1)
        # The out-of-range region is flagged as a data-quality issue.
        self.assertEqual(report["quality"]["invalid_class_regions"], 1)

    def test_object_size_buckets(self):
        ds = _make_dataset(
            self.src, "sizes", "", ["thing"], ["a.jpg"],
            labels={"a.txt": (
                "0 0.5 0.5 0.05 0.05\n"   # area 0.0025 -> small (<1%)
                "0 0.5 0.5 0.2 0.2\n"     # area 0.04   -> medium (1-10%)
                "0 0.5 0.5 0.5 0.5\n"     # area 0.25   -> large (>=10%)
            )},
        )
        size = {s["label"]: s["count"] for s in analytics_svc.analyze_dataset(ds)["size_dist"]}
        self.assertEqual(size["Small (<1%)"], 1)
        self.assertEqual(size["Medium (1–10%)"], 1)
        self.assertEqual(size["Large (≥10%)"], 1)

    def test_per_image_density_and_empties(self):
        # img1: 1 box; img2: 12 boxes (crowded); img3: empty label file; img4: no file.
        twelve = "0 0.5 0.5 0.1 0.1\n" * 12
        ds = _make_dataset(
            self.src, "density", "", ["thing"],
            ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"],
            labels={"img1.txt": "0 0.5 0.5 0.1 0.1\n", "img2.txt": twelve, "img3.txt": "\n"},
        )
        report = analytics_svc.analyze_dataset(ds)
        s = report["summary"]
        self.assertEqual(s["labeled_images"], 2)             # img1, img2
        self.assertEqual(s["max_regions_per_image"], 12)
        self.assertEqual(s["median_regions_per_labeled_image"], 6.5)  # median(1, 12)
        self.assertEqual(s["crowded_images"], 1)             # img2 (>=10)
        self.assertEqual(s["empty_images"], 2)               # img3 (empty file) + img4 (no file)
        self.assertEqual(report["quality"]["empty_label_files"], 1)  # img3 only

    def test_orphan_label_files_detected(self):
        ds = _make_dataset(
            self.src, "orphans", "", ["cat"], ["a.jpg"],
            labels={"a.txt": "0 0.1 0.1 0.1 0.1\n", "ghost.txt": "0 0.2 0.2 0.1 0.1\n"},
        )
        q = analytics_svc.analyze_dataset(ds)["quality"]
        self.assertEqual(q["orphan_label_files"], 1)
        self.assertEqual(q["orphan_examples"], ["ghost.txt"])

    def test_out_of_bounds_and_zero_area_boxes(self):
        ds = _make_dataset(
            self.src, "bad", "", ["cat"], ["a.jpg"],
            labels={"a.txt": (
                "0 0.95 0.5 0.2 0.2\n"   # extends past x=1 -> out of bounds
                "0 0.5 0.5 0 0.2\n"      # zero width -> zero area
                "0 0.5 0.5 0.1 0.1\n"    # fine
            )},
        )
        q = analytics_svc.analyze_dataset(ds)["quality"]
        self.assertEqual(q["out_of_bounds_boxes"], 1)
        self.assertEqual(q["zero_area_boxes"], 1)

    def test_per_class_average_box_size(self):
        ds = _make_dataset(
            self.src, "avg", "", ["big", "small"], ["a.jpg"],
            labels={"a.txt": (
                "0 0.5 0.5 0.4 0.5\n"    # big: area 0.20 -> 20%
                "1 0.5 0.5 0.1 0.1\n"    # small: area 0.01 -> 1%
            )},
        )
        by_name = {r["name"]: r for r in analytics_svc.analyze_dataset(ds)["rows"]}
        self.assertEqual(by_name["big"]["avg_size_pct"], 20.0)
        self.assertEqual(by_name["small"]["avg_size_pct"], 1.0)


class ImageSourceDirTests(TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_prefers_images_subdir_when_populated(self):
        ds = self.root / "ds"
        (ds / "images").mkdir(parents=True)
        (ds / "images" / "f.jpg").write_bytes(b"x")
        self.assertEqual(lsapi.image_source_dir(ds), ds / "images")

    def test_falls_back_to_flat_without_images_subdir(self):
        ds = self.root / "ds"
        ds.mkdir()
        (ds / "f.jpg").write_bytes(b"x")
        self.assertEqual(lsapi.image_source_dir(ds), ds)

    def test_falls_back_when_images_subdir_has_no_images(self):
        ds = self.root / "ds"
        (ds / "images").mkdir(parents=True)
        (ds / "images" / "notes.txt").write_text("nope")
        self.assertEqual(lsapi.image_source_dir(ds), ds)


class YoloPolygonBoxTests(TestCase):
    def test_axis_aligned_four_point_polygon_imports_as_rectangle(self):
        # Real shape from test_dataset_0: class 5, four corners of a box.
        line = ("5 0.2747093023255814 0.39112050739957716 0.2949260042283298 "
                "0.39112050739957716 0.2949260042283298 0.4355179704016914 "
                "0.2747093023255814 0.4355179704016914")
        results = txt_format.results_for_label_text(line, _PPE_NAMES)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["type"], "rectanglelabels")
        v = results[0]["value"]
        self.assertEqual(v["rectanglelabels"], ["no_helmet"])
        self.assertAlmostEqual(v["x"], 0.2747093023255814 * 100, places=4)
        self.assertAlmostEqual(v["width"], (0.2949260042283298 - 0.2747093023255814) * 100, places=4)

    def test_plain_yolo_box_still_imports_as_rectangle(self):
        results = txt_format.results_for_label_text("7 0.5 0.5 0.2 0.4", _PPE_NAMES)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["type"], "rectanglelabels")
        self.assertEqual(results[0]["value"]["rectanglelabels"], ["vest"])

    def test_genuine_polygon_still_imports_as_polygon(self):
        # A triangle (3 points) is not a box — must stay a polygon.
        results = txt_format.results_for_label_text("2 0.1 0.1 0.5 0.2 0.3 0.6", _PPE_NAMES)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["type"], "polygonlabels")
        self.assertEqual(results[0]["value"]["polygonlabels"], ["helmet"])

    def test_non_axis_aligned_quad_stays_polygon(self):
        # Four points but a rotated/skewed quad (3 distinct xs) is a real polygon.
        results = txt_format.results_for_label_text(
            "2 0.10 0.10 0.50 0.12 0.55 0.50 0.08 0.48", _PPE_NAMES
        )
        self.assertEqual(results[0]["type"], "polygonlabels")


class DatasetTeardownTests(TestCase):
    def test_deleting_dataset_deletes_ls_projects_for_all_annotators(self):
        dataset = Dataset.objects.create(name="ds", storage_type=Dataset.LOCAL)
        ann1 = Annotator.objects.create(username="ann1")
        ann2 = Annotator.objects.create(username="ann2")
        Project.objects.create(annotator=ann1, dataset=dataset, ls_project_id=11)
        Project.objects.create(annotator=ann2, dataset=dataset, ls_project_id=22)

        with mock.patch.object(lsapi, "container_running", return_value=True), \
                mock.patch.object(lsapi, "delete_project") as del_proj:
            dataset.delete()

        deleted_ids = sorted(c.kwargs["project_id"] for c in del_proj.call_args_list)
        self.assertEqual(deleted_ids, [11, 22])
        self.assertFalse(Project.objects.filter(dataset_id=dataset.id).exists())

    def test_teardown_skips_when_container_down_and_does_not_block_delete(self):
        dataset = Dataset.objects.create(name="ds", storage_type=Dataset.LOCAL)
        ann = Annotator.objects.create(username="ann1")
        Project.objects.create(annotator=ann, dataset=dataset, ls_project_id=11)

        with mock.patch.object(lsapi, "container_running", return_value=False), \
                mock.patch.object(lsapi, "delete_project") as del_proj:
            dataset.delete()

        del_proj.assert_not_called()
        self.assertFalse(Dataset.objects.filter(id=dataset.id).exists())
