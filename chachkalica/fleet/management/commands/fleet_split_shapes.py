"""Report bbox/polygon consistency, then split each dataset by shape."""

from django.core.management.base import BaseCommand, CommandError

from fleet.models import Dataset
from fleet.services import datasets as datasets_svc
from fleet.services import shape_split


class Command(BaseCommand):
    help = (
        "For each named dataset (or every labeled dataset with --all), report "
        "how many regions are boxes vs. polygons and how often a stored bbox "
        "disagrees with its own polygon, then create '<name>-boxes' (box-only) "
        "and '<name>-polygons' (polygon-only) datasets. Datasets with no "
        "polygon regions are skipped — there's nothing to separate."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "datasets", nargs="*",
            help="Dataset names to split. Omit and use --all to target every labeled dataset.",
        )
        parser.add_argument("--all", action="store_true", help="Target every local dataset with labels.")

    def handle(self, *args, **opts):
        if opts["all"]:
            if opts["datasets"]:
                raise CommandError("Pass dataset names or --all, not both.")
            candidates = [
                d for d in Dataset.objects.filter(storage_type=Dataset.LOCAL)
                if datasets_svc.detect_labels(d)
            ]
        else:
            if not opts["datasets"]:
                raise CommandError("Pass one or more dataset names, or --all.")
            candidates = []
            for name in opts["datasets"]:
                try:
                    candidates.append(Dataset.objects.get(name=name))
                except Dataset.DoesNotExist:
                    raise CommandError(f"Unknown dataset: {name}")

        if not candidates:
            self.stdout.write("No labeled datasets found.")
            return

        self.stdout.write(
            f"{'dataset':30} {'regions':>8} {'bbox':>8} {'polygon':>8} {'mismatched':>11}"
        )
        reports = {}
        for dataset in candidates:
            report = shape_split.check_shape_consistency(dataset)
            reports[dataset.id] = report
            self.stdout.write(
                f"{dataset.name:30} {report['total_regions']:>8} {report['bbox_regions']:>8} "
                f"{report['polygon_regions']:>8} {report['mismatched_bbox_regions']:>11}"
            )

        self.stdout.write("")
        for dataset in candidates:
            report = reports[dataset.id]
            if report["polygon_regions"] == 0:
                self.stdout.write(f"{dataset.name}: skipped (no polygons found)")
                continue

            boxes_name = f"{dataset.name}-boxes"
            polygons_name = f"{dataset.name}-polygons"
            if Dataset.objects.filter(name__in=[boxes_name, polygons_name]).exists():
                self.stdout.write(
                    f"{dataset.name}: skipped ({boxes_name!r} or {polygons_name!r} already exists)"
                )
                continue

            try:
                result = shape_split.split_by_shape(dataset, boxes_name, polygons_name)
            except RuntimeError as exc:
                self.stdout.write(f"{dataset.name}: FAILED - {exc}")
                continue

            self.stdout.write(
                f"{dataset.name}: created {result['boxes_name']!r} "
                f"({result['box_labels']} label files) and {result['polygons_name']!r} "
                f"({result['polygon_labels']} label files) from {result['images']} images "
                f"(recomputed {result['mismatched_bbox_regions']} inconsistent bbox regions)"
            )
