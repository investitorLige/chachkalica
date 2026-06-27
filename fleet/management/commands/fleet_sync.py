"""Reconcile target/ from Label Studio for selected projects (default: all)."""

from django.core.management.base import BaseCommand

from fleet.models import Project
from fleet.services import sync as sync_svc


class Command(BaseCommand):
    help = "Rebuild per-image txts + COCO from Label Studio. Defaults to all projects."

    def add_arguments(self, parser):
        parser.add_argument("dataset", nargs="?", help="Only sync this dataset (default: all).")
        group = parser.add_mutually_exclusive_group()
        group.add_argument("--annotator", help="Target a single annotator by username.")
        group.add_argument("--all", action="store_true", help="Target every annotator (default).")

    def handle(self, *args, **opts):
        projects = Project.objects.select_related("annotator", "dataset")
        if opts.get("dataset"):
            projects = projects.filter(dataset__name=opts["dataset"])
        if opts.get("annotator"):
            projects = projects.filter(annotator__username=opts["annotator"])

        projects = list(projects)
        if not projects:
            self.stdout.write("No matching projects. Run fleet_setup_dataset first.")
            return

        for result in sync_svc.sync_projects(projects):
            if "status" in result:
                self.stdout.write(f"{result['username']:20} {result['dataset']:14} {result['status']}")
                continue
            warn = f"  ⚠ {len(result['errors'])} coco errors" if result["errors"] else ""
            self.stdout.write(
                f"{result['username']:20} {result['dataset']:14} "
                f"images={result['images']:<4} anns={result['annotations']:<4} "
                f"pruned={result['pruned']:<3}-> {result['coco_path']}{warn}"
            )
