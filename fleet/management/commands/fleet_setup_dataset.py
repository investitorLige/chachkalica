"""Create a dataset's projects + webhooks for one or all annotators."""

from django.core.management.base import BaseCommand, CommandError

from fleet.models import Annotator, Dataset
from fleet.services import datasets as datasets_svc


class Command(BaseCommand):
    help = "Create a '<dataset> — <username>' project (+ webhook) for selected annotators."

    def add_arguments(self, parser):
        parser.add_argument("dataset", help="Dataset name (directory under the source root).")
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument("--annotator", help="Target a single annotator by username.")
        group.add_argument("--all", action="store_true", help="Target every active annotator.")

    def handle(self, *args, **opts):
        dataset, _ = Dataset.objects.get_or_create(name=opts["dataset"])
        datasets_svc.detect_labels(dataset)  # flip has_labels if a labels/ folder exists

        if opts["all"]:
            annotators = list(Annotator.objects.filter(status=Annotator.ACTIVE))
            if not annotators:
                raise CommandError("No active annotators. Run manage.py fleet_add <username> first.")
        else:
            try:
                annotators = [Annotator.objects.get(username=opts["annotator"])]
            except Annotator.DoesNotExist:
                raise CommandError(f"Unknown annotator: {opts['annotator']}")

        for result in datasets_svc.setup_dataset(dataset, annotators):
            self.stdout.write(
                f"{result['username']:20} {result.get('status', ''):10} "
                f"project={result.get('project_id', '-')} "
                f"tasks={result.get('tasks', '-')} preds={result.get('predictions', '-')} "
                f"webhook={result.get('webhook', '-')}"
            )
