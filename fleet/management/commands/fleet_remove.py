"""Stop/remove an annotator's container (optionally purge volume + row)."""

from django.core.management.base import BaseCommand, CommandError

from fleet.models import Annotator
from fleet.services import provisioning


class Command(BaseCommand):
    help = "Remove an annotator container. Keeps volume + row (restorable) unless --purge."

    def add_arguments(self, parser):
        parser.add_argument("username")
        parser.add_argument("--purge", action="store_true", help="Also delete the data volume and the row.")

    def handle(self, *args, **opts):
        try:
            annotator = Annotator.objects.get(username=opts["username"])
        except Annotator.DoesNotExist:
            raise CommandError(f"Unknown annotator: {opts['username']}")
        result = provisioning.remove_annotator(annotator, purge=opts["purge"])
        self.stdout.write(f"{opts['username']}: {result['outcome']}")
