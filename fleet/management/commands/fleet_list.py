"""List provisioned annotators and their container state."""

from django.core.management.base import BaseCommand

from fleet.models import Annotator
from fleet.services import lsapi


class Command(BaseCommand):
    help = "List annotators (live + retired) and whether their container is running."

    def handle(self, *args, **opts):
        annotators = Annotator.objects.all()
        if not annotators:
            self.stdout.write("No annotators provisioned yet.")
            return
        for annotator in annotators:
            if annotator.status == Annotator.RETIRED:
                state = "retired"
            elif lsapi.container_running(annotator.container_name):
                state = "running"
            else:
                state = "stopped"
            self.stdout.write(
                f"{annotator.username:20} {annotator.ls_url:25} {annotator.container_name:28} [{state}]"
            )
