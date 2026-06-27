"""Provision an annotator (or all of them) — the CLI twin of the admin action.

Runs synchronously (no queue), mirroring the old `fleet.py add`.
"""

from django.core.management.base import BaseCommand, CommandError

from fleet.models import Annotator
from fleet.services import provisioning


class Command(BaseCommand):
    help = "Create/start an annotator container and bootstrap its token. With no username, provisions every annotator."

    def add_arguments(self, parser):
        parser.add_argument("username", nargs="?", help="Annotator to provision; omit to provision all.")
        parser.add_argument("--email", help="Login email (default <username>@labelers.local).")

    def handle(self, *args, **opts):
        username = opts.get("username")
        if username:
            annotator, _ = Annotator.objects.get_or_create(username=username)
            if opts.get("email") and annotator.email != opts["email"]:
                annotator.email = opts["email"]
                annotator.save(update_fields=["email", "updated_at"])
            targets = [annotator]
        else:
            # Provision the live fleet only; retired annotators are restored
            # explicitly by name, not resurrected en masse.
            targets = list(Annotator.objects.filter(status=Annotator.ACTIVE))
            if not targets:
                raise CommandError("No active annotators. Add one: manage.py fleet_add <username>")

        for annotator in targets:
            result = provisioning.add_annotator(annotator)
            self.stdout.write(
                f"{annotator.username:20} {annotator.ls_url:25} {result.get('outcome')}"
            )
