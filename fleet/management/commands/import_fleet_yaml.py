"""One-time migration: import the legacy YAML fleet state into the database.

Reads ``configs/fleet.local.yaml`` (the live fleet) and ``configs/register.yaml``
(the durable roster) and upserts FleetSettings + Annotator rows. An annotator
present in fleet.local.yaml becomes ``active``; one only in the register becomes
``retired`` (removed-but-restorable). Idempotent — re-running updates in place.

With ``--backfill-projects`` it also asks each active annotator's Label Studio
for its projects, parses ``"<dataset> — <username>"`` titles, and creates
Project rows capturing the real ls_project_id (the value the old tool never
stored). That step makes live API calls, so it is opt-in.
"""

from pathlib import Path

import yaml
from django.conf import settings
from django.core.management.base import BaseCommand

from fleet.models import Annotator, Dataset, FleetSettings, Project, PROJECT_TITLE_SEP
from fleet.services import lsapi

_SETTINGS_KEYS = ["base_port", "image_name", "source_dir", "target_dir", "webhook_url"]
_RECORD_FIELDS = ["email", "password", "token", "port", "container_name", "volume_name", "ls_url"]


class Command(BaseCommand):
    help = "Import legacy configs/fleet.local.yaml + register.yaml into the database."

    def add_arguments(self, parser):
        configs = Path(settings.BASE_DIR) / "configs"
        parser.add_argument("--fleet", default=str(configs / "fleet.local.yaml"))
        parser.add_argument("--register", default=str(configs / "register.yaml"))
        parser.add_argument("--backfill-projects", action="store_true",
                            help="Also create Project rows by querying each active annotator's Label Studio.")

    def _load(self, path: str) -> dict:
        p = Path(path)
        if not p.exists():
            self.stdout.write(self.style.WARNING(f"missing (skipped): {p}"))
            return {}
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}

    def handle(self, *args, **opts):
        fleet = self._load(opts["fleet"])
        register = self._load(opts["register"])

        # FleetSettings from the live fleet's top-level keys.
        fs = FleetSettings.load()
        changed = False
        for key in _SETTINGS_KEYS:
            if key in fleet:
                setattr(fs, key, fleet[key])
                changed = True
        if changed:
            fs.save()
            self.stdout.write("Updated FleetSettings.")

        live = fleet.get("annotators") or {}
        roster = register or {}

        for username in sorted(set(live) | set(roster)):
            record = live.get(username) or roster.get(username) or {}
            status = Annotator.ACTIVE if username in live else Annotator.RETIRED
            defaults = {field: record[field] for field in _RECORD_FIELDS if record.get(field) is not None}
            defaults["status"] = status
            Annotator.objects.update_or_create(username=username, defaults=defaults)
            self.stdout.write(f"  {username:20} [{status}]")

        if opts["backfill_projects"]:
            self._backfill_projects()

    def _backfill_projects(self):
        self.stdout.write("Backfilling Project rows from Label Studio...")
        for annotator in Annotator.objects.filter(status=Annotator.ACTIVE):
            try:
                projects = lsapi.list_projects(ls_url=annotator.ls_url, api_token=annotator.token)
            except Exception as exc:  # noqa: BLE001
                self.stdout.write(self.style.WARNING(f"  {annotator.username}: {exc}"))
                continue
            for project in projects:
                title = project.get("title", "")
                if PROJECT_TITLE_SEP not in title:
                    continue
                dataset_name, _, owner = title.partition(PROJECT_TITLE_SEP)
                if owner.strip() != annotator.username:
                    continue
                dataset, _ = Dataset.objects.get_or_create(name=dataset_name.strip())
                Project.objects.update_or_create(
                    annotator=annotator,
                    dataset=dataset,
                    defaults={"ls_project_id": project.get("id"), "title": title},
                )
                self.stdout.write(f"  {annotator.username:20} {title} (id={project.get('id')})")
