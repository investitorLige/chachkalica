"""Report which catalogued VLM weights are present, and how to add the rest.

huggingface.co does not **verify** from this network — a TLS-intercepting proxy
re-signs it with an internal CA that is in no trust store here (see
``vlm.weights_catalog``) — so this command cannot download anything, and
deliberately does not try. What it does is tell you exactly which snapshots are
missing and give you the two commands to run on a machine that does have access,
which is the same manual process RT-DETR and D-FINE weights already went
through. The download command is `hf` (from `pip install -U huggingface_hub`) —
the older `huggingface-cli` alias is deprecated and, on huggingface_hub 1.x,
prints a warning and exits without downloading anything. The rsync destination it prints is the HOST path, not the /app path
this command sees from inside the container.

    python manage.py fetch_vlm_weights
    python manage.py fetch_vlm_weights --family qwen2_vl
"""

from django.core.management.base import BaseCommand

from fleet.services.provisioning import host_mount_path
from vlm import weights_catalog


class Command(BaseCommand):
    help = "Show which catalogued VLM weights are in the offline HF cache."

    def add_arguments(self, parser):
        parser.add_argument(
            "--family", default=None,
            help="Only check one family (%s)." % ", ".join(
                key for key, _label in weights_catalog.FAMILY_CHOICES),
        )

    def handle(self, *args, **options):
        family = options["family"]
        root = weights_catalog.hf_cache_root()

        self.stdout.write(f"HF cache: {root}")
        if not root.is_dir():
            self.stdout.write(self.style.WARNING(
                "  (does not exist yet — it is created by copying a snapshot in)"
            ))
        self.stdout.write("")

        missing = []
        for key, label in weights_catalog.FAMILY_CHOICES:
            if family and key != family:
                continue
            self.stdout.write(f"{label} [{key}]")
            for entry in weights_catalog.VLM_WEIGHTS_CATALOG.get(key, []):
                repo = entry["value"]
                if weights_catalog.is_cached(repo):
                    self.stdout.write(self.style.SUCCESS(f"  ✓ {repo}"))
                else:
                    self.stdout.write(self.style.WARNING(f"  ✗ {repo}"))
                    missing.append(repo)
            self.stdout.write("")

        if not missing:
            self.stdout.write(self.style.SUCCESS("Every catalogued checkpoint is cached."))
            return

        self.stdout.write(self.style.WARNING(
            f"{len(missing)} checkpoint(s) missing. On a machine with internet access:"
        ))
        self.stdout.write("")
        for repo in missing:
            # A repo may need companions (see the catalogue's ``requires``), so
            # emit a line per genuinely-absent repo id, not one per catalogue row.
            for needed in weights_catalog.missing_repos(repo):
                self.stdout.write(f"  hf download {weights_catalog.download_args(needed)}")
        self.stdout.write("")
        self.stdout.write(
            "then copy the matching directories out of that machine's "
            "~/.cache/huggingface/hub into this project's cache — for example:"
        )
        self.stdout.write("")
        dest = host_mount_path(root)
        for repo in missing:
            for needed in weights_catalog.missing_repos(repo):
                name = weights_catalog.cache_dir_name(needed)
                self.stdout.write(
                    f"  rsync -a ~/.cache/huggingface/hub/{name} <host>:{dest}/")
        self.stdout.write("")
        self.stdout.write(self.style.WARNING(
            "Use rsync -a (or tar), not scp -r: the HF cache stores each file once "
            "under blobs/ and symlinks it from snapshots/, and scp dereferences "
            "those symlinks — roughly doubling every transfer."
        ))
