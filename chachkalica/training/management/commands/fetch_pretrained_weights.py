"""Re-host the ByteTrack YOLOX pretrained weights into the project's weights dir.

ByteTrack (MIT) publishes YOLOX detectors trained on CrowdHuman+MOT17+Cityperson+
ETHZ — a strong person/crowd-detection start for fine-tuning. They live on Google
Drive, which ``torch.hub`` can't fetch, so this command downloads them once into
``data/training/weights`` where the "Pretrained weights" dropdown then offers them
(the dropdown is existence-gated, so options appear only after a successful fetch).

    python manage.py fetch_pretrained_weights            # all YOLOX variants
    python manage.py fetch_pretrained_weights --only yolox-s yolox-m
    python manage.py fetch_pretrained_weights --force    # re-download existing

Idempotent: an already-present, valid checkpoint is skipped unless ``--force``.
"""

import os
import shutil
import urllib.request

from django.core.management.base import BaseCommand, CommandError

from training import model_specs


# drive.usercontent with confirm=t bypasses Google Drive's large-file scan page
# and streams the binary directly (verified against these public ByteTrack ids).
_GDRIVE_URL = "https://drive.usercontent.google.com/download?id={id}&export=download&confirm=t"


class Command(BaseCommand):
    help = "Download the ByteTrack YOLOX pretrained weights into the weights dir."

    def add_arguments(self, parser):
        parser.add_argument(
            "--only", nargs="+", metavar="VARIANT",
            choices=sorted(model_specs.BYTETRACK_YOLOX),
            help="Fetch only these YOLOX variants (default: all).",
        )
        parser.add_argument(
            "--force", action="store_true",
            help="Re-download even if the checkpoint already exists.",
        )

    def handle(self, *args, **options):
        variants = options["only"] or list(model_specs.BYTETRACK_YOLOX)
        dest_dir = model_specs.weights_dir()
        os.makedirs(dest_dir, exist_ok=True)
        self.stdout.write(f"Weights dir: {dest_dir}")

        fetched, skipped, failed = 0, 0, 0
        for variant in variants:
            info = model_specs.BYTETRACK_YOLOX[variant]
            path = os.path.join(dest_dir, info["filename"])
            if os.path.exists(path) and not options["force"]:
                self.stdout.write(f"  {variant}: already present ({info['filename']}), skipping")
                skipped += 1
                continue
            try:
                self._download(info["gdrive_id"], path)
                self.stdout.write(self.style.SUCCESS(
                    f"  {variant}: fetched {info['filename']} "
                    f"({os.path.getsize(path) / 1e6:.1f} MB)"
                ))
                fetched += 1
            except Exception as exc:  # noqa: BLE001
                failed += 1
                if os.path.exists(path):
                    os.remove(path)  # don't leave a truncated file the dropdown offers
                self.stderr.write(self.style.ERROR(f"  {variant}: FAILED — {exc}"))

        summary = f"Fetched {fetched}, skipped {skipped}, failed {failed}."
        if failed:
            raise CommandError(summary)
        self.stdout.write(self.style.SUCCESS(summary))

    def _download(self, gdrive_id: str, path: str) -> None:
        url = _GDRIVE_URL.format(id=gdrive_id)
        tmp = path + ".part"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as out:
            shutil.copyfileobj(resp, out)
        # A torch checkpoint is a zip ("PK\x03\x04"); anything else (an HTML scan
        # page, a quota error) means the download didn't actually give us a model.
        with open(tmp, "rb") as fh:
            magic = fh.read(4)
        if magic[:2] != b"PK":
            os.remove(tmp)
            raise RuntimeError(
                "downloaded content is not a checkpoint (Google Drive may be "
                "rate-limiting or the file id changed)"
            )
        os.replace(tmp, path)
