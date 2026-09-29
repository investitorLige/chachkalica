"""Report which FireRed files are in the offline HF cache, and how to add the rest.

Like ``fetch_vlm_weights``: huggingface.co does not verify from this network
(TLS interception), so this downloads nothing. It checks the exact files
genaug-backend will ask for and prints the ``hf download`` commands to run on
a machine with access, then the rsync to bring them here.

    python manage.py fetch_genaug_weights                 # 4-bit, as used here
    python manage.py fetch_genaug_weights --lightning     # + the 8-step LoRA
    python manage.py fetch_genaug_weights --transformer bf16

What the 4-bit setup needs (~30 GB): the main repo *without* its 41 GB bf16
transformer weights (text encoder 16.6 GB, VAE, tokenizer, configs), plus the
13 GB q4_k_m GGUF from FireRed's ComfyUI repo.
"""

from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

from fleet.services.provisioning import host_mount_path
from ml_backends.genaug.editors.firered import (
    BASE_REPO, COMFY_REPO, GGUF_FILES, LIGHTNING_FILE,
)
from ml_backends.genaug.registry import required_repos


def hub_root() -> Path:
    return Path(settings.BASE_DIR) / "data" / "hf_cache" / "hub"


def is_present(repo: str, filename: str | None) -> bool:
    snapshots = hub_root() / ("models--" + repo.replace("/", "--")) / "snapshots"
    if not snapshots.is_dir():
        return False
    if not filename:
        return True
    return any((rev / filename).exists() for rev in snapshots.iterdir())


class Command(BaseCommand):
    help = "Show which FireRed-Image-Edit files are in the offline HF cache."

    def add_arguments(self, parser):
        parser.add_argument("--transformer", default="q4_k_m",
                            choices=[*GGUF_FILES, "bf16"])
        parser.add_argument("--lightning", action="store_true")

    def handle(self, *args, **options):
        transformer, lightning = options["transformer"], options["lightning"]
        root = hub_root()
        self.stdout.write(f"HF cache: {root}\n")
        missing = []
        for repo, filename in required_repos("firered", transformer, lightning):
            ok = is_present(repo, filename)
            label = f"{repo} :: {filename}"
            self.stdout.write(self.style.SUCCESS(f"  ✓ {label}") if ok
                              else self.style.WARNING(f"  ✗ {label}"))
            if not ok:
                missing.append(repo)
        self.stdout.write("")
        if not missing:
            self.stdout.write(self.style.SUCCESS("Everything genaug-backend needs is cached."))
            return

        self.stdout.write(self.style.WARNING("On a machine with internet access "
                                             "(pip install -U huggingface_hub):"))
        self.stdout.write("")
        if BASE_REPO in missing:
            # Everything but the bf16 transformer shards, unless those are wanted.
            exclude = "" if transformer == "bf16" else \
                ' --exclude "transformer/diffusion_pytorch_model-*.safetensors"'
            self.stdout.write(f"  hf download {BASE_REPO}{exclude}")
        if COMFY_REPO in missing:
            files = []
            if transformer != "bf16":
                files.append(GGUF_FILES[transformer])
            if lightning:
                files.append(LIGHTNING_FILE)
            self.stdout.write(f"  hf download {COMFY_REPO} {' '.join(files)}")
        self.stdout.write("")
        self.stdout.write("then copy the repo directories into this project's cache:")
        dest = host_mount_path(root)
        for repo in dict.fromkeys(missing):
            name = "models--" + repo.replace("/", "--")
            self.stdout.write(f"  rsync -a ~/.cache/huggingface/hub/{name} <host>:{dest}/")
        self.stdout.write("")
        self.stdout.write(self.style.WARNING(
            "Use rsync -a (or tar), not scp -r — scp dereferences the cache's "
            "blob symlinks and doubles every transfer."))
