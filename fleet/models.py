"""Fleet state, moved out of the gitignored YAMLs into the database.

The old tooling kept two YAML stores in lockstep — ``fleet.local.yaml`` (the
*live* fleet) and ``register.yaml`` (the *durable* roster of every annotator
ever added, so a removed one could be restored). Both collapse into a single
``Annotator`` table here: ``status`` distinguishes live (``active``) from
removed-but-restorable (``retired``), and a full purge just deletes the row.

Secrets (``password``/``token``) live as plain columns — the same threat model
as the gitignored YAMLs, on a local/trusted Postgres. They are kept out of the
admin list/detail views.
"""

import secrets

from django.db import models

# Project title format, kept byte-for-byte compatible with the old tooling so
# the idempotent "skip if a project with this title exists" check still matches
# projects the CLI created. The separator is an em dash with surrounding spaces.
PROJECT_TITLE_SEP = " — "


def project_title(dataset_name: str, username: str) -> str:
    return f"{dataset_name}{PROJECT_TITLE_SEP}{username}"


class FleetSettings(models.Model):
    """Singleton row holding fleet-wide defaults (was the top of fleet.local.yaml)."""

    base_port = models.PositiveIntegerField(
        default=8081, help_text="First host port; new annotators take the next free one."
    )
    image_name = models.CharField(
        max_length=255, default="heartexlabs/label-studio:latest"
    )
    source_dir = models.CharField(
        max_length=512, default="data/source",
        help_text="Shared read-only source mount (relative to the project root or absolute).",
    )
    target_dir = models.CharField(
        max_length=512, default="data/target",
        help_text="Shared target mount where per-image txts + COCO are written.",
    )
    webhook_url = models.CharField(
        max_length=512, default="http://host.docker.internal:9000",
        help_text="Base URL each container POSTs annotation events to (the /hook receiver).",
    )

    class Meta:
        verbose_name = "Fleet settings"
        verbose_name_plural = "Fleet settings"

    def __str__(self) -> str:
        return "Fleet settings"

    def save(self, *args, **kwargs):
        self.pk = 1  # enforce a single row
        super().save(*args, **kwargs)

    def delete(self, *args, **kwargs):  # never delete the singleton
        pass

    @classmethod
    def load(cls) -> "FleetSettings":
        obj, _ = cls.objects.get_or_create(pk=1)
        return obj


class Annotator(models.Model):
    """One annotator == one isolated Label Studio container.

    Fields mirror the old per-annotator record. Identity-derived fields
    (container/volume/email) and secrets are auto-filled on first save, and the
    port is reserved from the first gap at/above ``FleetSettings.base_port`` —
    reserving across *all* rows (active and retired) so re-provisioning a
    retired annotator reclaims its original port.
    """

    ACTIVE = "active"
    RETIRED = "retired"
    STATUS_CHOICES = [(ACTIVE, "active"), (RETIRED, "retired")]

    username = models.CharField(max_length=150, unique=True)
    email = models.EmailField(blank=True)
    password = models.CharField(max_length=255, blank=True)  # secret
    token = models.CharField(max_length=255, blank=True)     # secret
    port = models.PositiveIntegerField(unique=True, null=True, blank=True)
    container_name = models.CharField(max_length=255, unique=True, blank=True)
    volume_name = models.CharField(max_length=255, unique=True, blank=True)
    ls_url = models.CharField(max_length=255, blank=True)
    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default=ACTIVE)

    # Outcome of the last enqueued job (provisioning/removal), for admin display.
    last_action = models.CharField(max_length=64, blank=True)
    last_status = models.CharField(max_length=32, blank=True)
    last_error = models.TextField(blank=True)
    last_run_at = models.DateTimeField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["username"]

    def __str__(self) -> str:
        return self.username

    @classmethod
    def reserve_port(cls) -> int:
        """First free port at/above base_port, across active AND retired rows."""
        base = FleetSettings.load().base_port
        used = set(cls.objects.exclude(port__isnull=True).values_list("port", flat=True))
        port = base
        while port in used:
            port += 1
        return port

    def save(self, *args, **kwargs):
        if not self.email:
            self.email = f"{self.username}@labelers.local"
        if not self.password:
            self.password = secrets.token_urlsafe(16)
        if not self.token:
            self.token = secrets.token_hex(20)
        if not self.container_name:
            self.container_name = f"label-studio-{self.username}"
        if not self.volume_name:
            self.volume_name = f"label-studio-{self.username}-data"
        if self.port is None:
            self.port = self.reserve_port()
        if not self.ls_url:
            self.ls_url = f"http://localhost:{self.port}"
        super().save(*args, **kwargs)


class Dataset(models.Model):
    """A labelable dataset. Source/target stay on disk or in a bucket — only
    the name (and cloud bucket root) live here, never image or label data.

    Paths are derived: source is ``<source_dir>/<name>`` and per-annotator
    output is ``<target_dir>/<name>/<username>/``. The labels and labeling
    tools come from the on-disk ``classes.txt``, not the database."""

    LOCAL = "local"
    CLOUD = "cloud"
    STORAGE_CHOICES = [(LOCAL, "local"), (CLOUD, "cloud")]

    name = models.CharField(
        max_length=255, unique=True,
        help_text="Directory name under the source root (e.g. dataset1).",
    )
    storage_type = models.CharField(max_length=16, choices=STORAGE_CHOICES, default=LOCAL)
    storage_root = models.CharField(
        max_length=1024, blank=True, help_text="Cloud bucket root (required when storage is cloud)."
    )
    has_labels = models.BooleanField(
        default=False,
        help_text="A labels/ folder was detected in the source dir; its contents are "
                  "imported as predictions when projects are created.",
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["name"]

    def __str__(self) -> str:
        return self.name


class Project(models.Model):
    """A Label Studio project: one dataset set up for one annotator.

    Persisting ``ls_project_id`` is the improvement over the old tooling, which
    re-looked-up projects by the ``"<dataset> — <username>"`` title on every
    sync. The title is kept for display and as an import-time fallback.
    """

    annotator = models.ForeignKey(Annotator, on_delete=models.CASCADE, related_name="projects")
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name="projects")
    ls_project_id = models.PositiveIntegerField(null=True, blank=True)
    webhook_id = models.PositiveIntegerField(null=True, blank=True)
    title = models.CharField(max_length=512, blank=True)

    last_status = models.CharField(max_length=32, blank=True)
    last_error = models.TextField(blank=True)
    last_run_at = models.DateTimeField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [("annotator", "dataset")]
        ordering = ["dataset__name", "annotator__username"]

    def __str__(self) -> str:
        return self.title or project_title(self.dataset.name, self.annotator.username)

    def save(self, *args, **kwargs):
        if not self.title and self.dataset_id and self.annotator_id:
            self.title = project_title(self.dataset.name, self.annotator.username)
        super().save(*args, **kwargs)
