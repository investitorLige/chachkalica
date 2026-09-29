"""Generative (prompted) image augmentation: prompt library, studio, builds.

A generative editor (FireRed-Image-Edit) is asked to change how an annotated
image *looks* — night, compression, low light — without moving anything, and
the source's labels are copied onto the result once a validator agrees
nothing moved. Unlike the flip/crop checkboxes on an experiment, this is
dataset *generation*: far too slow to run per batch, so it happens ahead of
training and produces an ordinary dataset folder that training then reads like
any other.

Three tabs:

* :class:`AugPrompt`   — **Prompt library**: reusable named prompts (the
  presets, plus anything saved from the studio).
* :class:`AugSession`  — **Studio**: one dataset, a handful of preview images,
  and any number of prompts tried against them (:class:`AugSessionPrompt`,
  each with one :class:`AugPreview` per preview image). This is where a prompt
  earns its place before hours of GPU are spent on it.
* :class:`AugBuild`    — **Dataset builds**: one run that applies the kept
  prompts to (a fraction of) the dataset and writes a new dataset.

Builds snapshot their prompts and settings rather than reading them back
through the session, so editing a session later cannot rewrite what a
finished build did — the same reasoning as ``training.TrainedModel``'s
frozen pipeline metadata.
"""

from django.db import models

from fleet.models import Annotator, Dataset

FIRERED = "firered"
MOCK = "mock"
EDITOR_CHOICES = [
    (FIRERED, "FireRed-Image-Edit-1.1"),
    (MOCK, "Mock — classical filters, no GPU (testing only)"),
]
TRANSFORMER_CHOICES = [
    ("q4_k_m", "4-bit q4_k_m — fits a 16 GB GPU"),
    ("q4_1", "4-bit q4_1"),
    ("bf16", "bf16 — full quality, needs ~45 GB of GPU"),
]

# Same vocabulary as training.ExperimentDataset, so labels resolve the same
# way everywhere (training.services.config_gen.resolve_label_dir).
SOURCE, ANNOTATOR, EXPLICIT = "source", "annotator", "explicit"
LABEL_SOURCE_CHOICES = [
    (SOURCE, "source labels"),
    (ANNOTATOR, "annotator output"),
    (EXPLICIT, "explicit path"),
]


class AugPrompt(models.Model):
    """A reusable, named editing prompt."""

    name = models.SlugField(
        max_length=64, unique=True,
        help_text="Short id, used in generated file names (e.g. night_cctv).",
    )
    text = models.TextField(help_text="The instruction sent to the editor.")
    negative_prompt = models.TextField(
        blank=True, help_text="What to steer away from. Blank sends FireRed's default ' '.",
    )
    description = models.CharField(max_length=255, blank=True)
    builtin = models.BooleanField(default=False, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["name"]
        verbose_name = "prompt"
        verbose_name_plural = "Prompt library"

    def __str__(self) -> str:
        return self.name


class EditorSettings(models.Model):
    """The editor + validation knobs a session and a build both carry."""

    editor = models.CharField(max_length=16, choices=EDITOR_CHOICES, default=FIRERED)
    transformer = models.CharField(
        max_length=16, choices=TRANSFORMER_CHOICES, default="q4_k_m",
        help_text="FireRed only. The 4-bit files are what fit this machine's GPU.",
    )
    lightning = models.BooleanField(
        default=False,
        help_text="FireRed's official 8-step Lightning LoRA: ~5x faster per image, "
                  "slightly less faithful. Steps/CFG are forced to 8/1.0.",
    )

    validation_enabled = models.BooleanField(
        default=True,
        help_text="Check every variant against its source before copying the labels "
                  "onto it. Off means every variant is trusted.",
    )
    max_bbox_drift = models.FloatField(
        default=0.05, help_text="Reject if any box moved more than this fraction of its size.",
    )
    min_box_similarity = models.FloatField(
        default=0.45,
        help_text="Reject if any box's structure correlates less than this with the "
                  "source (0-1; lighting/colour changes do not lower it).",
    )
    max_global_shift = models.FloatField(
        default=0.02,
        help_text="Reject if the whole scene shifted more than this fraction of the "
                  "frame diagonal (camera moved / reframed).",
    )

    class Meta:
        abstract = True

    def editor_config(self) -> dict:
        return {"editor": self.editor, "transformer": self.transformer,
                "lightning": self.lightning}

    def validation_settings(self) -> dict:
        return {"enabled": self.validation_enabled, "max_bbox_drift": self.max_bbox_drift,
                "min_box_similarity": self.min_box_similarity,
                "max_global_shift": self.max_global_shift}


class AugSession(EditorSettings):
    """One studio: a dataset, its preview images, and the prompts tried on it."""

    name = models.CharField(max_length=255)
    dataset = models.ForeignKey(Dataset, on_delete=models.PROTECT, related_name="+")
    label_source = models.CharField(max_length=16, choices=LABEL_SOURCE_CHOICES, default=SOURCE)
    annotator = models.ForeignKey(
        Annotator, on_delete=models.PROTECT, null=True, blank=True, related_name="+",
        help_text="Required when label source is 'annotator output'.",
    )
    explicit_labels_path = models.CharField(
        max_length=1024, blank=True, help_text="Required when label source is 'explicit path'.",
    )
    preview_count = models.PositiveSmallIntegerField(
        default=3, help_text="How many images each prompt is previewed on.",
    )
    preview_images = models.JSONField(
        default=list, blank=True,
        help_text="File names of the preview images (picked automatically).",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]
        verbose_name = "studio session"
        verbose_name_plural = "Studio"

    def __str__(self) -> str:
        return self.name

    def clean(self):
        from django.core.exceptions import ValidationError

        if self.label_source == ANNOTATOR and self.annotator_id is None:
            raise ValidationError({"annotator": "Pick an annotator for 'annotator output'."})
        if self.label_source == EXPLICIT and not self.explicit_labels_path.strip():
            raise ValidationError({"explicit_labels_path": "Provide a path for 'explicit path'."})
        if not 1 <= (self.preview_count or 0) <= 12:
            raise ValidationError({"preview_count": "Between 1 and 12."})


class Status(models.TextChoices):
    QUEUED = "queued"
    RUNNING = "running"
    OK = "ok"
    ERROR = "error"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED = "cancelled"


TERMINAL = {Status.OK, Status.ERROR, Status.CANCELLED}


class AugSessionPrompt(models.Model):
    """One prompt tried in a studio, with its generation settings."""

    session = models.ForeignKey(AugSession, on_delete=models.CASCADE, related_name="prompts")
    preset = models.ForeignKey(
        AugPrompt, on_delete=models.SET_NULL, null=True, blank=True, related_name="+",
    )
    name = models.SlugField(max_length=64)
    text = models.TextField()
    negative_prompt = models.TextField(blank=True)
    seed = models.IntegerField(default=42)
    num_inference_steps = models.PositiveSmallIntegerField(default=40)
    true_cfg_scale = models.FloatField(default=4.0)
    editor_config = models.JSONField(default=dict)
    keep = models.BooleanField(
        default=True, help_text="Include this prompt when building the dataset.",
    )
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.QUEUED)
    last_error = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.name} (seed {self.seed})"

    def generation(self) -> dict:
        """The prompt's generation parameters — the build's snapshot unit."""
        return {"name": self.name, "text": self.text, "negative_prompt": self.negative_prompt,
                "seed": self.seed, "num_inference_steps": self.num_inference_steps,
                "true_cfg_scale": self.true_cfg_scale}


class AugPreview(models.Model):
    """One preview image run through one studio prompt."""

    prompt = models.ForeignKey(AugSessionPrompt, on_delete=models.CASCADE, related_name="previews")
    source_filename = models.CharField(max_length=512)
    #: Relative to genaug_root() — the content-addressed cache file.
    output_relpath = models.CharField(max_length=512, blank=True)
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.QUEUED)
    error = models.TextField(blank=True)
    validation = models.JSONField(null=True, blank=True)
    accepted = models.BooleanField(null=True)
    edit_ms = models.PositiveIntegerField(null=True, blank=True)

    class Meta:
        ordering = ["id"]


class AugBuild(EditorSettings):
    """One dataset generation run over the kept prompts."""

    session = models.ForeignKey(
        AugSession, on_delete=models.SET_NULL, null=True, blank=True, related_name="builds",
    )
    source_dataset = models.ForeignKey(Dataset, on_delete=models.PROTECT, related_name="+")
    source_labels_dir = models.CharField(max_length=1024)
    output_name = models.CharField(
        max_length=255, help_text="Directory (and Dataset row) name under the source root.",
    )
    output_dataset = models.ForeignKey(
        Dataset, on_delete=models.SET_NULL, null=True, blank=True, related_name="+",
    )
    prompts_snapshot = models.JSONField(default=list)
    fraction = models.FloatField(default=0.3)
    variants_per_image = models.PositiveSmallIntegerField(default=2)
    class_sampling = models.JSONField(default=dict, blank=True)
    seed = models.IntegerField(default=42)
    include_originals = models.BooleanField(default=True)
    force = models.BooleanField(default=False)

    status = models.CharField(max_length=16, choices=Status.choices, default=Status.QUEUED)
    last_error = models.TextField(blank=True)
    #: Shown while the backend refuses to load because the GPU is in use.
    waiting_reason = models.CharField(max_length=512, blank=True)
    images_total = models.PositiveIntegerField(default=0)
    images_selected = models.PositiveIntegerField(default=0)
    variants_planned = models.PositiveIntegerField(default=0)
    variants_attempted = models.PositiveIntegerField(default=0)
    variants_accepted = models.PositiveIntegerField(default=0)
    variants_rejected = models.PositiveIntegerField(default=0)
    variants_errored = models.PositiveIntegerField(default=0)
    variants_cached = models.PositiveIntegerField(default=0)

    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "dataset build"
        verbose_name_plural = "Dataset builds"

    def __str__(self) -> str:
        return self.output_name

    def is_terminal(self) -> bool:
        return self.status in TERMINAL
