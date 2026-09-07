"""Admin forms for the VLM section.

Two of them:

* :class:`VlmModelForm` — the Models tab's add/change form. The weights dropdown
  lists every family's checkpoints at once and lets JS narrow them to the
  selected family, the same ``data-``-attribute technique
  ``training.forms.VariantAwareSelect`` uses for architecture variants.
* :class:`VlmVideoAddForm` — the Videos tab's add form, which extends
  ``videos.admin.VideoAddForm``'s "import a file OR paste a link" to a third
  option: upload an mp4 straight from the browser.
"""

from pathlib import Path

from django import forms

from vlm import weights_catalog
from vlm.models import VlmModel, VlmVideo
from vlm.services.videos import VIDEO_EXTENSIONS, list_vlm_video_files


class FamilyAwareSelect(forms.Select):
    """A ``<select>`` tagging each ``<option>`` with the family it belongs to.

    ``vlm_model_form.js`` shows only the options whose ``data-family`` matches
    the family currently selected. Options for weights that are not in the
    offline HF cache are additionally marked ``data-cached="0"`` and disabled —
    they cannot possibly load on this network, and finding that out at add time
    is far better than at frame 1 of a run.
    """

    def __init__(self, *args, family_map=None, cached_map=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.family_map = family_map or {}
        self.cached_map = cached_map or {}

    def create_option(self, name, value, *args, **kwargs):
        option = super().create_option(name, value, *args, **kwargs)
        family = self.family_map.get(str(value))
        if family:
            option["attrs"]["data-family"] = family
        if str(value) in self.cached_map:
            cached = self.cached_map[str(value)]
            option["attrs"]["data-cached"] = "1" if cached else "0"
            if not cached:
                option["attrs"]["disabled"] = "disabled"
        return option


class VlmModelForm(forms.ModelForm):
    """Add/change a VLM model row.

    ``weights`` is a dropdown over the catalogue plus a "custom" escape hatch;
    picking custom reveals ``custom_weights``, which wins on save.
    """

    weights_choice = forms.ChoiceField(
        required=False, label="Weights",
        help_text="Only checkpoints already present in data/hf_cache can be selected — "
                  "huggingface.co is not reachable from this network.",
    )
    custom_weights = forms.CharField(
        required=False, label="Custom repo id / path",
        help_text="Used when 'Custom…' is selected above.",
    )

    class Meta:
        model = VlmModel
        fields = ["name", "description", "backend", "family", "prompt",
                  "max_new_tokens", "quantization", "variant"]

    class Media:
        js = ["vlm/vlm_model_form.js"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        options = weights_catalog.weights_options()
        family_map, cached_map = {}, {}
        for key, entries in weights_catalog.VLM_WEIGHTS_CATALOG.items():
            for entry in entries:
                family_map[entry["value"]] = key
                cached_map[entry["value"]] = weights_catalog.is_cached(entry["value"])

        field = self.fields["weights_choice"]
        field.choices = options
        field.widget = FamilyAwareSelect(
            choices=options, family_map=family_map, cached_map=cached_map,
        )

        self.fields["variant"].required = False
        self.fields["variant"].widget = forms.HiddenInput()

        current = getattr(self.instance, "weights", "")
        if current:
            known = {value for value, _label in options}
            if current in known:
                self.initial["weights_choice"] = current
            else:
                self.initial["weights_choice"] = weights_catalog.WEIGHTS_CUSTOM
                self.initial["custom_weights"] = current

    def clean(self):
        cleaned = super().clean()
        choice = cleaned.get("weights_choice")
        custom = (cleaned.get("custom_weights") or "").strip()

        if choice == weights_catalog.WEIGHTS_CUSTOM:
            if not custom:
                raise forms.ValidationError(
                    "Pick 'Custom…' and give a repo id or local path, or choose a "
                    "catalogued checkpoint."
                )
            resolved = custom
        elif choice:
            resolved = choice
        else:
            raise forms.ValidationError("Choose the weights this model should load.")

        cleaned["weights"] = resolved

        # A catalogued entry carries its own variant label; a custom one keeps
        # whatever the operator typed (or nothing).
        entry = weights_catalog.entry_for(resolved)
        if entry and entry.get("variant"):
            cleaned["variant"] = entry["variant"]

        if cleaned.get("backend") == weights_catalog.TRANSFORMERS and not weights_catalog.is_cached(resolved):
            self.add_error(
                "custom_weights" if choice == weights_catalog.WEIGHTS_CUSTOM else "weights_choice",
                f"{resolved} is not in data/hf_cache — copy its "
                f"'{weights_catalog.cache_dir_name(resolved)}' directory in first "
                f"(see the fetch_vlm_weights command).",
            )
        return cleaned

    def save(self, commit=True):
        obj = super().save(commit=False)
        obj.weights = self.cleaned_data["weights"]
        obj.variant = self.cleaned_data.get("variant") or ""
        if commit:
            obj.save()
        return obj


class VlmVideoAddForm(forms.ModelForm):
    """Add a video: upload one, import one already on disk, or paste a link.

    Exactly one of the three must be given. The two non-upload branches are the
    same ones ``videos.admin.VideoAddForm`` offers; the upload is new — nothing
    in this project accepted a browser file upload before.
    """

    QUALITY_CHOICES = [
        ("", "best available"),
        ("2160", "2160p (4K)"),
        ("1440", "1440p"),
        ("1080", "1080p"),
        ("720", "720p"),
        ("480", "480p"),
    ]

    upload = forms.FileField(
        required=False,
        label="Upload a video",
        help_text="An mp4 (or mov/mkv/webm/m4v) from this computer.",
    )
    import_file = forms.ChoiceField(
        required=False,
        label="Import existing file",
        help_text="A video already sitting in the VLM videos folder.",
    )
    source_url = forms.URLField(
        required=False,
        label="Download from link",
        help_text="Paste a YouTube (or other) link to download into the folder.",
    )
    quality = forms.ChoiceField(
        choices=QUALITY_CHOICES, required=False,
        label="Download quality",
        help_text="Only used when downloading from a link.",
    )

    class Meta:
        model = VlmVideo
        fields = ["name", "quality"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        taken = set(VlmVideo.objects.values_list("filename", flat=True))
        choices = [(f, f) for f in list_vlm_video_files() if f not in taken]
        self.fields["import_file"].choices = [("", "— select a file —")] + choices
        self.fields["name"].required = False
        self.fields["name"].help_text = "Optional — defaults to the file/video title."

    def clean(self):
        cleaned = super().clean()
        upload = cleaned.get("upload")
        import_file = cleaned.get("import_file")
        source_url = cleaned.get("source_url")

        given = [bool(upload), bool(import_file), bool(source_url)]
        if sum(given) != 1:
            raise forms.ValidationError(
                "Provide exactly one of: a file to upload, an existing file to "
                "import, or a link to download."
            )

        if upload:
            suffix = Path(upload.name).suffix.lower()
            if suffix not in VIDEO_EXTENSIONS:
                raise forms.ValidationError(
                    f"{upload.name}: not a video file "
                    f"({', '.join(sorted(VIDEO_EXTENSIONS))})."
                )
            if not cleaned.get("name"):
                cleaned["name"] = Path(upload.name).stem
        if import_file and not cleaned.get("name"):
            cleaned["name"] = Path(import_file).stem
        return cleaned
