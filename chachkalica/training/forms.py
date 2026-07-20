"""Admin form for :class:`~training.models.ExperimentModel`.

The DB row stays model-agnostic — ``arch`` is a string and every builder kwarg
lives in the ``params`` JSON. This form is the human-facing layer over that JSON:
it renders a real widget per builder option (see :mod:`training.model_specs`) for
*every* architecture, and JavaScript (``experiment_model_form.js``) shows only the
ones belonging to the currently selected ``arch``. On save the selected arch's
values are folded back into ``params``.

Pretrained weights are a dropdown of their own (one per arch): the published,
license-checked checkpoints from ``model_specs.WEIGHTS_CATALOG``, plus the
operator's own promoted :class:`~training.models.TrainedModel` checkpoints for
that arch, plus a free-text custom path/URL. The selection resolves into
``params["weights"]`` on save, so ``config_gen`` needs no changes. Options can be
tied to a single variant (e.g. RF-DETR's Objects365 base weights); the JS shows
those only while their variant is selected.

Fields for the non-selected archs are still submitted but ignored: :meth:`save`
only reads the specs for the chosen ``arch``, and first strips every spec-owned
key so switching arch never leaves a stale kwarg a different adapter would reject.
"""

import json

from django import forms

from training import model_specs
from training.models import ExperimentModel, TrainedModel


class VariantAwareSelect(forms.Select):
    """A ``<select>`` that tags each ``<option>`` with metadata the JS reads.

    ``data-variant``: the option belongs to a single variant, so
    ``experiment_model_form.js`` shows it only while that variant is selected
    (options with no variant are always shown).

    ``data-train-res`` / ``data-train-res-map``: the resolution the checkpoint was
    pretrained at, annotated onto the option's label. A fixed-resolution option
    carries ``data-train-res``; the variant-resolved "default" option carries a
    ``data-train-res-map`` JSON ``{variant: res}`` the JS resolves against the row.
    """

    def __init__(self, *args, variant_map=None, res_map=None, default_res=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.variant_map = variant_map or {}
        self.res_map = res_map or {}
        self.default_res = default_res

    def create_option(self, name, value, *args, **kwargs):
        option = super().create_option(name, value, *args, **kwargs)
        variant = self.variant_map.get(str(value))
        if variant:
            option["attrs"]["data-variant"] = variant
        res = self.res_map.get(str(value))
        if res:
            option["attrs"]["data-train-res"] = res
        elif str(value) == model_specs.WEIGHTS_DEFAULT and self.default_res is not None:
            if isinstance(self.default_res, dict):
                option["attrs"]["data-train-res-map"] = json.dumps(self.default_res)
            else:
                option["attrs"]["data-train-res"] = self.default_res
        return option


def _build_field(spec: dict, arch: str) -> forms.Field:
    """One form field for a spec, tagged so the JS can show/hide it by arch."""
    kind = spec["kind"]
    label = spec.get("label", spec["key"])
    help_text = spec.get("help", "")
    default = spec.get("default")
    if default is not None and kind != "bool":
        help_text = (help_text + f" (adapter default: {default})").strip()

    attrs = {"class": "xm-spec-field", "data-arch": arch}
    attrs.update(spec.get("attrs", {}))
    common = {"required": False, "label": label, "help_text": help_text}

    if kind == "choice":
        choices = [("", "(default)")] + model_specs.normalized_choices(spec)
        return forms.ChoiceField(
            choices=choices, widget=forms.Select(attrs=attrs), **common
        )
    if kind == "int":
        return forms.IntegerField(widget=forms.NumberInput(attrs=attrs), **common)
    if kind == "float":
        return forms.FloatField(
            widget=forms.NumberInput(attrs={**attrs, "step": "any"}), **common
        )
    if kind == "bool":
        # NullBooleanSelect gives a three-way Unknown/Yes/No; Unknown = adapter default.
        return forms.NullBooleanField(widget=forms.NullBooleanSelect(attrs=attrs), **common)
    return forms.CharField(widget=forms.TextInput(attrs=attrs), **common)


def _spec_fields() -> dict:
    """A declared form field per builder option of every arch.

    Declared at class-definition time (below) so the fields land in
    ``base_fields``/``declared_fields``: Django admin only renders — and the
    inline formset factory only accepts — fields declared on the form class,
    never ones added dynamically in ``__init__``.
    """
    return {
        model_specs.field_name(arch, spec["key"]): _build_field(spec, arch)
        for arch, specs in model_specs.ARCH_FIELD_SPECS.items()
        for spec in specs
    }


def _weights_field(arch: str) -> forms.Field:
    """The pretrained-weights dropdown for one arch (static options only).

    The operator's own trained models are appended per instance in ``__init__``.
    """
    return forms.ChoiceField(
        required=False,
        label="Pretrained weights",
        choices=model_specs.weights_base_choices(arch),
        widget=VariantAwareSelect(
            attrs={"class": "xm-spec-field xm-weights-field", "data-arch": arch},
            variant_map=model_specs.weights_variant_map(arch),
            res_map=model_specs.weights_res_map(arch),
            default_res=model_specs.weights_default_res(arch),
        ),
        help_text="Published/native options use the architecture's own weight "
                  "format. 'Your model' performs a checked Friendy warm-start and "
                  "reinitializes incompatible task-head tensors. A custom native "
                  "reference is available only for adapters that support paths, "
                  "URLs, or repository ids.",
    )


def _weights_fields() -> dict:
    """One weights dropdown field per arch (declared on the class, like specs)."""
    return {
        model_specs.weights_field_name(arch): _weights_field(arch)
        for arch in model_specs.ARCH_FIELD_SPECS
    }


def _default_new_weights(arch: str) -> str:
    """The weights option a freshly added row of this arch starts on."""
    if arch == ExperimentModel.RTDETR:
        return "PekingU/rtdetr_r50vd"  # RT-DETR's original default size/checkpoint
    if arch in model_specs.WEIGHTS_DEFAULT_ARCHS:
        return model_specs.WEIGHTS_DEFAULT  # COCO pretrained
    return model_specs.WEIGHTS_NONE


class ExperimentModelForm(forms.ModelForm):
    # Free-text path/URL used when a weights dropdown is set to "Custom path or
    # URL…". Shared across archs — only the selected arch's dropdown is read.
    weights_custom = forms.CharField(
        required=False,
        label="Custom native pretrained reference",
        widget=forms.TextInput(attrs={"class": "xm-weights-custom", "size": "60"}),
        help_text="Architecture-native path, URL, or repository id—not a Friendy "
                  "best.pt/last.pt file. For those, choose the corresponding "
                  "'Your model' option.",
    )

    class Meta:
        model = ExperimentModel
        # `pretrained` is no longer a form field — the weights dropdown supersedes
        # it. The DB column stays (config_gen still honours it for legacy rows);
        # save() keeps it in sync with the dropdown selection.
        fields = ["arch", "num_classes", "params"]

    # Inject the per-option widgets into the class namespace so the metaclass
    # picks them up as declared fields (see _spec_fields / _weights_fields).
    locals().update(_spec_fields())
    locals().update(_weights_fields())

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        params = dict(getattr(self.instance, "params", None) or {})
        arch = self.instance.arch or ""

        self._append_dynamic_weights()

        # Seed each field of the *selected* arch with the stored value; only the
        # selected arch's fields are ever read back, so cross-arch key sharing
        # (e.g. "variant") is harmless.
        for spec in model_specs.ARCH_FIELD_SPECS.get(arch, []):
            if spec["key"] not in params:
                continue
            field = self.fields[model_specs.field_name(arch, spec["key"])]
            stored = params[spec["key"]]
            if spec["kind"] == "choice":
                stored_str = str(stored)
                if stored_str not in [c[0] for c in field.choices]:
                    # Preserve a hand-set value not in our list as a selectable option.
                    field.choices = list(field.choices) + [
                        (stored_str, f"(custom) {stored_str}")
                    ]
                field.initial = stored_str
            else:
                field.initial = stored

        if arch:
            self._seed_weights(arch, params)

        # The raw JSON stays for open-ended ModelConfig kwargs the specs don't cover.
        self.fields["params"].help_text = (
            "Advanced: extra architecture kwargs as JSON. The fields above override "
            "any matching keys here on save."
        )

    def _append_dynamic_weights(self) -> None:
        """Add runtime weights options the static catalog can't hold.

        Two sources: the locally re-hosted ByteTrack YOLOX weights (variant-tagged,
        existence-gated) and the operator's own promoted trained models (per arch,
        no variant). Both are inserted before the trailing "Custom path or URL…"
        sentinel, and variant-tagged ones also extend the widget's variant map so
        the JS filters them by the selected variant.
        """
        bytetrack = model_specs.bytetrack_yolox_options()
        if bytetrack:
            self._add_weight_options(ExperimentModel.YOLOX, bytetrack)

        by_arch: dict[str, list[dict]] = {}
        for tm in TrainedModel.objects.exclude(checkpoint_path="").order_by("name"):
            by_arch.setdefault(tm.arch, []).append(
                {
                    "value": model_specs.friendy_weights_value(tm.checkpoint_path),
                    "label": f"Your model: {tm.name}",
                }
            )
        for arch, options in by_arch.items():
            self._add_weight_options(arch, options)

    def _add_weight_options(self, arch: str, entries: list[dict]) -> None:
        """Splice weights options into an arch's dropdown + widget variant map."""
        field = self.fields.get(model_specs.weights_field_name(arch))
        if field is None:
            return
        choices = list(field.choices)
        insert_at = len(choices)
        if choices and choices[-1][0] == model_specs.WEIGHTS_CUSTOM:
            insert_at -= 1
        choices[insert_at:insert_at] = [
            (entry["value"], entry["label"]) for entry in entries]
        field.choices = choices
        variant_map = dict(getattr(field.widget, "variant_map", {}) or {})
        res_map = dict(getattr(field.widget, "res_map", {}) or {})
        for entry in entries:
            if entry.get("variant"):
                variant_map[str(entry["value"])] = entry["variant"]
            if entry.get("train_res"):
                res_map[str(entry["value"])] = entry["train_res"]
        field.widget.variant_map = variant_map
        field.widget.res_map = res_map

    def _seed_weights(self, arch: str, params: dict) -> None:
        """Set the weights dropdown (and custom field) to reflect stored state."""
        fname = model_specs.weights_field_name(arch)
        field = self.fields[fname]
        known = {c[0] for c in field.choices}

        init_checkpoint = params.get(model_specs.INIT_CHECKPOINT_KEY)
        if init_checkpoint:
            encoded = model_specs.friendy_weights_value(str(init_checkpoint))
            if encoded not in known:
                choices = list(field.choices)
                insert_at = len(choices)
                if choices and choices[-1][0] == model_specs.WEIGHTS_CUSTOM:
                    insert_at -= 1
                choices.insert(
                    insert_at,
                    (encoded, f"Friendy checkpoint: {init_checkpoint}"),
                )
                field.choices = choices
            field.initial = encoded
            return

        if "weights" in params:
            stored = params["weights"]
            if stored is True:
                field.initial = model_specs.WEIGHTS_DEFAULT
            elif isinstance(stored, str) and stored:
                if stored in known:
                    field.initial = stored
                else:
                    field.initial = model_specs.WEIGHTS_CUSTOM
                    self.fields["weights_custom"].initial = stored
            else:  # False / None / "" → explicit scratch
                field.initial = model_specs.WEIGHTS_NONE
        elif self.instance.pk and self.instance.pretrained:
            # Legacy row saved via the old checkbox.
            field.initial = model_specs.WEIGHTS_DEFAULT
        elif self.instance.pk:
            field.initial = model_specs.WEIGHTS_NONE
        else:  # brand-new row
            field.initial = _default_new_weights(arch)

    def clean(self):
        cleaned = super().clean()
        arch = cleaned.get("arch") or ""

        # RF-DETR's DINOv2 backbone needs the square input divisible by 56; a bad
        # value only surfaces as an epoch-1 crash deep in the trainer, so reject it
        # here at config time instead.
        if arch == ExperimentModel.RFDETR:
            fname = model_specs.field_name(ExperimentModel.RFDETR, "resolution")
            resolution = cleaned.get(fname)
            # The required multiple is patch_size * num_windows, which differs per
            # variant (56 for base, 32 for the others). Blank variant → the adapter
            # default (base, 56).
            variant = cleaned.get(
                model_specs.field_name(ExperimentModel.RFDETR, "variant")
            ) or "base"
            multiple = model_specs.RFDETR_VARIANT_RESOLUTION.get(
                variant, {"multiple": 56}
            )["multiple"]
            if resolution is not None and resolution % multiple != 0:
                self.add_error(
                    fname,
                    f"RF-DETR '{variant}' resolution must be divisible by {multiple} "
                    f"(got {resolution}).",
                )

        # A "Custom path or URL…" selection needs the accompanying text filled in.
        if arch:
            weights_sel = cleaned.get(model_specs.weights_field_name(arch))
            if weights_sel == model_specs.WEIGHTS_CUSTOM and not (
                cleaned.get("weights_custom") or ""
            ).strip():
                self.add_error(
                    "weights_custom",
                    "Enter a checkpoint path or URL, or pick a different "
                    "'Pretrained weights' option.",
                )
        return cleaned

    def save(self, commit=True):
        obj = super().save(commit=False)
        arch = self.cleaned_data.get("arch") or ""

        params = dict(self.cleaned_data.get("params") or {})
        # Drop every spec-owned key and the weights key, then re-apply only the
        # selected arch's values, so options from a previously selected arch don't
        # linger.
        for key in model_specs.ALL_SPEC_KEYS:
            params.pop(key, None)
        params.pop(model_specs.WEIGHTS_KEY, None)
        params.pop(model_specs.INIT_CHECKPOINT_KEY, None)
        for spec in model_specs.ARCH_FIELD_SPECS.get(arch, []):
            fname = model_specs.field_name(arch, spec["key"])
            value = self.cleaned_data.get(fname)
            if value in (None, ""):
                continue  # blank / "(default)" → let the adapter default apply
            params[spec["key"]] = value

        # Resolve the weights dropdown into params["weights"] (or leave it unset
        # for random init), and keep the legacy `pretrained` column consistent.
        weights_sel = self.cleaned_data.get(model_specs.weights_field_name(arch))
        friendy_checkpoint = model_specs.friendy_checkpoint_from_value(weights_sel)
        obj.pretrained = weights_sel == model_specs.WEIGHTS_DEFAULT
        if friendy_checkpoint:
            params[model_specs.INIT_CHECKPOINT_KEY] = friendy_checkpoint
        elif weights_sel == model_specs.WEIGHTS_DEFAULT:
            params[model_specs.WEIGHTS_KEY] = True
        elif weights_sel == model_specs.WEIGHTS_CUSTOM:
            custom = (self.cleaned_data.get("weights_custom") or "").strip()
            if custom:
                params[model_specs.WEIGHTS_KEY] = custom
        elif weights_sel == model_specs.WEIGHTS_NONE:
            # Explicit False matters for RF-DETR, whose adapter default is pretrained.
            params[model_specs.WEIGHTS_KEY] = False
        elif weights_sel not in (None, model_specs.WEIGHTS_NONE):
            params[model_specs.WEIGHTS_KEY] = weights_sel

        obj.params = params

        if commit:
            obj.save()
            self.save_m2m()
        return obj
