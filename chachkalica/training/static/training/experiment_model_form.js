// Show only the builder-option fields (rendered by ExperimentModelForm) that
// belong to the architecture currently selected in each ExperimentModel inline
// row. Fields are tagged with class "xm-spec-field" and data-arch="<arch>"; we
// toggle the enclosing admin .form-row so labels/help hide with the widget.
//
// The pretrained-weights dropdown (class "xm-weights-field") is one such field,
// but its <option>s are additionally variant-aware: an option tagged
// data-variant="<v>" is shown only while that variant is selected in the row
// (e.g. RF-DETR's Objects365 weights, which only fit the "base" variant). The
// shared "Custom weights (path or URL)" text field (class "xm-weights-custom")
// is shown only while the visible weights dropdown is set to the "__custom__"
// sentinel.
//
// Vanilla JS on purpose: relying on django.jQuery meant the script ran before
// jQuery was defined and threw, leaving every arch's fields visible.
(function () {
    "use strict";

    var CUSTOM = "__custom__";

    function setRowVisible(field, visible) {
        var formRow = field.closest(".form-row") || field;
        formRow.style.display = visible ? "" : "none";
    }

    // Annotate a weights <option>'s label with the resolution its checkpoint was
    // pretrained at, so operators can pick a matching training resolution. A fixed
    // option carries data-train-res; the variant-resolved "default" option carries
    // data-train-res-map ({variant: res}) resolved against the row's variant. The
    // pristine label is cached in data-base-label so re-annotating (on variant
    // change) never stacks suffixes. Informational only — nothing auto-sets the
    // training resolution.
    function annotateTrainRes(opt, variant) {
        var base = opt.getAttribute("data-base-label");
        if (base === null) {
            base = opt.textContent;
            opt.setAttribute("data-base-label", base);
        }
        var res = opt.getAttribute("data-train-res");
        if (!res) {
            var mapAttr = opt.getAttribute("data-train-res-map");
            if (mapAttr) {
                try {
                    res = JSON.parse(mapAttr)[variant];
                } catch (err) {
                    res = null;
                }
            }
        }
        opt.textContent = res ? base + " · trained @" + res : base;
    }

    // Filter the weights <select>'s options by the row's selected variant, then
    // reset the selection if the current choice was hidden.
    function syncWeightsOptions(row, arch) {
        var select = row.querySelector(
            '.xm-weights-field[data-arch="' + arch + '"]'
        );
        if (!select) {
            return;
        }
        // RT-DETR has no variant field (size == checkpoint), so nothing to
        // filter — its options carry no data-variant and all stay visible.
        var variantSelect = row.querySelector(
            'select[id$="-xm_' + arch + '_variant"]'
        );
        var variant = variantSelect ? variantSelect.value : "";

        var current = select.value;
        var currentHidden = false;
        Array.prototype.forEach.call(select.options, function (opt) {
            var optVariant = opt.getAttribute("data-variant");
            var show = !optVariant || optVariant === variant;
            opt.hidden = !show;
            opt.disabled = !show;
            if (!show && opt.value === current) {
                currentHidden = true;
            }
            annotateTrainRes(opt, variant);
        });
        if (currentHidden) {
            // Fall back to the first still-visible option (None / default).
            for (var i = 0; i < select.options.length; i++) {
                if (!select.options[i].hidden) {
                    select.value = select.options[i].value;
                    break;
                }
            }
        }
    }

    // Show the shared custom-weights text field only when the visible weights
    // dropdown is on the "__custom__" sentinel.
    function syncCustomWeights(row, arch) {
        var custom = row.querySelector(".xm-weights-custom");
        if (!custom) {
            return;
        }
        var select = row.querySelector(
            '.xm-weights-field[data-arch="' + arch + '"]'
        );
        setRowVisible(custom, !!select && select.value === CUSTOM);
    }

    function syncRow(row) {
        if (!row || !row.querySelectorAll) {
            return;
        }
        var archSelect = row.querySelector('select[id$="-arch"]');
        if (!archSelect) {
            return;
        }
        var arch = archSelect.value;
        row.querySelectorAll(".xm-spec-field").forEach(function (field) {
            setRowVisible(field, field.getAttribute("data-arch") === arch);
        });
        syncWeightsOptions(row, arch);
        syncCustomWeights(row, arch);
    }

    function rowFor(el) {
        return el.closest(".inline-related") || el.closest("tr") || el.closest(".form-row") || el;
    }

    // Selecting an RF-DETR variant fills the resolution field with that variant's
    // native square resolution (read from the select's data-native-resolutions map),
    // giving the user the right starting point to edit from.
    function fillRfdetrResolution(variantSelect) {
        var map;
        try {
            map = JSON.parse(variantSelect.getAttribute("data-native-resolutions") || "{}");
        } catch (err) {
            return;
        }
        var native = map[variantSelect.value];
        if (native === undefined) {
            return;  // "(default)" or a custom value — leave the field alone
        }
        var row = rowFor(variantSelect);
        var resInput = row.querySelector('input[id$="-xm_rfdetr_resolution"]');
        if (resInput) {
            resInput.value = native;
        }
    }

    function init() {
        // Skip the hidden empty-form template; real rows get cloned from it.
        document.querySelectorAll(".inline-related:not(.empty-form)").forEach(syncRow);

        document.addEventListener("change", function (event) {
            var target = event.target;
            if (!target || !target.matches) {
                return;
            }
            if (target.matches('select[id$="-arch"]')) {
                syncRow(rowFor(target));
            }
            // A variant change re-filters the weights options for its row.
            if (target.matches('select[id*="_variant"]')) {
                var row = rowFor(target);
                var archSelect = row.querySelector('select[id$="-arch"]');
                if (archSelect) {
                    syncWeightsOptions(row, archSelect.value);
                    syncCustomWeights(row, archSelect.value);
                }
            }
            if (target.matches('select[id$="-xm_rfdetr_variant"]')) {
                fillRfdetrResolution(target);
            }
            // Toggling the weights dropdown shows/hides the custom text field.
            if (target.matches(".xm-weights-field")) {
                var wrow = rowFor(target);
                var warchSelect = wrow.querySelector('select[id$="-arch"]');
                if (warchSelect) {
                    syncCustomWeights(wrow, warchSelect.value);
                }
            }
        });

        // Django 4.1+ dispatches a native CustomEvent on the freshly added row
        // (it bubbles to document) after "Add another".
        document.addEventListener("formset:added", function (event) {
            var target = event.target;
            if (target && target.closest) {
                syncRow(target.closest(".inline-related") || target);
            }
        });
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
