// Three behaviours on the "Live inference" inline of the camera change page:
// (1) show only the Pipeline-section fields the selected pipeline uses,
// (2) show only the model picker the selected model_source uses, and
// (3) when a model is picked (trained or exported), prefill the pipeline
// fields with the config it was trained through, so the default is "serve it
// the way it was trained" — same idea as
// videos/templates/admin/videos/run_inference.html, but scoped per inline row
// via id-suffix matching, since a StackedInline uses id_<prefix>-<index>-
// <field> ids rather than a single top-level id.
//
// Bundles are the exception to (3): they prefill on demand, via the "Sync bundle"
// button that training/static/training/bundle_sync.js drives — a bundle can have
// come from another machine, so it is worth *reporting* what it carries and
// whether it loads rather than quietly filling the form in. This file only hands
// the bundle rows over to that script and undoes its locks when the operator
// switches to a different source.
//
// Vanilla JS, matching training/static/training/experiment_model_form.js.
(function () {
    "use strict";

    // model_source value -> the field rows only that source uses. The bundle rows
    // include the "Sync bundle" block (a readonly field, hence field-bundle_tools).
    var SOURCE_FIELDS = {
        trained: ["trained_model"],
        exported: ["artifact_path"],
        bundle: ["bundle_path", "bundle_tools"],
    };

    // field name -> the pipeline values that show it. "raw" (and no pipeline
    // picked yet) never appear here, so either hides every pipeline-specific row.
    var FIELD_PIPELINES = {
        detector_checkpoint: ["people_detect_first", "batch_people", "chain"],
        detector_expand_ratio: ["people_detect_first", "batch_people", "chain"],
        detector_min_box_size: ["people_detect_first"],
        tile_size_px: ["batch_detect"],
        tile_width_pct: ["batch_detect", "batch_people", "chain"],
        tile_height_pct: ["batch_detect", "batch_people", "chain"],
        overlap: ["batch_detect", "batch_people", "chain"],
        // Every pipeline merges several per-frame predictions back together; only
        // "raw" (absent here, so the row hides) has nothing to merge.
        merge_nms_iou: ["batch_detect", "people_detect_first", "batch_people", "chain"],
        chain: ["chain"],
    };

    var NUMERIC_FIELDS = [
        "detector_expand_ratio", "detector_min_box_size", "tile_size_px",
        "tile_width_pct", "tile_height_pct", "overlap", "merge_nms_iou",
    ];

    function rowFor(el) {
        return el.closest(".inline-related") || el;
    }

    function syncVisibility(row) {
        var select = row.querySelector('select[id$="-pipeline"]');
        if (!select) {
            return;
        }
        var current = select.value;
        Object.keys(FIELD_PIPELINES).forEach(function (name) {
            var fieldRow = row.querySelector(".form-row.field-" + name);
            if (!fieldRow) {
                return;
            }
            var allowed = FIELD_PIPELINES[name];
            fieldRow.style.display = allowed.indexOf(current) !== -1 ? "" : "none";
        });
    }

    // Shows the one model picker the chosen source uses, so three selects (a
    // trained model, an exported artifact, a bundle) don't sit side by side with
    // two of them meaningless. The row for a picker in a *tuple* fieldset entry is
    // a .fieldBox rather than a .form-row, so both are tried.
    function syncSourceVisibility(row) {
        var select = row.querySelector('select[id$="-model_source"]');
        if (!select) {
            return;
        }
        Object.keys(SOURCE_FIELDS).forEach(function (source) {
            var shown = source === select.value;
            SOURCE_FIELDS[source].forEach(function (name) {
                var box = row.querySelector(".fieldBox.field-" + name)
                    || row.querySelector(".form-row.field-" + name);
                if (box) {
                    box.style.display = shown ? "" : "none";
                }
            });
        });
        // Leaving the bundle's locks in place while a different source is selected
        // would leave the operator with fields they can't edit and no bundle to
        // explain why.
        if (select.value !== "bundle" && window.BundleSync) {
            window.BundleSync.unlockAll(row);
        }
    }

    // Reads the JSON map a model-picker select carries (data-pipeline-defaults,
    // keyed by the select's own option value: trained_model's pk, or
    // artifact_path's relpath) and returns the entry for its current value.
    function defaultsFor(select) {
        if (!select) {
            return null;
        }
        var raw = select.getAttribute("data-pipeline-defaults");
        if (!raw) {
            return null;
        }
        var defaults;
        try {
            defaults = JSON.parse(raw);
        } catch (err) {
            return null;
        }
        return defaults[select.value] || null;
    }

    function fillDefaults(row, sourceSelect) {
        var pipelineSelect = row.querySelector('select[id$="-pipeline"]');
        if (!pipelineSelect) {
            return;
        }
        var d = defaultsFor(sourceSelect);
        if (!d) {
            return;
        }
        pipelineSelect.value = d.pipeline;
        NUMERIC_FIELDS.forEach(function (name) {
            var input = row.querySelector('[id$="-' + name + '"]');
            if (input) {
                input.value = (d[name] === null || d[name] === undefined) ? "" : d[name];
            }
        });
        var detectorInput = row.querySelector('[id$="-detector_checkpoint"]');
        if (detectorInput) {
            detectorInput.value = d.detector_checkpoint || "";
        }
        // The operating point the model was evaluated at, when it recorded one —
        // left alone otherwise, since the field has a non-null model default and
        // blanking it would fail validation.
        var scoreInput = row.querySelector('[id$="-score_threshold"]');
        if (scoreInput && d.score_threshold !== null && d.score_threshold !== undefined) {
            scoreInput.value = d.score_threshold;
        }
        var chainInput = row.querySelector('[id$="-chain"]');
        if (chainInput) {
            chainInput.value = JSON.stringify(d.chain || []);
        }
        syncVisibility(row);
    }

    function init() {
        document.querySelectorAll(".inline-related:not(.empty-form)").forEach(function (row) {
            syncVisibility(row);
            syncSourceVisibility(row);
        });

        document.addEventListener("change", function (event) {
            var target = event.target;
            if (!target || !target.matches) {
                return;
            }
            if (target.matches('select[id$="-pipeline"]')) {
                syncVisibility(rowFor(target));
            }
            if (target.matches('select[id$="-model_source"]')) {
                syncSourceVisibility(rowFor(target));
            }
            // Prefill runs only on an explicit change of the field that was just
            // edited, so switching model_source alone (or a form re-rendered
            // after a validation warning) never clobbers what the operator typed.
            if (target.matches('select[id$="-trained_model"]') ||
                target.matches('select[id$="-artifact_path"]')) {
                fillDefaults(rowFor(target), target);
            }
        });

        document.addEventListener("formset:added", function (event) {
            var target = event.target;
            if (target && target.closest) {
                var row = rowFor(target);
                syncVisibility(row);
                syncSourceVisibility(row);
                // A row cloned from the empty form carries its own copy of the
                // "Sync bundle" block, which bundle_sync.js has never seen.
                if (window.BundleSync) {
                    row.querySelectorAll("[data-bundle-sync]").forEach(window.BundleSync.attach);
                }
            }
        });
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
