// Two behaviours on the "Live inference" inline of the camera change page:
// (1) show only the Pipeline-section fields the selected pipeline uses, and
// (2) when a model is picked (trained or exported), prefill the pipeline
// fields with the config it was trained through, so the default is "serve it
// the way it was trained" — same idea as
// videos/templates/admin/videos/run_inference.html, but scoped per inline row
// via id-suffix matching, since a StackedInline uses id_<prefix>-<index>-
// <field> ids rather than a single top-level id.
//
// Vanilla JS, matching training/static/training/experiment_model_form.js.
(function () {
    "use strict";

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
        document.querySelectorAll(".inline-related:not(.empty-form)").forEach(syncVisibility);

        document.addEventListener("change", function (event) {
            var target = event.target;
            if (!target || !target.matches) {
                return;
            }
            if (target.matches('select[id$="-pipeline"]')) {
                syncVisibility(rowFor(target));
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
                syncVisibility(rowFor(target));
            }
        });
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
