// Show only the pipeline-config fields that apply to the pipeline currently
// selected in the Experiment "Training / Eval pipeline" section. Mirrors the
// eval flow's data-pipelines show/hide (see admin/training/evaluate_model.html),
// but on the admin change form: each field's enclosing .form-row is toggled so
// its label/help hide with the widget. A blank pipeline hides every pipeline row.
//
// Vanilla JS on purpose, matching experiment_model_form.js.
(function () {
    "use strict";

    // field name -> the pipeline values that use it.
    var FIELD_PIPELINES = {
        detector_checkpoint: ["people_detect_first", "batch_people", "chain"],
        detector_expand_ratio: ["people_detect_first", "batch_people", "chain"],
        detector_min_box_size: ["people_detect_first", "batch_people", "chain"],
        tile_size_px: ["batch_detect"],
        tile_width_pct: ["batch_detect", "batch_people", "chain"],
        tile_height_pct: ["batch_detect", "batch_people", "chain"],
        overlap: ["batch_detect", "batch_people", "chain"],
        merge_nms_iou: ["batch_detect", "people_detect_first", "batch_people", "chain"],
        chain: ["chain"],
    };

    function sync(select) {
        var current = select.value;
        Object.keys(FIELD_PIPELINES).forEach(function (name) {
            var row = document.querySelector(".form-row.field-" + name);
            if (!row) {
                return;
            }
            var allowed = FIELD_PIPELINES[name];
            var show = current !== "" && allowed.indexOf(current) !== -1;
            row.style.display = show ? "" : "none";
        });
    }

    function init() {
        var select = document.getElementById("id_pipeline");
        if (!select) {
            return;
        }
        select.addEventListener("change", function () {
            sync(select);
        });
        sync(select);
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
