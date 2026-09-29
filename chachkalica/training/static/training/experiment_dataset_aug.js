// In the ExperimentDataset inline, show each augmentation's fraction input
// only while its checkbox is ticked: id_datasets-<n>-aug_hflip toggles
// id_datasets-<n>-aug_hflip_fraction (same for aug_scale_crop). On val/test
// rows (id_datasets-<n>-role != "train") the checkbox is hidden too, since
// augmentations only apply to train datasets — unless it is already ticked, so
// a row that model.clean() rejects still shows the box to untick. Progressive
// enhancement only — with JS off the inputs stay visible and the model's
// clean() still validates them.
//
// Vanilla JS on purpose, matching experiment_model_form.js: django.jQuery may
// not be defined yet when this runs.
(function () {
    "use strict";

    var TOGGLES = ["aug_hflip", "aug_scale_crop"];
    var TRAIN = "train";

    // "id_datasets-3-aug_hflip" -> "id_datasets-3-"
    function rowPrefix(id, suffix) {
        return id.slice(0, id.length - suffix.length);
    }

    function isTrainRow(prefix) {
        var role = document.getElementById(prefix + "role");
        return !role || role.value === TRAIN;
    }

    function syncCheckbox(checkbox, toggle) {
        var prefix = rowPrefix(checkbox.id, toggle);
        var train = isTrainRow(prefix);
        checkbox.style.display = train || checkbox.checked ? "" : "none";
        var fraction = document.getElementById(checkbox.id + "_fraction");
        if (fraction) {
            fraction.style.display = checkbox.checked ? "" : "none";
        }
    }

    function syncRow(prefix) {
        TOGGLES.forEach(function (toggle) {
            var checkbox = document.getElementById(prefix + toggle);
            if (checkbox) {
                syncCheckbox(checkbox, toggle);
            }
        });
    }

    function syncAll(root) {
        TOGGLES.forEach(function (toggle) {
            root.querySelectorAll('input[type="checkbox"][id$="-' + toggle + '"]')
                .forEach(function (checkbox) { syncCheckbox(checkbox, toggle); });
        });
    }

    function toggleOf(el) {
        if (!el || !el.matches) {
            return null;
        }
        for (var i = 0; i < TOGGLES.length; i++) {
            if (el.matches('input[type="checkbox"][id$="-' + TOGGLES[i] + '"]')) {
                return TOGGLES[i];
            }
        }
        return null;
    }

    function init() {
        syncAll(document);

        document.addEventListener("change", function (event) {
            var target = event.target;
            var toggle = toggleOf(target);
            if (toggle) {
                syncCheckbox(target, toggle);
            } else if (target && target.matches && target.matches('select[id$="-role"]')) {
                syncRow(rowPrefix(target.id, "role"));
            }
        });

        // Django 4.1+ dispatches a native CustomEvent on the freshly added row
        // (it bubbles to document) after "Add another".
        document.addEventListener("formset:added", function (event) {
            if (event.target && event.target.querySelectorAll) {
                syncAll(event.target);
            }
        });
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
