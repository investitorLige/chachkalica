// The "Check classes" button on the dataset-eval form.
//
// An eval matches predictions to ground truth *by class name*, and drops both
// sides of anything the two class spaces don't share. So a model and a dataset
// that name their classes differently score nothing at all — every box dropped,
// 0.0 reported for every metric, indistinguishable from a model that detects
// nothing. This button says so before the run instead: it POSTs the selected
// model to admin/classes/sync/ (fleetsite.admin_views.class_sync_view), renders
// the returned checklist, and builds one row per dataset class where the
// operator maps it onto a class the model actually predicts — or drops it.
//
// Leaving the block closed posts no class_map fields at all, and the eval runs
// exactly as it did before this existed. That is the intended default: the map
// is for the mismatch, not for every run.
//
// Vanilla JS in the style of bundle_sync.js, whose contract this mirrors:
//   data-class-sync            marks the container, holds the endpoint URL
//   data-class-dataset         the dataset pk to compare against
//   [data-class-button]        a type="button" that runs the check
//   [data-class-report]        where the checklist is rendered
//   [data-class-table]         where the mapping rows are built
// The model <select> is found the way the form names it, so one script serves
// all three model sources.
(function () {
    "use strict";

    var STATUS_MARKS = { ok: "✔", fail: "✖", info: "•" };
    var STATUS_COLORS = { ok: "#22c55e", fail: "#ef4444", info: "#9ca3af" };
    var DROP_LABEL = "— drop from this eval —";
    var FIELD_PREFIX = "class_map__";

    // Whichever of the three the current model_source rendered.
    function modelSelect(scope) {
        return scope.querySelector('[name="trained_model"]')
            || scope.querySelector('[name="artifact_path"]')
            || scope.querySelector('[name="bundle_path"]');
    }

    function scopeFor(container) {
        return container.closest("form") || document;
    }

    function line(text, color, bold) {
        var el = document.createElement("div");
        el.textContent = text;
        if (color) { el.style.color = color; }
        if (bold) { el.style.fontWeight = "bold"; }
        return el;
    }

    function csrfToken(scope) {
        var input = scope.querySelector('[name="csrfmiddlewaretoken"]')
            || document.querySelector('[name="csrfmiddlewaretoken"]');
        return input ? input.value : "";
    }

    // Text nodes rather than innerHTML throughout: every name here comes from a
    // classes.txt or a .meta.json, i.e. a file that arrived from another machine.
    function renderReport(target, result) {
        target.textContent = "";
        if (result.error) {
            target.appendChild(line("✖ " + result.error, STATUS_COLORS.fail, true));
            return;
        }
        target.appendChild(line(
            result.ok
                ? "✔ " + result.name + " — shares class names with this dataset"
                : "✖ " + result.name + " — would score nothing as it stands",
            result.ok ? STATUS_COLORS.ok : STATUS_COLORS.fail, true));
        (result.checks || []).forEach(function (check) {
            target.appendChild(line(
                (STATUS_MARKS[check.status] || "•") + " " + check.label + ": " + check.detail,
                STATUS_COLORS[check.status]));
        });
    }

    // One row per dataset class: its name, and a <select> of the model's classes
    // plus "drop". Named class_map__<dataset class> so the server reads them back
    // per class (training.services.class_sync.parse_posted).
    function renderTable(target, result) {
        target.textContent = "";
        if (result.error || result.unknown || !(result.dataset_classes || []).length) {
            return;
        }
        var table = document.createElement("table");
        table.style.margin = "8px 0";
        var head = document.createElement("tr");
        ["Dataset class", "scored as", "Model class"].forEach(function (text) {
            var th = document.createElement("th");
            th.textContent = text;
            th.style.textAlign = "left";
            th.style.padding = "2px 10px 2px 0";
            th.style.fontSize = "11px";
            th.style.color = "#666";
            head.appendChild(th);
        });
        table.appendChild(head);

        result.dataset_classes.forEach(function (name) {
            var row = document.createElement("tr");

            var label = document.createElement("td");
            label.textContent = name;
            label.style.padding = "2px 10px 2px 0";
            label.style.fontFamily = "monospace";
            row.appendChild(label);

            var arrow = document.createElement("td");
            arrow.textContent = "→";
            arrow.style.padding = "2px 10px 2px 0";
            arrow.style.color = "#999";
            row.appendChild(arrow);

            var cell = document.createElement("td");
            var select = document.createElement("select");
            select.name = FIELD_PREFIX + name;

            var drop = document.createElement("option");
            drop.value = "";
            drop.textContent = DROP_LABEL;
            select.appendChild(drop);

            (result.model_classes || []).forEach(function (modelClass) {
                var option = document.createElement("option");
                option.value = modelClass;
                option.textContent = modelClass;
                select.appendChild(option);
            });

            // The server's proposal, which the operator is free to change. null
            // means "drop", and is the empty option already in place.
            var suggested = (result.suggested || {})[name];
            if (suggested !== null && suggested !== undefined) {
                select.value = suggested;
            }
            cell.appendChild(select);
            cell.style.padding = "2px 0";
            row.appendChild(cell);
            table.appendChild(row);
        });
        target.appendChild(table);
        target.appendChild(line(
            "Submitting with this table showing sends it with the eval. A row left on "
            + "“drop” excludes that class from the run entirely — its boxes are "
            + "neither scored nor counted as misses.", STATUS_COLORS.info));
    }

    function attach(container) {
        if (container.dataset.classSyncReady) { return; }
        container.dataset.classSyncReady = "1";

        var url = container.getAttribute("data-class-sync");
        var dataset = container.getAttribute("data-class-dataset");
        var button = container.querySelector("[data-class-button]");
        var report = container.querySelector("[data-class-report]");
        var table = container.querySelector("[data-class-table]");
        var scope = scopeFor(container);
        if (!url || !button || !report || !table) { return; }

        // A different model has a different class space, so the table on screen
        // describes the *previous* one and must not look settled.
        var select = modelSelect(scope);
        if (select) {
            select.addEventListener("change", function () {
                report.textContent = "";
                table.textContent = "";
                report.appendChild(line("Model changed — press “Check classes” again.",
                                        STATUS_COLORS.info));
            });
        }

        button.addEventListener("click", function () {
            var body = new URLSearchParams();
            body.set("csrfmiddlewaretoken", csrfToken(scope));
            body.set("dataset", dataset);
            var source = scope.querySelector('[name="model_source"]');
            body.set("model_source", source ? source.value : "");
            var current = modelSelect(scope);
            if (current) { body.set(current.name, current.value); }

            var label = button.value;
            button.disabled = true;
            button.value = "Checking…";
            report.textContent = "";
            table.textContent = "";
            report.appendChild(line("Reading the model's class space…", STATUS_COLORS.info));

            fetch(url, {
                method: "POST",
                body: body,
                credentials: "same-origin",
                headers: { "X-Requested-With": "XMLHttpRequest" },
            }).then(function (response) {
                return response.json().catch(function () {
                    return { error: "Check failed: HTTP " + response.status + "." };
                });
            }).then(function (result) {
                renderReport(report, result);
                renderTable(table, result);
            }).catch(function (err) {
                renderReport(report, { error: "Check failed: " + err.message });
            }).then(function () {
                button.disabled = false;
                button.value = label;
            });
        });
    }

    function init() {
        document.querySelectorAll("[data-class-sync]").forEach(attach);
    }

    window.ClassSync = { attach: attach };

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
