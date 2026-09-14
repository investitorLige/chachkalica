// The "Sliceable by" panel on both evaluate forms.
//
// An eval's tag answers are read from whichever labels it scored against, so the
// dataset / label source / annotator fields on these forms are what decide which
// slices Tag analytics will later be able to cut. This keeps the panel in step
// with those fields, so that consequence is visible while the choice is being
// made rather than on a page days later.
//
// The server renders the panel; this only swaps the HTML in. That is deliberate
// -- the dataset-side form includes the very same partial directly, and a JS
// string-builder here would be a second renderer free to drift from it.
//
// Vanilla JS, matching bundle_sync.js, and the same data-attribute contract:
//   [data-tag-panel]          marks the container; its value is the endpoint URL
//   [data-tag-panel-report]   where the rendered panel goes
// plus, anywhere in the document, the fields it reads:
//   #dataset (a <select>, or a hidden input when the form has a fixed dataset),
//   #label_source, #annotator, #explicit_labels_path
(function () {
    "use strict";

    var DEBOUNCE_MS = 350;

    function value(id) {
        var node = document.getElementById(id);
        return node ? (node.value || "") : "";
    }

    function refresh(container) {
        var report = container.querySelector("[data-tag-panel-report]");
        var url = container.getAttribute("data-tag-panel");
        if (!report || !url) {
            return;
        }
        var dataset = value("dataset");
        if (!dataset) {
            report.innerHTML = "<p class=\"help\">Choose a dataset to see what an eval " +
                "against it could be sliced by.</p>";
            return;
        }

        var query = new URLSearchParams({
            dataset: dataset,
            label_source: value("label_source"),
            annotator: value("annotator"),
            explicit_labels_path: value("explicit_labels_path")
        });
        report.setAttribute("aria-busy", "true");
        fetch(url + "?" + query.toString(), {credentials: "same-origin"})
            .then(function (response) { return response.json(); })
            .then(function (data) {
                report.removeAttribute("aria-busy");
                if (data.error) {
                    report.innerHTML = "<p class=\"help\"></p>";
                    report.firstChild.textContent = data.error;
                    return;
                }
                report.innerHTML = data.html;
            })
            .catch(function (error) {
                report.removeAttribute("aria-busy");
                // A failed lookup costs the operator nothing but this panel, so
                // say so in place rather than interrupting the form.
                report.innerHTML = "<p class=\"help\"></p>";
                report.firstChild.textContent =
                    "Could not load the tag list: " + error;
            });
    }

    function attach(container) {
        var timer = null;
        function schedule() {
            window.clearTimeout(timer);
            timer = window.setTimeout(function () { refresh(container); }, DEBOUNCE_MS);
        }
        ["dataset", "label_source", "annotator", "explicit_labels_path"].forEach(
            function (id) {
                var node = document.getElementById(id);
                if (!node) {
                    return;
                }
                node.addEventListener("change", schedule);
                if (node.tagName === "INPUT") {
                    node.addEventListener("input", schedule);
                }
            });
        // The dataset-side form ships the panel already rendered; only fetch on
        // load when there is nothing in it yet (the model-side form).
        var report = container.querySelector("[data-tag-panel-report]");
        if (report && !report.children.length) {
            refresh(container);
        }
    }

    function init() {
        document.querySelectorAll("[data-tag-panel]").forEach(attach);
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
}());
