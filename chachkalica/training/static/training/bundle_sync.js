// The "Sync bundle" button, shared by every inference form that can run a bundle
// (the video "Run model inference…" wizard and the camera live-inference inline).
//
// A bundle is not just a model file: its pipeline.json carries the model, the
// detector and the geometry they were tuned with. So picking one is only half the
// job — "Sync bundle" is the other half. It POSTs the selection to
// admin/bundles/sync/ (fleetsite.admin_views.bundle_sync_view), renders the
// returned checklist, then writes the bundle's own values into the pipeline fields
// and locks them: for a bundle the *bundle* is the record of what runs, not the
// form. The server re-derives the same values on save
// (training.services.bundles.apply_defaults), so the lock is a statement of that
// fact rather than the thing enforcing it.
//
// Vanilla JS, matching cameras/camera_inference_form.js. Exposes
// window.BundleSync.attach(container) so each page can wire its own markup: this
// file knows the *contract*, the pages know their DOM.
//
// Contract — a container element carrying:
//   data-bundle-sync           marks it, and holds the endpoint URL as its value
//   data-bundle-scope          optional selector for where the pipeline fields
//                              live; defaults to the container's closest form
// and containing:
//   [data-bundle-button]       a type="button" that runs the sync
//   [data-bundle-load-test]    optional checkbox — adds the real load test
//   [data-bundle-report]       where the checklist is rendered
// plus, in the container *or* anywhere in the scope (the admin renders the field
// and this block as separate rows, so it can't be nested):
//   [data-bundle-select]       the <select> of bundles
(function () {
    "use strict";

    // Written from the bundle manifest on a successful sync, and locked
    // afterwards. Mirrors training.services.bundles._GEOMETRY_FIELDS, plus
    // score_threshold — which is *not* locked, being the one knob a bundle's own
    // README treats as tunable.
    var GEOMETRY_FIELDS = [
        "pipeline", "detector_checkpoint", "detector_expand_ratio",
        "detector_min_box_size", "tile_size_px", "tile_width_pct",
        "tile_height_pct", "overlap", "merge_nms_iou", "chain",
    ];

    var LOCKED_BACKGROUND = "#f0f0f0";
    var LOCKED_COLOR = "#666";
    var STATUS_MARKS = { ok: "✔", fail: "✖", info: "•" };
    var STATUS_COLORS = { ok: "#22c55e", fail: "#ef4444", info: "#9ca3af" };

    // Field lookup goes through the `name` attribute, not the id: a top-level form
    // names its input "pipeline" while an admin inline names the same field
    // "inference-0-pipeline", and matching either way keeps one script serving
    // both pages.
    function findField(scope, name) {
        return scope.querySelector('[name="' + name + '"]')
            || scope.querySelector('[name$="-' + name + '"]');
    }

    function scopeFor(container) {
        var selector = container.getAttribute("data-bundle-scope");
        if (selector) {
            return container.closest(selector) || document.querySelector(selector) || document;
        }
        return container.closest("form") || document;
    }

    // `chain` is a list in the metadata and a string in every form — but not the
    // same string: a plain form takes "a, b" while an admin JSONField takes JSON.
    function chainValue(field, chain) {
        var list = chain || [];
        return field.tagName === "TEXTAREA" ? JSON.stringify(list) : list.join(", ");
    }

    function lock(field) {
        if (field.tagName === "SELECT") {
            // A disabled select submits nothing, which would fail the admin's
            // required `pipeline` field — so a locked select stays enabled and
            // snaps back instead (see the guard listener below).
            field.dataset.bundleLocked = field.value;
            field.setAttribute("aria-readonly", "true");
        } else {
            field.readOnly = true;
        }
        field.style.background = LOCKED_BACKGROUND;
        field.style.color = LOCKED_COLOR;
        field.title = "Set by the bundle — its geometry was tuned with its weights.";
    }

    function unlock(field) {
        delete field.dataset.bundleLocked;
        field.removeAttribute("aria-readonly");
        field.readOnly = false;
        field.style.background = "";
        field.style.color = "";
        field.title = "";
    }

    // Public: drop every lock in `scope`, for when the form stops describing a
    // bundle (the operator switched model_source, or picked another bundle).
    function unlockAll(scope) {
        GEOMETRY_FIELDS.concat(["score_threshold"]).forEach(function (name) {
            var field = findField(scope, name);
            if (field) {
                unlock(field);
            }
        });
    }

    function applyDefaults(scope, defaults) {
        GEOMETRY_FIELDS.forEach(function (name) {
            var field = findField(scope, name);
            if (!field) {
                return;
            }
            unlock(field);  // so the write lands even on a re-sync
            if (name === "chain") {
                field.value = chainValue(field, defaults.chain);
            } else {
                var value = defaults[name];
                field.value = (value === null || value === undefined) ? "" : value;
            }
            lock(field);
        });
        // The bundle's shipped operating point, left editable: a bundle treats
        // confidence as the one thing worth tuning per deployment.
        var score = findField(scope, "score_threshold");
        if (score && defaults.score_threshold !== null && defaults.score_threshold !== undefined) {
            score.value = defaults.score_threshold;
        }
        // Let the page react (the pipeline rows that show/hide are its business,
        // not this script's).
        var pipeline = findField(scope, "pipeline");
        if (pipeline) {
            pipeline.dispatchEvent(new Event("change", { bubbles: true }));
        }
    }

    function line(text, color, bold) {
        var el = document.createElement("div");
        el.textContent = text;
        if (color) {
            el.style.color = color;
        }
        if (bold) {
            el.style.fontWeight = "bold";
        }
        return el;
    }

    // Built as text nodes rather than innerHTML on purpose: every `detail` string
    // originates in a bundle manifest or an exception message, i.e. a file that
    // arrived from another machine.
    function renderReport(target, result) {
        target.textContent = "";
        if (result.error) {
            target.appendChild(line("✖ " + result.error, STATUS_COLORS.fail, true));
            return;
        }
        target.appendChild(line(
            result.ok ? "✔ " + result.name + " — ready to run"
                      : "✖ " + result.name + " — not usable as-is",
            result.ok ? STATUS_COLORS.ok : STATUS_COLORS.fail, true));
        (result.checks || []).forEach(function (check) {
            target.appendChild(line(
                (STATUS_MARKS[check.status] || "•") + " " + check.label + ": " + check.detail,
                STATUS_COLORS[check.status]));
        });
        if (result.ok) {
            target.appendChild(line(
                "Pipeline fields below are set from this bundle and locked — they are "
                + "re-read from it on save.", STATUS_COLORS.info));
        }
    }

    function csrfToken(scope) {
        var input = scope.querySelector('[name="csrfmiddlewaretoken"]')
            || document.querySelector('[name="csrfmiddlewaretoken"]');
        return input ? input.value : "";
    }

    function attach(container) {
        if (container.dataset.bundleSyncReady) {
            return;
        }
        container.dataset.bundleSyncReady = "1";

        var url = container.getAttribute("data-bundle-sync");
        var button = container.querySelector("[data-bundle-button]");
        var report = container.querySelector("[data-bundle-report]");
        var loadTest = container.querySelector("[data-bundle-load-test]");
        var scope = scopeFor(container);
        var select = container.querySelector("[data-bundle-select]")
            || scope.querySelector("[data-bundle-select]");
        if (!url || !select || !button || !report) {
            return;
        }

        // Picking a different bundle invalidates the last sync: the fields still
        // hold the *previous* bundle's geometry, so they must not look settled.
        select.addEventListener("change", function () {
            unlockAll(scope);
            report.textContent = "";
            if (select.value) {
                report.appendChild(line("Not synced yet — press “Sync bundle”.",
                                        STATUS_COLORS.info));
            }
        });

        button.addEventListener("click", function () {
            if (!select.value) {
                report.textContent = "";
                report.appendChild(line("✖ Pick a bundle first.", STATUS_COLORS.fail, true));
                return;
            }
            var body = new URLSearchParams();
            body.set("bundle", select.value);
            body.set("csrfmiddlewaretoken", csrfToken(scope));
            if (loadTest && loadTest.checked) {
                body.set("load_test", "1");
            }

            var label = button.value || button.textContent;
            button.disabled = true;
            button.value = "Syncing…";
            report.textContent = "";
            report.appendChild(line(
                loadTest && loadTest.checked
                    ? "Loading the bundle on the GPU — this can take a while for a "
                      + "TensorRT engine…"
                    : "Checking the bundle…",
                STATUS_COLORS.info));

            fetch(url, {
                method: "POST",
                body: body,
                credentials: "same-origin",
                headers: { "X-Requested-With": "XMLHttpRequest" },
            }).then(function (response) {
                return response.json().catch(function () {
                    return { error: "Sync failed: HTTP " + response.status + "." };
                });
            }).then(function (result) {
                renderReport(report, result);
                // Applied whenever the manifest was readable, failing checks or
                // not: a broken bundle is not something to work around by hand in
                // the form, since the server re-derives these values anyway.
                if (result.defaults) {
                    applyDefaults(scope, result.defaults);
                }
            }).catch(function (err) {
                renderReport(report, { error: "Sync failed: " + err.message });
            }).then(function () {
                button.disabled = false;
                button.value = label;
            });
        });
    }

    // One document-level guard for every locked select: restore the bundle's value
    // instead of letting an edit stand, so "read-only" is what it looks like.
    document.addEventListener("change", function (event) {
        var target = event.target;
        if (target && target.dataset && target.dataset.bundleLocked !== undefined
                && target.value !== target.dataset.bundleLocked) {
            target.value = target.dataset.bundleLocked;
        }
    });

    function init() {
        document.querySelectorAll("[data-bundle-sync]").forEach(function (container) {
            // Django's inline template ships a hidden `.empty-form` row that is
            // cloned for each "add another". Attaching to the template would mark
            // it ready and every clone would inherit that flag, arriving inert —
            // so the clones are wired by whoever handles `formset:added`.
            if (!container.closest(".empty-form")) {
                attach(container);
            }
        });
    }

    window.BundleSync = {
        attach: attach,
        unlockAll: unlockAll,
        applyDefaults: applyDefaults,
        findField: findField,
        GEOMETRY_FIELDS: GEOMETRY_FIELDS,
    };

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
