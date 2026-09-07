// The Look section of "Run model inference for marketing…" — four behaviours,
// all of them about keeping the loop "change a knob, see the frame" tight.
//
//   1. Range inputs show their value, because "0.35" means nothing as a slider
//      position and everything as a number you can write down.
//   2. Rows opt into a condition with data-when="<field>=<values>" and hide when
//      it doesn't hold, so the section shows the knobs that currently do
//      something. Same idea as the pipeline rows' data-pipelines in
//      _inference_scripts.html, generalised: values are space-separated, "on"
//      means a ticked checkbox and "*" means a non-empty field.
//   3. The preview: POST this very form to the preview endpoint and show the
//      frame it renders. The form is posted whole and unmodified — the server
//      parses it with exactly the code that will parse the submitted job — so
//      nothing here knows the name of a single style knob.
//   4. The one knob whose vocabulary is the *model's* rather than this module's
//      — "label every box as" — is refilled when a different model is picked.
//
// Hidden rows still submit, and that is deliberate: the renderer ignores knobs
// its style doesn't use (a dash length with a solid outline), and keeping the
// value means flipping back to "dashed" restores what you had.
//
// Vanilla JS, matching cameras/camera_inference_form.js and
// training/bundle_sync.js.
(function () {
    "use strict";

    var form = document.getElementById("mk-form");
    if (!form) { return; }

    // ------------------------------------------------------------ range values
    function showRanges() {
        form.querySelectorAll("input.mk-range").forEach(function (input) {
            var output = input.parentNode.querySelector("output");
            if (output) { output.textContent = Number(input.value).toFixed(2); }
        });
    }

    function mirrorRanges() {
        form.querySelectorAll("input.mk-range").forEach(function (input) {
            input.addEventListener("input", showRanges);
        });
        showRanges();
    }

    // -------------------------------------------------------- conditional rows
    function fieldValue(name) {
        var field = form.querySelector('[name="' + name + '"]');
        if (!field) { return null; }
        if (field.type === "checkbox") { return field.checked ? "on" : "off"; }
        return (field.value || "").trim();
    }

    function syncRows() {
        form.querySelectorAll("[data-when]").forEach(function (row) {
            var parts = row.getAttribute("data-when").split("=");
            var value = fieldValue(parts[0]);
            var wanted = (parts[1] || "").split(/\s+/);
            var shown = wanted.indexOf("*") !== -1
                ? Boolean(value)
                : wanted.indexOf(value) !== -1;
            row.style.display = shown ? "" : "none";
        });
    }

    // ----------------------------------------------------------------- preview
    var shot = document.getElementById("mk-shot");
    var meta = document.getElementById("mk-meta");
    var errorBox = document.getElementById("mk-error");
    var position = document.getElementById("mk-position");
    var previewButton = document.getElementById("mk-preview-button");
    var rerunButton = document.getElementById("mk-rerun-button");
    var pending = null;
    var everRendered = false;

    function say(message) {
        errorBox.textContent = "";
        meta.textContent = message;
    }

    function fail(message) {
        meta.textContent = "";
        errorBox.textContent = message;
    }

    function preview(refresh) {
        var data = new FormData(form);
        data.set("video", form.getAttribute("data-video"));
        data.set("position", position.value);
        if (refresh) { data.set("refresh", "1"); }

        previewButton.disabled = rerunButton.disabled = true;
        say("Rendering…");
        fetch(form.getAttribute("data-preview-url"), {
            method: "POST",
            body: data,
            credentials: "same-origin",
            headers: { "X-Requested-With": "XMLHttpRequest" }
        }).then(function (response) {
            return response.json().then(function (payload) {
                return { ok: response.ok, payload: payload };
            });
        }).then(function (result) {
            if (!result.ok || result.payload.error) {
                fail(result.payload.error || "Preview failed.");
                return;
            }
            var data = result.payload;
            shot.innerHTML = "";
            var image = new Image();
            image.src = data.image;
            shot.appendChild(image);
            everRendered = true;
            say("frame " + data.frame_index + " of " + data.frames_total
                + " · " + data.boxes + " box" + (data.boxes === 1 ? "" : "es")
                + " drawn of " + data.detections + " detected · "
                + data.width + "×" + data.height
                + (data.cached ? " · detections re-used" : " · model re-run"));
        }).catch(function (exc) {
            fail(String(exc));
        }).then(function () {
            previewButton.disabled = rerunButton.disabled = false;
        });
    }

    // Style edits re-draw on their own, but only once there is a preview to
    // re-draw: nobody wants opening this page to queue work on the trainer, and
    // the first frame is an explicit press.
    function schedulePreview() {
        if (!everRendered) { return; }
        window.clearTimeout(pending);
        pending = window.setTimeout(function () { preview(false); }, 400);
    }

    // ------------------------------------------------------- relabel dropdown
    // "Label every box as" offers the selected model's class space, and the
    // model can be changed on this same form — so refill the options when it is.
    // The current pick survives a model change even if the new model has no such
    // class: the value is only ever *drawn* (see render_style.MarketingRenderer),
    // and dropping what the operator chose would be the more surprising of the
    // two behaviours.
    var forceSelect = document.getElementById("style_force_class");
    var classesByModel = {};
    (function () {
        var node = document.getElementById("mk-classes-by-model");
        if (node) { classesByModel = JSON.parse(node.textContent) || {}; }
    }());

    function refillForceClass(names) {
        if (!forceSelect) { return; }
        var current = forceSelect.value;
        var options = (names || []).slice();
        if (current && options.indexOf(current) === -1) { options.push(current); }
        forceSelect.length = 1;   // keep the "— keeps its own class —" option
        options.forEach(function (name) {
            forceSelect.appendChild(new Option(name, name, false, name === current));
        });
        forceSelect.value = current;
    }

    function watchModelSelect(id) {
        var select = document.getElementById(id);
        if (!select) { return; }
        select.addEventListener("change", function () {
            refillForceClass(classesByModel[select.value]);
        });
    }
    ["trained_model", "artifact_path", "bundle_path"].forEach(watchModelSelect);

    // ----------------------------------------------------------------- presets
    // A preset arrives already in the form's own shape — {"style_thickness": 3,
    // …} — so applying one is "assign each value to the input of that name" and
    // this file never learns what a knob is. Same on the way out: the save
    // endpoint is handed the whole form and parses it with the code that parses
    // the submitted job.
    var presetBlock = document.getElementById("mk-presets");
    var presetSelect = document.getElementById("mk-preset-select");
    var presetName = document.getElementById("mk-preset-name");
    var presetMessage = document.getElementById("mk-preset-message");
    var presets = {};
    (function () {
        var node = document.getElementById("mk-preset-data");
        if (!node) { return; }
        JSON.parse(node.textContent).forEach(function (preset) {
            presets[String(preset.pk)] = preset;
        });
    }());

    function applyValues(values) {
        Object.keys(values).forEach(function (name) {
            var field = form.querySelector('[name="' + name + '"]');
            if (!field) { return; }
            if (field.type === "checkbox") {
                field.checked = Boolean(values[name]);
            } else {
                // A <select> silently ignores a value it has no option for, which
                // for the one select with a per-model vocabulary ("label every box
                // as") would drop a preset's relabel without saying so. Every
                // other select's vocabulary is fixed, so this never fires there.
                if (field.tagName === "SELECT" && values[name]
                        && !field.querySelector('option[value="'
                                                + CSS.escape(values[name]) + '"]')) {
                    field.appendChild(new Option(values[name], values[name]));
                }
                field.value = values[name];
            }
        });
        showRanges();
        syncRows();
        schedulePreview();
    }

    function loadPreset() {
        var preset = presets[presetSelect.value];
        if (!preset) { return; }
        applyValues(preset.values);
        // Prefilled so that "load, tweak, save" amends the preset instead of
        // silently making a second one under a blank name.
        presetName.value = preset.name;
        presetMessage.textContent = "Loaded " + preset.name + ".";
    }

    function savePreset() {
        var name = (presetName.value || "").trim();
        if (!name) {
            presetMessage.textContent = "Name it first.";
            presetName.focus();
            return;
        }
        var data = new FormData(form);
        data.set("preset_name", name);
        presetMessage.textContent = "Saving…";
        fetch(presetBlock.getAttribute("data-save-url"), {
            method: "POST",
            body: data,
            credentials: "same-origin",
            headers: { "X-Requested-With": "XMLHttpRequest" }
        }).then(function (response) {
            return response.json().then(function (payload) {
                return { ok: response.ok, payload: payload };
            });
        }).then(function (result) {
            if (!result.ok || result.payload.error) {
                presetMessage.textContent = result.payload.error || "Could not save.";
                return;
            }
            var preset = result.payload;
            presets[String(preset.pk)] = preset;
            var option = presetSelect.querySelector('option[value="' + preset.pk + '"]');
            if (!option) {
                option = document.createElement("option");
                option.value = preset.pk;
                presetSelect.appendChild(option);
            }
            option.textContent = preset.name;
            presetSelect.value = String(preset.pk);
            presetMessage.textContent = preset.created
                ? "Saved as " + preset.name + "."
                : "Updated " + preset.name + ".";
        }).catch(function (exc) {
            presetMessage.textContent = String(exc);
        });
    }

    if (presetBlock) {
        document.getElementById("mk-preset-load")
            .addEventListener("click", loadPreset);
        document.getElementById("mk-preset-save")
            .addEventListener("click", savePreset);
        // Enter in the name box means save, not submit the whole render.
        presetName.addEventListener("keydown", function (event) {
            if (event.key === "Enter") { event.preventDefault(); savePreset(); }
        });
    }

    mirrorRanges();
    syncRows();
    form.addEventListener("change", function () { syncRows(); schedulePreview(); });
    form.addEventListener("input", function (event) {
        if (event.target.type === "range" || event.target.tagName === "INPUT") {
            syncRows();
            schedulePreview();
        }
    });
    position.addEventListener("change", function () { if (everRendered) { preview(false); } });
    previewButton.addEventListener("click", function () { preview(false); });
    rerunButton.addEventListener("click", function () { preview(true); });
}());
