// The generative-augmentation studio: fills the preview cards from the state
// endpoint, polls while anything is still generating, and wires the view
// toggles, the keep checkbox, preset loading and the build estimate.
//
// Vanilla JS, like the rest of the admin's scripts. Every preview figure is
// rendered empty server-side and filled here, so there is one render path.
(function () {
    "use strict";

    var root = document.getElementById("studio");
    if (!root) {
        return;
    }
    var POLL_MS = 3000;
    var stateUrl = root.dataset.stateUrl;
    var imageUrl = root.dataset.imageUrl;
    var lastState = null;

    function csrfToken() {
        var input = document.querySelector("input[name=csrfmiddlewaretoken]");
        return input ? input.value : "";
    }

    function viewMode() {
        var checked = document.querySelector("input[name=view]:checked");
        return checked ? checked.value : "generated";
    }

    function showBoxes() {
        var box = document.getElementById("show-boxes");
        return !box || box.checked;
    }

    function pct(value) {
        return (100 * value).toFixed(1) + "%";
    }

    function verdictHtml(pv) {
        if (pv.status === "error") {
            return '<span class="err">' + escapeHtml(pv.error || "failed") + "</span>";
        }
        if (pv.status !== "ok" || !pv.validation) {
            return "";
        }
        var v = pv.validation;
        if (!v.checked) {
            return '<span class="pass">accepted</span> <span class="note">(validation off)</span>';
        }
        var parts = [];
        if (v.min_similarity !== null && v.min_similarity !== undefined) {
            parts.push("min similarity " + v.min_similarity.toFixed(2));
        }
        if (v.bbox_drift !== null && v.bbox_drift !== undefined) {
            parts.push("drift " + pct(v.bbox_drift));
        }
        if (v.global_shift) {
            parts.push("scene shift " + pct(v.global_shift));
        }
        parts.push(v.boxes_checked + " box(es) checked" +
                   (v.boxes_skipped ? ", " + v.boxes_skipped + " too small" : ""));
        var head = pv.accepted
            ? '<span class="pass">✓ accepted</span>'
            : '<span class="fail">✗ rejected</span> — ' + escapeHtml(v.reason || "");
        var time = pv.edit_ms ? " · " + (pv.edit_ms / 1000).toFixed(1) + " s" : "";
        return head + '<br><span class="note">' + parts.join(" · ") + time + "</span>";
    }

    function escapeHtml(text) {
        var div = document.createElement("div");
        div.textContent = text;
        return div.innerHTML;
    }

    function previewSrc(pv, figure) {
        var boxes = showBoxes() ? "&boxes=1" : "";
        if (viewMode() === "original") {
            var session = new URLSearchParams(stateUrl.split("?")[1]).get("session");
            return imageUrl + "?session=" + session + "&source=" +
                encodeURIComponent(figure.dataset.source) + boxes;
        }
        return pv.url ? pv.url + boxes : null;
    }

    function renderPreview(figure, pv) {
        var frame = figure.querySelector(".frame");
        var src = previewSrc(pv, figure);
        if (src) {
            var img = frame.querySelector("img");
            if (!img) {
                frame.innerHTML = "";
                img = document.createElement("img");
                img.addEventListener("click", function () { window.open(img.src, "_blank"); });
                frame.appendChild(img);
            }
            if (img.getAttribute("src") !== src) {
                img.setAttribute("src", src);
            }
        } else {
            frame.innerHTML = '<span class="wait">' +
                (pv.status === "error" ? "failed" : escapeHtml(pv.status) + "…") + "</span>";
        }
        figure.querySelector("[data-role=verdict]").innerHTML = verdictHtml(pv);
    }

    function render(state) {
        lastState = state;
        var busy = false;
        state.prompts.forEach(function (prompt) {
            var card = document.getElementById("p" + prompt.id);
            if (!card) {
                return;
            }
            var badge = card.querySelector("[data-role=status]");
            badge.textContent = prompt.status;
            badge.className = "badge b-" + prompt.status;
            card.querySelector("[data-role=error]").textContent = prompt.error || "";
            var judged = 0, passed = 0;
            prompt.previews.forEach(function (pv) {
                if (pv.status === "queued" || pv.status === "running") {
                    busy = true;
                }
                if (pv.accepted !== null && pv.accepted !== undefined) {
                    judged += 1;
                    passed += pv.accepted ? 1 : 0;
                }
                var figure = card.querySelector('[data-preview-id="' + pv.id + '"]');
                if (figure) {
                    renderPreview(figure, pv);
                }
            });
            card.querySelector("[data-role=summary]").textContent =
                judged ? passed + "/" + judged + " pass validation" : "";
            if (prompt.status === "queued" || prompt.status === "running") {
                busy = true;
            }
        });
        return busy;
    }

    function poll() {
        fetch(stateUrl, { credentials: "same-origin" })
            .then(function (resp) { return resp.json(); })
            .then(function (state) {
                if (render(state)) {
                    setTimeout(poll, POLL_MS);
                }
            })
            .catch(function () { setTimeout(poll, POLL_MS * 3); });
    }

    function checkBackend() {
        var banner = document.getElementById("backend-status");
        fetch(root.dataset.backendUrl, { credentials: "same-origin" })
            .then(function (resp) { return resp.json(); })
            .then(function (data) {
                if (!data.ok) {
                    banner.className = "banner bad";
                    banner.textContent = "genaug-backend is not reachable: " + data.error +
                        " — start it with `docker compose up -d genaug-backend`. " +
                        "The mock editor needs it too.";
                    return;
                }
                var gpu = data.gpu || {};
                var text = "genaug-backend up · " + (gpu.device || "?");
                if (gpu.free_gib !== undefined) {
                    text += " · " + gpu.free_gib + " / " + gpu.total_gib + " GiB free";
                }
                text += data.loaded ? " · warm: " + data.loaded.label : " · no model loaded";
                banner.className = "banner ok";
                if (!data.loaded && gpu.free_gib !== undefined && gpu.free_gib < 14) {
                    banner.className = "banner";
                    text += " — FireRed needs ~14 GiB free to load; it will wait while the GPU is busy.";
                }
                banner.textContent = text;
            })
            .catch(function () {
                banner.className = "banner bad";
                banner.textContent = "Could not check genaug-backend.";
            });
    }

    function wireKeep() {
        document.querySelectorAll("[data-role=keep]").forEach(function (box) {
            box.addEventListener("change", function () {
                var card = box.closest(".card");
                var body = new FormData();
                body.append("prompt", card.dataset.promptId);
                body.append("action", box.checked ? "keep" : "drop");
                fetch(root.dataset.promptUrl, {
                    method: "POST", body: body, credentials: "same-origin",
                    headers: { "X-CSRFToken": csrfToken(), "X-Requested-With": "fetch" },
                }).then(function () {
                    card.classList.toggle("dropped", !box.checked);
                });
            });
        });
    }

    function wirePresets() {
        var select = document.getElementById("preset");
        var dataEl = document.getElementById("presets-data");
        if (!select || !dataEl) {
            return;
        }
        var presets = JSON.parse(dataEl.textContent);
        select.addEventListener("change", function () {
            var preset = presets[select.value];
            if (!preset) {
                return;
            }
            document.getElementById("text").value = preset.text;
            document.getElementById("negative_prompt").value = preset.negative || "";
            document.getElementById("name").value = preset.name;
        });
    }

    function wireEstimate() {
        var form = document.getElementById("build-form");
        var out = document.getElementById("estimate");
        if (!form || !out) {
            return;
        }
        var images = parseInt(root.dataset.nImages, 10) || 0;
        var avg = parseFloat(root.dataset.avgEditS);
        function update() {
            var fraction = parseFloat(form.fraction.value) || 0;
            var variants = parseInt(form.variants_per_image.value, 10) || 0;
            var selected = Math.round(fraction * images);
            var total = selected * variants;
            var text = "≈ " + selected + " of " + images + " images × " + variants +
                " = " + total + " variants";
            if (avg) {
                var hours = total * avg / 3600;
                text += " · ≈ " + (hours < 1 ? Math.round(hours * 60) + " min" : hours.toFixed(1) + " h") +
                    " of GPU at the preview speed so far (cached variants are free)";
            }
            out.textContent = text + ". Class weights skew which images, not how many.";
        }
        form.addEventListener("input", update);
        update();
    }

    document.querySelectorAll("input[name=view], #show-boxes").forEach(function (el) {
        el.addEventListener("change", function () {
            if (lastState) {
                render(lastState);
            }
        });
    });

    wireKeep();
    wirePresets();
    wireEstimate();
    checkBackend();
    poll();
})();
