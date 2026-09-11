/* Live progress for a dataset inference run.
 *
 * The same polling shape as `vlm/dataset_run.js`, and for the same reasons —
 * this project is WSGI-only, so there is nothing else available, and the poll is
 * NOT a heartbeat: the worker keeps measuring whether or not this page is open,
 * so closing the tab loses the live view and nothing else.
 *
 * Rows are appended client-side (an image's numbers are the row, and there is
 * nothing to derive from them), but the *summary* is not: when the run goes
 * terminal the page reloads once and takes the server-rendered percentile,
 * stage and geometry tables. Rebuilding those in JS would be two
 * implementations of the same statistics, and they are the numbers the run
 * exists to produce.
 */
(function () {
  "use strict";

  function readJson(id, fallback) {
    var el = document.getElementById(id);
    if (!el) return fallback;
    try {
      return JSON.parse(el.textContent);
    } catch (err) {
      return fallback;
    }
  }

  var progressUrl = readJson("dir-progress-url", null);
  var imageUrl = readJson("dir-image-url", null);
  var lastSeq = readJson("dir-last-seq", 0);
  var isTerminal = readJson("dir-is-terminal", false);
  var liveAppend = readJson("dir-live-append", true);
  var rowLimit = readJson("dir-row-limit", 300);

  var tbody = document.getElementById("dir-rows");
  var emptyEl = document.getElementById("dir-empty");
  var statusEl = document.getElementById("dir-status");
  var progressEl = document.getElementById("dir-progress");
  var errorsEl = document.getElementById("dir-errors");
  var detectionsEl = document.getElementById("dir-detections");
  var fpsEl = document.getElementById("dir-fps");
  var p50El = document.getElementById("dir-p50");

  var POLL_MS = 2000;

  function cell(text, className) {
    var td = document.createElement("td");
    if (className) td.className = className;
    td.textContent = text;
    return td;
  }

  function ms(value) {
    // A null latency is an image that failed, or a trainer that reports no
    // timing of its own — an em dash, never a 0 that would read as instant.
    return value == null ? "—" : value.toFixed(1);
  }

  function appendRow(row) {
    var tr = document.createElement("tr");
    tr.dataset.seq = row.seq;
    tr.appendChild(cell(String(row.seq), "seq"));

    var thumb = document.createElement("td");
    thumb.className = "thumb";
    // Addressed by result row, so the server resolves the file with one stat
    // rather than re-listing the dataset directory per image. The link adds
    // &boxes=1: the click-through is annotated, the thumbnail is a small
    // re-encode of the plain file, so a full page of rows costs no redrawing.
    var link = document.createElement("a");
    link.href = imageUrl + "?result=" + row.pk + "&boxes=1";
    link.target = "_blank";
    link.title = "Open full size with the model's boxes drawn on";
    var img = document.createElement("img");
    img.loading = "lazy";
    img.src = imageUrl + "?result=" + row.pk;
    img.alt = row.image;
    link.appendChild(img);
    thumb.appendChild(link);
    tr.appendChild(thumb);

    var file = cell(row.image);
    if (row.error) {
      var err = document.createElement("div");
      err.className = "row-error";
      err.textContent = row.error;
      file.appendChild(err);
    }
    tr.appendChild(file);

    tr.appendChild(cell(String(row.detections), "num"));
    tr.appendChild(cell(ms(row.model_ms), "num"));
    tr.appendChild(cell(ms(row.request_ms), "num"));

    tbody.appendChild(tr);
  }

  function renderRun(run) {
    if (statusEl) statusEl.textContent = run.status;
    if (progressEl) {
      progressEl.textContent = run.images_total
        ? run.images_processed + "/" + run.images_total
        : String(run.images_processed);
    }
    if (errorsEl) errorsEl.textContent = String(run.errors_count);
    if (detectionsEl) detectionsEl.textContent = String(run.detections_total);
    // The rolling summary is refreshed every N images by the worker, so these
    // two tiles move in steps rather than per image — the tables below them are
    // filled in on the reload once the run is done.
    if (fpsEl && run.fps != null) fpsEl.textContent = run.fps.toFixed(1);
    if (p50El && run.p50_ms != null) p50El.textContent = run.p50_ms.toFixed(1);
  }

  function poll() {
    fetch(progressUrl + "&after=" + lastSeq, {
      credentials: "same-origin",
      headers: { "X-Requested-With": "XMLHttpRequest" },
    })
      .then(function (resp) {
        if (!resp.ok) throw new Error("progress poll failed: " + resp.status);
        return resp.json();
      })
      .then(function (data) {
        if (data.results.length) {
          data.results.forEach(function (row) {
            if (liveAppend && tbody && tbody.children.length < rowLimit) {
              appendRow(row);
              if (emptyEl) emptyEl.hidden = true;
            }
            lastSeq = Math.max(lastSeq, row.seq);
          });
        }
        renderRun(data.run);
        if (data.run.terminal) {
          // The finished run's statistics are server-rendered; ask for them once.
          window.location.reload();
          return;
        }
        window.setTimeout(poll, POLL_MS);
      })
      .catch(function () {
        // A transient failure must not end the live view — back off and retry.
        window.setTimeout(poll, POLL_MS * 3);
      });
  }

  if (!isTerminal && progressUrl) {
    poll();
  }
})();
