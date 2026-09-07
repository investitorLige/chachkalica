/* Live progress for a VLM dataset run.
 *
 * The same polling shape as `run_live.js` — this project is WSGI-only, so there
 * is nothing else available — with two differences that matter:
 *
 *  * This poll is NOT a heartbeat. A dataset run is a measurement you start and
 *    come back to, so the worker keeps going whether or not this page is open;
 *    closing the tab loses the live view and nothing else.
 *  * When the run goes terminal the page reloads once, rather than trying to
 *    build the analytics client-side. The per-class table, the confusion matrix
 *    and the re-grade form are all server-rendered, and duplicating them in JS
 *    would be two implementations of the same numbers.
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

  var progressUrl = readJson("vds-progress-url", null);
  var imageUrl = readJson("vds-image-url", null);
  var lastSeq = readJson("vds-last-seq", 0);
  var isTerminal = readJson("vds-is-terminal", false);
  var liveAppend = readJson("vds-live-append", true);
  var rowLimit = readJson("vds-row-limit", 300);

  var tbody = document.getElementById("vds-rows");
  var emptyEl = document.getElementById("vds-empty");
  var statusEl = document.getElementById("vds-status");
  var progressEl = document.getElementById("vds-progress");
  var errorsEl = document.getElementById("vds-errors");
  var accuracyEl = document.getElementById("vds-accuracy");

  var POLL_MS = 2000;
  var graded = tbody && tbody.dataset.graded === "1";

  function cell(text, className) {
    var td = document.createElement("td");
    if (className) td.className = className;
    td.textContent = text;
    return td;
  }

  function chips(names, extraClass) {
    var td = document.createElement("td");
    if (!names || !names.length) {
      td.appendChild(cellSpan("—", "verdict-na"));
      return td;
    }
    names.forEach(function (name) {
      td.appendChild(cellSpan(name, "cls" + (extraClass ? " " + extraClass : "")));
    });
    return td;
  }

  function cellSpan(text, className) {
    var span = document.createElement("span");
    span.className = className;
    span.textContent = text;
    return span;
  }

  function appendRow(row) {
    var tr = document.createElement("tr");
    tr.dataset.seq = row.seq;
    tr.appendChild(cell(String(row.seq), "seq"));

    var thumb = document.createElement("td");
    thumb.className = "thumb";
    // Thumbnails are addressed by result row, so the server resolves the file
    // with one stat rather than re-listing the dataset directory per image.
    // The link adds &labels=1: the click-through is annotated, the thumbnail is
    // the plain file, so a full page of rows costs no redrawing.
    var link = document.createElement("a");
    link.href = imageUrl + "?result=" + row.pk + "&labels=1";
    link.target = "_blank";
    link.title = row.has_label
      ? "Open with its labels drawn on"
      : "No label file — opens the plain image";
    var img = document.createElement("img");
    img.loading = "lazy";
    img.src = imageUrl + "?result=" + row.pk;
    img.alt = row.image;
    link.appendChild(img);
    thumb.appendChild(link);
    tr.appendChild(thumb);

    tr.appendChild(cell(row.image));

    if (graded) {
      if (!row.has_label) {
        var unlabeled = document.createElement("td");
        unlabeled.appendChild(cellSpan("unlabeled", "verdict-na"));
        tr.appendChild(unlabeled);
      } else if (!row.gt.length) {
        var nothing = document.createElement("td");
        nothing.appendChild(cellSpan("nothing", "verdict-na"));
        tr.appendChild(nothing);
      } else {
        tr.appendChild(chips(row.gt, "gt"));
      }
      tr.appendChild(chips(row.predicted));

      var verdict = document.createElement("td");
      if (row.matched === null) verdict.appendChild(cellSpan("—", "verdict-na"));
      else if (row.matched) verdict.appendChild(cellSpan("✓", "verdict-ok"));
      else verdict.appendChild(cellSpan("✗", "verdict-bad"));
      tr.appendChild(verdict);
    }

    var answer = cell(row.text || "", "answer");
    if (row.error) {
      var err = document.createElement("div");
      err.className = "row-error";
      err.textContent = row.error;
      answer.appendChild(err);
    }
    tr.appendChild(answer);
    tr.appendChild(cell(row.latency_ms == null ? "—" : String(row.latency_ms)));

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
    if (accuracyEl && run.exact_match_accuracy != null) {
      accuracyEl.textContent = run.exact_match_accuracy.toFixed(3);
    }
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
          // The finished run's analytics are server-rendered; ask for them once.
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
