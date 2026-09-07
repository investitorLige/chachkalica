/* Live alert rail for a VLM run.
 *
 * There are no websockets in this project — it is WSGI-only, with no Channels —
 * so new alerts arrive by polling a small JSON endpoint once a second. That
 * same request is the "someone is watching" heartbeat the worker checks: stop
 * polling and the run pauses, which is exactly what should happen when the tab
 * is closed.
 *
 * The cursor is the per-run `seq`, seeded from what the server already rendered,
 * so a refresh never replays alerts that are already on the page.
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

  var alertsUrl = readJson("vlm-alerts-url", null);
  var cancelUrl = readJson("vlm-cancel-url", null);
  var lastSeq = readJson("vlm-last-seq", 0);
  var isTerminal = readJson("vlm-is-terminal", false);

  var rail = document.getElementById("vlm-rail");
  var railEmpty = document.getElementById("vlm-rail-empty");
  var statusEl = document.getElementById("vlm-status");
  var progressEl = document.getElementById("vlm-progress");
  var errorEl = document.getElementById("vlm-error");
  var catchupEl = document.getElementById("vlm-catchup");
  var cancelBtn = document.getElementById("vlm-cancel");
  var video = document.getElementById("vlm-video");

  var POLL_MS = 1000;
  var videoEnded = false;

  function csrfToken() {
    var match = document.cookie.match(/(?:^|;\s*)csrftoken=([^;]*)/);
    return match ? decodeURIComponent(match[1]) : "";
  }

  function appendAlert(alert) {
    var li = document.createElement("li");
    li.dataset.seq = alert.seq;

    var time = document.createElement("span");
    time.className = "vlm-t";
    time.textContent = alert.t.toFixed(1) + "s";

    var text = document.createElement("span");
    text.textContent = alert.text || "(empty answer)";

    li.appendChild(time);
    li.appendChild(text);

    if (alert.latency_ms) {
      var lat = document.createElement("div");
      lat.className = "vlm-lat";
      lat.textContent = alert.latency_ms + " ms";
      li.appendChild(lat);
    }

    // Only follow the feed when the reader is already at the bottom, so
    // scrolling back to re-read something isn't yanked away.
    var pinned = rail.scrollTop + rail.clientHeight >= rail.scrollHeight - 4;
    rail.appendChild(li);
    if (pinned) rail.scrollTop = rail.scrollHeight;
  }

  function renderRun(run) {
    if (statusEl) statusEl.textContent = run.status;
    if (progressEl) {
      progressEl.textContent = run.frames_total
        ? run.frames_processed + "/" + run.frames_total
        : String(run.frames_processed);
    }
    if (errorEl) {
      errorEl.textContent = run.last_error || "";
      errorEl.hidden = !run.last_error;
    }
    if (catchupEl) {
      catchupEl.hidden = !(videoEnded && !run.terminal);
    }
    if (run.terminal && cancelBtn) cancelBtn.hidden = true;
  }

  function poll() {
    fetch(alertsUrl + "&after=" + lastSeq, {
      credentials: "same-origin",
      headers: { "X-Requested-With": "XMLHttpRequest" },
    })
      .then(function (resp) {
        if (!resp.ok) throw new Error("alerts poll failed: " + resp.status);
        return resp.json();
      })
      .then(function (data) {
        if (data.alerts.length) {
          if (railEmpty) railEmpty.hidden = true;
          data.alerts.forEach(function (alert) {
            appendAlert(alert);
            lastSeq = Math.max(lastSeq, alert.seq);
          });
        }
        renderRun(data.run);
        // Keep polling once more after the run goes terminal so the last batch
        // of alerts lands, then stop.
        if (!data.run.terminal) {
          window.setTimeout(poll, POLL_MS);
        }
      })
      .catch(function () {
        // A transient failure (a redeploy, a dropped connection) must not kill
        // the feed — back off a little and try again.
        window.setTimeout(poll, POLL_MS * 3);
      });
  }

  if (video) {
    video.addEventListener("ended", function () {
      videoEnded = true;
      if (catchupEl && !isTerminal) catchupEl.hidden = false;
    });
  }

  if (cancelBtn && cancelUrl) {
    cancelBtn.addEventListener("click", function () {
      cancelBtn.disabled = true;
      fetch(cancelUrl, {
        method: "POST",
        credentials: "same-origin",
        headers: { "X-CSRFToken": csrfToken() },
      }).then(function () {
        if (statusEl) statusEl.textContent = "cancel requested";
      });
    });
  }

  // A finished run was rendered whole by the server; there is nothing to poll
  // for, and polling would only keep a pointless heartbeat alive.
  if (!isTerminal && alertsUrl) {
    poll();
  }
})();
