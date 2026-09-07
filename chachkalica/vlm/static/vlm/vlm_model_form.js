/* Narrow the weights dropdown to the selected VLM family.
 *
 * Same technique as training/experiment_model_form.js: the server renders every
 * option once, tagged with data-family, and this hides the ones that belong to
 * other families. Options for weights missing from the offline HF cache are
 * rendered disabled — they cannot load on this network — and the "Custom…"
 * entry reveals a free-text field.
 */
(function () {
  "use strict";

  function ready(fn) {
    if (document.readyState !== "loading") fn();
    else document.addEventListener("DOMContentLoaded", fn);
  }

  ready(function () {
    var family = document.getElementById("id_family");
    var weights = document.getElementById("id_weights_choice");
    var custom = document.getElementById("id_custom_weights");
    if (!family || !weights) return;

    var customRow = custom ? custom.closest(".form-row") : null;

    function syncCustom() {
      if (!customRow) return;
      customRow.style.display = weights.value === "__custom__" ? "" : "none";
    }

    function syncFamily() {
      var selected = family.value;
      var current = weights.value;
      var currentStillVisible = false;

      Array.prototype.forEach.call(weights.options, function (option) {
        var optionFamily = option.dataset.family;
        // Options with no family (the placeholder, "Custom…") always show.
        var visible = !optionFamily || optionFamily === selected;
        option.hidden = !visible;
        if (visible && option.value === current) currentStillVisible = true;
      });

      // Switching family away from the selected checkpoint would otherwise
      // leave a hidden option selected, which saves the wrong weights.
      if (!currentStillVisible) {
        weights.value = "";
      }
      syncCustom();
    }

    family.addEventListener("change", syncFamily);
    weights.addEventListener("change", syncCustom);
    syncFamily();
  });
})();
