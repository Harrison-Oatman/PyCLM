// Method zoo: filter buttons above the thumbnail grid
// (cards are written by documentation/ext/zoo_gallery.py with data-* attributes).
document.addEventListener("DOMContentLoaded", function () {
  var grid = document.querySelector(".zoo-grid");
  if (!grid) return;
  var cards = Array.prototype.slice.call(grid.querySelectorAll(".zoo-card"));

  var filters = [
    { label: "All", test: function () { return true; } },
    { label: "Open loop", test: function (c) { return c.dataset.loop === "open"; } },
    { label: "Closed loop", test: function (c) { return c.dataset.loop === "closed"; } },
    { label: "Changes over time", test: function (c) { return c.dataset.varies === "yes"; } },
  ];
  var seen = {};
  cards.forEach(function (c) {
    var key = c.dataset.family;
    if (seen[key]) return;
    seen[key] = true;
    filters.push({
      label: c.dataset.familyTitle,
      family: true,
      test: function (card) { return card.dataset.family === key; },
    });
  });

  var bar = document.createElement("div");
  bar.className = "zoo-filters";
  bar.setAttribute("role", "toolbar");
  bar.setAttribute("aria-label", "Filter the methods");
  var buttons = filters.map(function (f, i) {
    var b = document.createElement("button");
    b.type = "button";
    b.textContent = f.label;
    if (f.family) b.className = "zoo-filter-family";
    b.setAttribute("aria-pressed", i === 0 ? "true" : "false");
    b.addEventListener("click", function () {
      buttons.forEach(function (other) { other.setAttribute("aria-pressed", "false"); });
      b.setAttribute("aria-pressed", "true");
      cards.forEach(function (c) { c.hidden = !f.test(c); });
    });
    bar.appendChild(b);
    return b;
  });
  grid.parentNode.insertBefore(bar, grid);
});
