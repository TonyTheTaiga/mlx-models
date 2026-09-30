// The model-builder report: Shortcuts and Internals, rendered from analysis.json (analyze.py).
//
// Charts are plain SVG. Colors come from CSS custom properties validated for the dark surface:
// --s1..--s3 (categorical), --seq-* (one-hue sequential ramp), --status-* (flags, always with an
// icon and a label). Names from the data go in via textContent.

const NS = "http://www.w3.org/2000/svg";
const pct = (v, d = 0) => `${(v * 100).toFixed(d)}%`;
const pts = (v, d = 1) => `${v >= 0 ? "+" : "−"}${Math.abs(v * 100).toFixed(d)}`;
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v);
  }
  for (const c of children.flat()) if (c != null) el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return el;
}
function s(tag, attrs = {}, parent) {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  if (parent) parent.appendChild(el);
  return el;
}

// One-hue sequential ramp (dark → light on the dark surface), t in [0, 1].
function seq(t) {
  const stops = ["--seq-1", "--seq-2", "--seq-3", "--seq-4", "--seq-5"].map((n) => hex(css(n)));
  const x = Math.max(0, Math.min(1, t)) * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(x)), f = x - i;
  const c = stops[i].map((a, k) => Math.round(a + (stops[i + 1][k] - a) * f));
  return { fill: `rgb(${c})`, ink: t > 0.62 ? "var(--ink)" : "var(--paper)" };
}
function hex(v) {
  const m = v.replace("#", "");
  return [0, 2, 4].map((i) => parseInt(m.slice(i, i + 2), 16));
}

// ---------- tooltip (shared) ----------
let tip;
function showTip(e, rows) {
  tip = tip || document.body.appendChild(h("div", { class: "rtip", role: "tooltip" }));
  tip.replaceChildren(...rows.map(([value, label, key]) =>
    h("div", { class: "row" }, key ? h("span", { class: "key", style: `background:${key}` }) : null, h("b", {}, value), h("span", {}, label))));
  tip.hidden = false;
  const w = tip.offsetWidth, x = e.clientX + 14 + w > innerWidth ? e.clientX - w - 14 : e.clientX + 14;
  tip.style.left = `${x}px`;
  tip.style.top = `${e.clientY + 14}px`;
}
const hideTip = () => tip && (tip.hidden = true);
function hover(el, rows) {
  el.setAttribute("tabindex", "0");
  el.addEventListener("pointermove", (e) => showTip(e, rows()));
  el.addEventListener("pointerleave", hideTip);
  el.addEventListener("focus", () => { const r = el.getBoundingClientRect(); showTip({ clientX: r.right, clientY: r.top }, rows()); });
  el.addEventListener("blur", hideTip);
}

function dataTable(headers, rows) {
  return h("details", { class: "data" }, h("summary", {}, "data"),
    h("div", { class: "scroll" }, h("table", {},
      h("thead", {}, h("tr", {}, headers.map((x) => h("th", {}, x)))),
      h("tbody", {}, rows.map((r) => h("tr", {}, r.map((x) => h("td", {}, x))))))));
}

const tile = (value, label, note) => h("div", { class: "tile" }, h("div", { class: "v" }, value), h("div", { class: "l" }, label), note ? h("div", { class: "n" }, note) : null);
const flag = (level, text) => h("span", { class: `flag ${level}` }, level === "warn" ? "⚠ " : level === "good" ? "✓ " : "● ", text);

// ---------- Shortcuts ----------
// Retained = (accuracy − false alarms) under a test ÷ the same on clean images: how much of the
// model's above-chance ability to pick out this class survives the change.
function retained(A, test, c) {
  const base = A.accuracy.per_class[c] - A.accuracy.false_alarm[c];
  const t = A.tests[test];
  return (t.per_class[c] - t.false_alarm[c]) / Math.max(1e-6, base);
}

export function renderShortcuts(root, A, api) {
  const C = A.classes, tests = Object.keys(A.tests), T = A.tests;
  root.replaceChildren();
  root.append(
    h("header", { class: "rhead" },
      h("h2", {}, "Is it looking at the right thing?"),
      h("p", {}, `Controlled tests on all ${A.n_test.toLocaleString()} test images, and occlusion evidence on ${A.evidence.n_images.toLocaleString()} of them. Checkpoint: `, h("code", {}, A.checkpoint), ".")),
    h("div", { class: "tiles" },
      tile(pct(A.accuracy.overall, 1), "test accuracy"),
      tile(pct(T.border_only.overall, 1), "from the border alone", "chance is 10%"),
      tile(pts(T.grayscale.overall - A.accuracy.overall), "points lost without color"),
      tile(pct(T.shuffled.overall, 1), "with patches shuffled", "global shape destroyed")));

  // Class × test table, shaded by retained ability, with flags.
  const flagsFor = (c) => {
    const f = [];
    if (retained(A, "border_only", c) >= 0.5) f.push(flag("warn", "background shortcut"));
    if (retained(A, "grayscale", c) < 0.55) f.push(flag("warn", "color-reliant"));
    if (retained(A, "shuffled", c) >= 0.4) f.push(flag("warn", "texture over shape"));
    return f;
  };
  const table = h("table", { class: "heat" },
    h("thead", {}, h("tr", {}, h("th", {}, "class"), h("th", { class: "num" }, "clean"),
      tests.map((t) => h("th", { class: "num", title: T[t].description }, T[t].name)), h("th", {}, "flags"))),
    h("tbody", {}, C.map((name, c) => h("tr", {},
      h("th", {}, name),
      h("td", { class: "num" }, pct(A.accuracy.per_class[c])),
      tests.map((t) => {
        const r = retained(A, t, c), col = seq(r);
        const td = h("td", { class: "cell", style: `background:${col.fill};color:${col.ink}` }, pct(Math.max(0, r)));
        hover(td, () => [
          [pct(Math.max(0, r)), `of ${name}'s above-chance accuracy kept · ${T[t].name.toLowerCase()}`],
          [pct(T[t].per_class[c]), `of ${name} images still called ${name}`],
          [pct(T[t].false_alarm[c]), `of other images called ${name} under this test`],
        ]);
        return td;
      }),
      h("td", { class: "flags" }, flagsFor(c))))));
  root.append(h("section", { class: "card" },
    h("h3", {}, "How much of each class survives each test"),
    h("p", { class: "sub" }, "Each cell: how much of the class's above-chance accuracy (accuracy minus how often other images get that label) survives the change. High under ", h("i", {}, "border only"), " or ", h("i", {}, "patches shuffled"), " means the model doesn't need the object or its shape; low under ", h("i", {}, "grayscale"), " means it leans on color."),
    h("div", { class: "scroll" }, table),
    h("div", { class: "scale" }, h("span", {}, "0%"), h("span", { class: "bar" }), h("span", {}, "100% kept")),
    h("ul", { class: "defs" }, tests.map((t) => h("li", {}, h("b", {}, T[t].name), ": ", T[t].description)))));

  // Average evidence maps per class.
  const E = A.evidence, g = Math.round(Math.sqrt(E.per_class[0].map.length));
  const detail = h("div", { class: "examples" });
  const cards = C.map((name, c) => {
    const pc = E.per_class[c], max = Math.max(...pc.map) || 1;
    const size = 96, cell = size / g;
    const svgEl = s("svg", { viewBox: `0 0 ${size} ${size}`, width: size, height: size, role: "img", "aria-label": `average evidence map for ${name}` });
    pc.map.forEach((v, p) => s("rect", { x: (p % g) * cell + 1, y: Math.floor(p / g) * cell + 1, width: cell - 2, height: cell - 2, rx: 1, fill: seq(v / max).fill }, svgEl));
    s("rect", { x: 0.5, y: 0.5, width: size - 1, height: size - 1, fill: "none", stroke: "var(--line-strong)", "stroke-dasharray": "0", "stroke-width": 1 }, svgEl);
    const b = pc.border_share;
    const card = h("button", { class: "evmap", onclick: () => select(c) },
      svgEl, h("div", { class: "name" }, name),
      h("div", { class: "note" }, b == null ? "—" : `${pct(b)} on the border`),
      b != null && b > E.border_baseline ? flag("warn", "above uniform") : null);
    hover(card, () => [[b == null ? "—" : pct(b), `of ${name}'s evidence sits on the outer ring`], [pct(E.border_baseline), "if evidence were spread evenly"], [String(pc.n_correct), "correct images averaged"]]);
    return card;
  });
  function select(c) {
    cards.forEach((el, i) => el.classList.toggle("on", i === c));
    const ex = A.examples.filter((e) => e.label === c);
    detail.replaceChildren(
      h("h4", {}, `${C[c]}: the most border-reliant correct images, and the most centered`),
      h("div", { class: "strip" }, ex.map((e) => {
        const cv = h("canvas", { width: 32, height: 32 });
        api.paintExample(cv, e.slot);
        return h("figure", {}, cv,
          h("figcaption", {}, `${e.kind === "border" ? "border-reliant" : "centered"} · ${pct(e.border_share)} border`),
          h("button", { onclick: () => api.inspectExample(e.slot, e.label) }, "inspect →"));
      })));
  }
  root.append(h("section", { class: "card" },
    h("h3", {}, "Where the evidence sits, per class"),
    h("p", { class: "sub" }, `Average occlusion evidence for the predicted class over correctly classified images (brighter = hiding that patch hurts more). The outer ring is ${pct(E.border_baseline)} of the image; overall, ${pct(E.border_share_overall)} of the evidence lands there. Click a class for examples.`),
    h("div", { class: "evmaps" }, cards), detail,
    dataTable(["class", "border share", "border share (mistakes)", "images"], C.map((name, c) => [name, E.per_class[c].border_share == null ? "—" : pct(E.per_class[c].border_share), E.per_class[c].border_share_wrong == null ? "—" : pct(E.per_class[c].border_share_wrong), E.per_class[c].n_correct]))));
  const worst = C.map((_, c) => c).sort((a, b) => retained(A, "border_only", b) - retained(A, "border_only", a))[0];
  select(worst);
}

// ---------- Internals ----------
function lineChart(W, { xs, series, yMax = 1, yLabel, fmt = pct }) {
  const H = 220, m = { l: 40, r: 110, t: 12, b: 26 }, iw = W - m.l - m.r, ih = H - m.t - m.b;
  const X = (i) => m.l + (i / (xs.length - 1)) * iw, Y = (v) => m.t + (1 - v / yMax) * ih;
  const svgEl = s("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img", "aria-label": yLabel });
  for (let k = 0; k <= 4; k++) {
    const v = (yMax * k) / 4;
    s("line", { x1: m.l, x2: m.l + iw, y1: Y(v), y2: Y(v), stroke: "var(--grid)", "stroke-width": 1 }, svgEl);
    s("text", { x: m.l - 6, y: Y(v) + 3, "text-anchor": "end", class: "tick" }, svgEl).textContent = fmt(v);
  }
  xs.forEach((x, i) => (s("text", { x: X(i), y: H - 8, "text-anchor": "middle", class: "tick" }, svgEl).textContent = x));
  for (const se of series) {
    s("polyline", { points: se.values.map((v, i) => `${X(i)},${Y(v)}`).join(" "), fill: "none", stroke: se.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }, svgEl);
    se.values.forEach((v, i) => s("circle", { cx: X(i), cy: Y(v), r: 4, fill: se.color, stroke: "var(--ink)", "stroke-width": 2 }, svgEl));
  }
  // End labels, pushed apart so lines that finish together stay readable.
  const last = xs.length - 1, gap = 13;
  const ends = series.map((se) => ({ se, y: Y(se.values[last]) })).sort((a, b) => a.y - b.y);
  for (let i = 1; i < ends.length; i++) ends[i].y = Math.max(ends[i].y, ends[i - 1].y + gap);
  for (const { se, y } of ends)
    s("text", { x: X(last) + 10, y: y + 3, class: "dlabel" }, svgEl).textContent = `${se.name} ${fmt(se.values[last])}`;
  // Crosshair snapping to the nearest depth, with every series in one readout.
  const cross = s("line", { y1: m.t, y2: m.t + ih, stroke: "var(--muted)", "stroke-width": 1, visibility: "hidden" }, svgEl);
  const hit = s("rect", { x: m.l - 10, y: m.t, width: iw + 20, height: ih, fill: "transparent" }, svgEl);
  hit.addEventListener("pointermove", (e) => {
    const r = svgEl.getBoundingClientRect(), px = ((e.clientX - r.left) / r.width) * W;
    const i = Math.max(0, Math.min(xs.length - 1, Math.round(((px - m.l) / iw) * (xs.length - 1))));
    cross.setAttribute("x1", X(i)); cross.setAttribute("x2", X(i)); cross.setAttribute("visibility", "visible");
    showTip(e, [[xs[i], ""], ...series.map((se) => [fmt(se.values[i]), se.name, se.color])]);
  });
  hit.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); hideTip(); });
  return svgEl;
}

function barChart(W, { labels, values, fmt, emphasis = -1, aria }) {
  const H = 200, m = { l: 36, r: 12, t: 22, b: 26 }, iw = W - m.l - m.r, ih = H - m.t - m.b;
  const max = Math.max(...values) * 1.1 || 1, band = iw / labels.length, bw = Math.min(24, band * 0.6);
  const Y = (v) => m.t + (1 - v / max) * ih;
  const svgEl = s("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img", "aria-label": aria });
  s("line", { x1: m.l, x2: m.l + iw, y1: Y(0), y2: Y(0), stroke: "var(--grid)", "stroke-width": 1 }, svgEl);
  values.forEach((v, i) => {
    const x = m.l + band * i + (band - bw) / 2, y = Y(v), hgt = Y(0) - y;
    // 4px rounded data-end, square at the baseline.
    const r = Math.min(4, hgt / 2, bw / 2);
    s("path", { d: `M${x},${Y(0)} V${y + r} Q${x},${y} ${x + r},${y} H${x + bw - r} Q${x + bw},${y} ${x + bw},${y + r} V${Y(0)} Z`, fill: i === emphasis ? "var(--s2)" : "var(--s1)" }, svgEl);
    s("text", { x: x + bw / 2, y: y - 6, "text-anchor": "middle", class: "dlabel" }, svgEl).textContent = fmt(v);
    s("text", { x: x + bw / 2, y: H - 8, "text-anchor": "middle", class: "tick" }, svgEl).textContent = labels[i];
    const hit = s("rect", { x: m.l + band * i, y: m.t, width: band, height: ih, fill: "transparent" }, svgEl);
    hover(hit, () => [[fmt(v), labels[i]]]);
  });
  return svgEl;
}

function groupedBars(W, { labels, series, fmt, aria }) {
  const H = 200, m = { l: 36, r: 12, t: 22, b: 26 }, iw = W - m.l - m.r, ih = H - m.t - m.b;
  const max = Math.max(...series.flatMap((x) => x.values)) * 1.1, band = iw / labels.length;
  const bw = Math.min(18, (band * 0.7) / series.length), Y = (v) => m.t + (1 - v / max) * ih;
  const svgEl = s("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img", "aria-label": aria });
  s("line", { x1: m.l, x2: m.l + iw, y1: Y(0), y2: Y(0), stroke: "var(--grid)", "stroke-width": 1 }, svgEl);
  labels.forEach((lab, i) => {
    const x0 = m.l + band * i + (band - bw * series.length - 2 * (series.length - 1)) / 2;
    series.forEach((se, k) => {
      const v = se.values[i], x = x0 + k * (bw + 2), y = Y(v), hgt = Y(0) - y, r = Math.min(4, hgt / 2, bw / 2);
      s("path", { d: `M${x},${Y(0)} V${y + r} Q${x},${y} ${x + r},${y} H${x + bw - r} Q${x + bw},${y} ${x + bw},${y + r} V${Y(0)} Z`, fill: se.color }, svgEl);
    });
    s("text", { x: m.l + band * i + band / 2, y: H - 8, "text-anchor": "middle", class: "tick" }, svgEl).textContent = lab;
    const hit = s("rect", { x: m.l + band * i, y: m.t, width: band, height: ih, fill: "transparent" }, svgEl);
    hover(hit, () => [[lab, ""], ...series.map((se) => [fmt(se.values[i]), se.name, se.color])]);
  });
  // Label only the extreme of each series.
  series.forEach((se) => {
    const i = se.values.indexOf(Math.max(...se.values));
    const x0 = m.l + band * i + (band - bw * series.length - 2 * (series.length - 1)) / 2 + series.indexOf(se) * (bw + 2);
    s("text", { x: x0 + bw / 2, y: Y(se.values[i]) - 6, "text-anchor": "middle", class: "dlabel" }, svgEl).textContent = fmt(se.values[i]);
  });
  return svgEl;
}

const legend = (items, kind = "line") => h("div", { class: "legend" }, items.map(([name, color]) =>
  h("span", {}, h("i", { class: kind, style: `background:${color}` }), name)));

export function renderInternals(root, A) {
  const R = A.readout, K = A.block_knockout, Hd = A.heads, Rs = A.residual;
  const W = Math.max(300, Math.min(760, root.clientWidth - 40));
  const nBlocks = K.length, nHeads = Math.max(...Hd.map((x) => x.head));
  const byBlock = (b) => Hd.filter((x) => x.block === b);
  const laterMax = Math.max(...Hd.filter((x) => x.block > 1).map((x) => -x.knockout_delta));
  const firstHeads = byBlock(1).sort((a, b) => a.knockout_delta - b.knockout_delta);
  const laterBlocks = K.slice(1).map((k) => -k.delta);

  // Findings, written from the numbers.
  const findings = [
    ["info", `Class information builds steadily: a linear probe on CLS reads ${pct(R.probe_cls[1])} after block 1 and ${pct(R.probe_cls[nBlocks])} after block ${nBlocks}; the trained head only works on the final CLS (${pct(R.head_on_cls[nBlocks - 1])} one block earlier).`],
    [-K[0].delta > 0.3 ? "warn" : "info", `Block 1 is load-bearing: skipping it costs ${pts(K[0].delta)} points. Skipping any later block costs ${pct(Math.min(...laterBlocks), 1)}–${pct(Math.max(...laterBlocks), 1)}.`],
    [laterMax < 0.02 ? "warn" : "info", `No single head in blocks 2–${nBlocks} costs more than ${pct(laterMax, 1)} when zeroed: individually they're redundant, so fewer heads or pruning is worth testing. Block 1's heads matter most (head ${firstHeads[0].head}: ${pts(firstHeads[0].knockout_delta)}).`],
    ["info", `Block 1's attention update is ${Rs[0].attn_update.toFixed(1)}× the size of its input, so it effectively writes the representation. Later MLP updates are ${Math.min(...Rs.slice(1, -1).map((r) => r.mlp_update)).toFixed(2)}–${Math.max(...Rs.slice(1, -1).map((r) => r.mlp_update)).toFixed(2)}× until block ${nBlocks} (${Rs[nBlocks - 1].mlp_update.toFixed(2)}×).`],
  ];

  const series = [
    { name: "probe on CLS", values: R.probe_cls, color: "var(--s1)" },
    { name: "head on CLS", values: R.head_on_cls, color: "var(--s2)" },
    { name: "probe on patches", values: R.probe_patch_mean, color: "var(--s3)" },
  ];

  // Head grid with a metric switch.
  const metrics = {
    knockout: { name: "knock-out cost", get: (x) => -x.knockout_delta, fmt: (v) => `${pts(-v)} pts`, note: "accuracy lost when this head's output is zeroed" },
    entropy: { name: "entropy", get: (x) => x.entropy, fmt: (v) => v.toFixed(2), note: "attention spread: 1 = uniform over all tokens, 0 = one token" },
    reach: { name: "spatial reach", get: (x) => x.reach, fmt: (v) => `${v.toFixed(1)} patches`, note: "average distance from a patch to the patches it attends to" },
    to_cls: { name: "attention to CLS", get: (x) => x.to_cls, fmt: (v) => pct(v), note: "share of patch attention spent on the CLS token" },
  };
  const gridBox = h("div", {});
  let metric = "knockout";
  const drawGrid = () => {
    const M = metrics[metric], vals = Hd.map(M.get), max = Math.max(...vals), min = Math.min(0, ...vals);
    const cw = Math.min(56, (W - 70) / nHeads), chh = 30;
    const svgEl = s("svg", { viewBox: `0 0 ${70 + cw * nHeads} ${chh * nBlocks + 24}`, width: 70 + cw * nHeads, height: chh * nBlocks + 24, role: "img", "aria-label": `${M.name} per head` });
    for (let hh = 1; hh <= nHeads; hh++) s("text", { x: 70 + (hh - 0.5) * cw, y: 12, "text-anchor": "middle", class: "tick" }, svgEl).textContent = `h${hh}`;
    for (let b = 1; b <= nBlocks; b++) {
      s("text", { x: 60, y: 24 + (b - 0.5) * chh + 4, "text-anchor": "end", class: "tick" }, svgEl).textContent = `block ${b}`;
      for (const x of byBlock(b)) {
        const v = M.get(x), c = seq((v - min) / (max - min || 1));
        const cell = s("rect", { x: 70 + (x.head - 1) * cw + 1, y: 24 + (b - 1) * chh + 1, width: cw - 2, height: chh - 2, rx: 2, fill: c.fill, class: "hcell" }, svgEl);
        hover(cell, () => [[`block ${b} · head ${x.head}`, ""], [`${pts(x.knockout_delta)} pts`, "if zeroed"], [x.entropy.toFixed(2), "entropy"], [`${x.reach.toFixed(1)}`, "patches of reach"], [pct(x.to_cls), "attention to CLS"]]);
      }
    }
    gridBox.replaceChildren(
      h("div", { class: "seg" }, Object.entries(metrics).map(([k, m]) => h("button", { class: k === metric ? "on" : "", onclick: () => { metric = k; drawGrid(); } }, m.name))),
      h("p", { class: "sub" }, M.note), svgEl,
      h("div", { class: "scale" }, h("span", {}, M.fmt(min)), h("span", { class: "bar" }), h("span", {}, M.fmt(max))));
  };
  drawGrid();

  root.replaceChildren(
    h("header", { class: "rhead" }, h("h2", {}, "Is every part of the network earning its keep?"),
      h("p", {}, `Read-outs, knock-outs and attention statistics over all ${A.n_test.toLocaleString()} test images. ${nBlocks} blocks × ${nHeads} heads, d = ${A.config.d_model}.`)),
    h("ul", { class: "findings" }, findings.map(([lvl, text]) => h("li", { class: lvl }, h("span", { class: "icon" }, lvl === "warn" ? "⚠" : "●"), text))),
    h("section", { class: "card" }, h("h3", {}, "Where class information appears"),
      h("p", { class: "sub" }, "Accuracy of a linear classifier trained on each depth's features, and of the model's own head applied early."),
      legend(series.map((x) => [x.name, x.color])),
      lineChart(W, { xs: R.depths, series, yLabel: "read-out accuracy by depth" }),
      dataTable(["depth", ...series.map((x) => x.name)], R.depths.map((d, i) => [d, ...series.map((x) => pct(x.values[i], 1))]))),
    h("section", { class: "card" }, h("h3", {}, "Accuracy lost when a block is skipped"),
      h("p", { class: "sub" }, `Points of test accuracy lost with the block replaced by the identity (baseline ${pct(A.accuracy.overall, 1)}).`),
      barChart(W, { labels: K.map((k) => `block ${k.block}`), values: K.map((k) => -k.delta), fmt: (v) => pts(-v), emphasis: K.reduce((a, k, i) => (-k.delta > -K[a].delta ? i : a), 0), aria: "accuracy lost per skipped block" }),
      dataTable(["block", "accuracy", "Δ points"], K.map((k) => [k.block, pct(k.accuracy, 1), pts(k.delta)]))),
    h("section", { class: "card" }, h("h3", {}, "Every head"), gridBox,
      dataTable(["block", "head", "Δ if zeroed", "entropy", "reach", "to CLS"], Hd.map((x) => [x.block, x.head, pts(x.knockout_delta), x.entropy.toFixed(2), x.reach.toFixed(1), pct(x.to_cls)]))),
    h("section", { class: "card" }, h("h3", {}, "The residual stream"),
      h("p", { class: "sub" }, "Update size relative to the stream it's added to (‖update‖ ÷ ‖input‖)."),
      legend([["attention", "var(--s1)"], ["MLP", "var(--s2)"]], "box"),
      groupedBars(W, { labels: Rs.map((r) => `block ${r.block}`), series: [{ name: "attention", values: Rs.map((r) => r.attn_update), color: "var(--s1)" }, { name: "MLP", values: Rs.map((r) => r.mlp_update), color: "var(--s2)" }], fmt: (v) => `${v.toFixed(2)}×`, aria: "update size per block" }),
      dataTable(["block", "input norm", "attention update", "MLP update"], Rs.map((r) => [r.block, r.input_norm.toFixed(1), `${r.attn_update.toFixed(2)}×`, `${r.mlp_update.toFixed(2)}×`]))),
  );
}
