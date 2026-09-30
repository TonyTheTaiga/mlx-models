// Lumen — which patches did the model lean on for its final prediction?
//
// The model runs in the browser (worker.js, a JS port of the MLX ViT). For the current image a
// pool of workers hides each patch in turn and records the prediction (the occlusion sweep), then
// checks the resulting ranking by keeping or removing the top-k patches (the curves). The
// scoring, drawing and wording live in evidence.js.

import { $, clamp, rgba, fit, CLASS_COLORS } from "./util.js";
import { evidence, curvePoints, currentK, predicted, drawFigure, drawChart, answerHtml, tooltipHtml } from "./evidence.js";

// ---------- state ----------
const S = {
  cfg: null, classes: [],
  samples: null, labels: [], sampleIndex: 0, source: null, label: null,
  pixels: new Uint8ClampedArray(32 * 32 * 3),
  result: null, resultId: 0,
  occ: null, occWant: 0, curves: null, curvesWant: 0, curvesKey: "",
  ev: { cls: null, k: null, heat: false }, // target class (null = prediction), kept k (null = the headline k)
  hover: null, pointer: null,
  brush: 0, paint: [235, 230, 218], showOverlay: true,
};

// Draw on demand: only when something changed.
let dirty = true;
const invalidate = () => (dirty = true);

// ---------- the model ----------
const worker = new Worker("worker.js", { type: "module" });
let inflight = false, pending = false;
function requestForward() {
  if (!S.cfg) return;
  if (inflight) { pending = true; return; }
  inflight = true;
  worker.postMessage({ type: "forward", pixels: S.pixels.slice() });
}
worker.onmessage = ({ data }) => {
  if (data.type === "ready") {
    S.cfg = data.cfg;
    S.classes = data.classes;
    const c = S.cfg;
    $("meta").textContent = `ViT · ${c.layers} blocks · ${c.heads} heads · ${c.grid}×${c.grid} patches of ${c.win}px · CIFAR-10`;
    boot();
  } else if (data.type === "result") {
    inflight = false;
    S.result = data.out;
    S.resultId++;
    S.ev.k = null; // back to the headline k for the new image
    requestOcclusion();
    buildControls();
    if (pending) { pending = false; requestForward(); }
  } else if (data.type === "check") {
    const ok = data.worst < 1e-3;
    $("check").textContent = `${ok ? "✓" : "✗"} browser forward pass vs MLX on ${data.n} images: max logit error ${data.worst.toExponential(1)}`;
  }
};

// ---------- the sweeps, on a worker pool ----------
// Each worker takes every n-th patch (or k) and returns only its share; parts sum without overlap.
const pool = Array.from({ length: clamp((navigator.hardwareConcurrency || 4) - 1, 1, 4) }, () => new Worker("worker.js", { type: "module" }));
let jobId = 0, occParts = null, curveParts = null, occTimer = 0;
function mergeParts(parts) {
  const out = new Float32Array(parts[0].length);
  for (const part of parts) for (let j = 0; j < out.length; j++) out[j] += part[j];
  return out;
}
const shares = (count, i) => [...Array(count).keys()].filter((x) => x % pool.length === i);

function requestOcclusion() {
  clearTimeout(occTimer);
  // Debounced so a paint stroke only triggers one sweep once it settles.
  occTimer = setTimeout(() => {
    const n = S.cfg.grid * S.cfg.grid;
    S.occWant = ++jobId;
    S.occ = { id: S.occWant, done: 0, logits: null };
    occParts = { id: S.occWant, done: new Array(pool.length).fill(0), logits: [], left: pool.length };
    pool.forEach((w, i) => w.postMessage({ type: "occlude", id: S.occWant, pixels: S.pixels.slice(), patches: shares(n, i) }));
    invalidate();
  }, 250);
}

// The keep/remove check runs once per (sweep, target class), on the ranking that sweep produced.
function ensureCurves() {
  const E = evidence(S);
  if (!E.ready) return;
  const key = [S.occ.id, E.c].join();
  if (key === S.curvesKey) return;
  S.curvesKey = key;
  S.curvesWant = ++jobId;
  S.curves = { id: S.curvesWant, keep: null, drop: null };
  curveParts = { id: S.curvesWant, keep: [], drop: [], left: pool.length };
  const n = S.cfg.grid * S.cfg.grid;
  pool.forEach((w, i) => w.postMessage({ type: "curves", id: S.curvesWant, pixels: S.pixels.slice(), order: E.order, ks: shares(n + 1, i) }));
}

pool.forEach((w, i) => {
  w.onmessage = ({ data }) => {
    if (data.type === "curves") {
      if (!curveParts || data.id !== curveParts.id) return;
      curveParts.keep[i] = data.keep;
      curveParts.drop[i] = data.drop;
      if (--curveParts.left === 0) {
        S.curves = { id: data.id, keep: mergeParts(curveParts.keep), drop: mergeParts(curveParts.drop) };
        buildControls(); // the slider starts at the headline k
      }
    } else {
      if (!occParts || data.id !== occParts.id) return;
      if (data.type === "occlusion-progress") occParts.done[i] = data.done;
      else if (data.type === "occlusion") { occParts.logits[i] = data.logits; occParts.left--; }
      const done = occParts.done.reduce((a, b) => a + b, 0);
      S.occ = { id: data.id, done, logits: occParts.left === 0 ? mergeParts(occParts.logits) : null };
      if (S.occ.logits) ensureCurves(); // start the check right away, even if nothing repaints
    }
    invalidate();
  };
});

async function boot() {
  const [meta, buf, ref] = await Promise.all([
    fetch("assets/samples.json").then((r) => r.json()),
    fetch("assets/samples.bin").then((r) => r.arrayBuffer()),
    fetch("assets/reference.json").then((r) => r.json()),
  ]);
  S.samples = new Uint8Array(buf);
  S.labels = meta.labels;
  buildGallery();
  selectSample(Math.floor(Math.random() * meta.count));
  worker.postMessage({
    type: "check",
    images: ref.indices.map((i) => S.samples.slice(i * 3072, (i + 1) * 3072)),
    logits: ref.logits,
  });
}

// ---------- the input image ----------
const inputCtx = $("input").getContext("2d");
const inputImage = inputCtx.createImageData(32, 32);
function drawInput() {
  for (let i = 0; i < 1024; i++) {
    for (let c = 0; c < 3; c++) inputImage.data[i * 4 + c] = S.pixels[i * 3 + c];
    inputImage.data[i * 4 + 3] = 255;
  }
  inputCtx.putImageData(inputImage, 0, 0);
}
function pixelsChanged() {
  drawInput();
  requestForward();
}
function setPixels(px, label = null) {
  S.pixels.set(px);
  S.source = { pixels: S.pixels.slice(), label };
  S.label = label;
  S.ev.cls = null;
  pixelsChanged();
}
function selectSample(i) {
  S.sampleIndex = i;
  setPixels(S.samples.subarray(i * 3072, (i + 1) * 3072), S.labels[i]);
  document.querySelectorAll(".gallery canvas").forEach((c, k) => c.classList.toggle("on", k === i));
}

function buildGallery() {
  const g = $("gallery");
  for (let i = 0; i < S.labels.length; i++) {
    const c = document.createElement("canvas");
    c.width = c.height = 32;
    const ctx = c.getContext("2d");
    const img = ctx.createImageData(32, 32);
    for (let p = 0; p < 1024; p++) {
      for (let k = 0; k < 3; k++) img.data[p * 4 + k] = S.samples[i * 3072 + p * 3 + k];
      img.data[p * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    c.title = S.classes[S.labels[i]];
    c.onclick = () => selectSample(i);
    g.appendChild(c);
  }
}

// Painting.
const swatchColors = [[235, 230, 218], [16, 16, 20], [214, 64, 52], [238, 196, 62], [72, 150, 70], [74, 128, 214], [140, 96, 60]];
function buildSwatches() {
  const box = $("swatches");
  swatchColors.forEach((c, i) => {
    const b = document.createElement("button");
    b.style.background = rgba(c);
    b.title = "paint color";
    if (i === 0) b.classList.add("on");
    b.onclick = () => setPaint(c, b);
    box.appendChild(b);
  });
  const pick = document.createElement("input");
  pick.type = "color";
  pick.title = "custom color";
  pick.oninput = () => setPaint([1, 3, 5].map((k) => parseInt(pick.value.slice(k, k + 2), 16)), null);
  box.appendChild(pick);
}
function setPaint(c, btn) {
  S.paint = c;
  document.querySelectorAll(".swatches button").forEach((b) => b.classList.toggle("on", b === btn));
}
$("brush").onclick = (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  S.brush = +b.dataset.size;
  $("brush").querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
};

const frame = $("frame");
let painting = false;
function paintAt(e) {
  const r = frame.getBoundingClientRect();
  const x = Math.floor(((e.clientX - r.left) / r.width) * 32), y = Math.floor(((e.clientY - r.top) / r.height) * 32);
  if (e.altKey) {
    if (x < 0 || y < 0 || x > 31 || y > 31) return;
    const i = (y * 32 + x) * 3;
    setPaint([S.pixels[i], S.pixels[i + 1], S.pixels[i + 2]], null);
    return;
  }
  const rad = S.brush;
  for (let dy = -rad; dy <= rad; dy++)
    for (let dx = -rad; dx <= rad; dx++) {
      if (dx * dx + dy * dy > rad * rad + rad) continue;
      const px = x + dx, py = y + dy;
      if (px < 0 || py < 0 || px > 31 || py > 31) continue;
      S.pixels.set(S.paint, (py * 32 + px) * 3);
    }
  pixelsChanged();
}
frame.addEventListener("pointerdown", (e) => { painting = true; frame.setPointerCapture(e.pointerId); paintAt(e); });
frame.addEventListener("pointermove", (e) => painting && paintAt(e));
frame.addEventListener("pointerup", () => (painting = false));
frame.addEventListener("pointercancel", () => (painting = false));

$("reset").onclick = () => S.source && setPixels(S.source.pixels, S.source.label);
$("scramble").onclick = () => {
  const { grid, win } = S.cfg;
  const order = [...Array(grid * grid).keys()].sort(() => Math.random() - 0.5);
  const src = S.pixels.slice();
  order.forEach((from, to) => {
    const fr = Math.floor(from / grid), fc = from % grid, tr = Math.floor(to / grid), tc = to % grid;
    for (let i = 0; i < win; i++)
      for (let j = 0; j < win; j++) {
        const a = ((fr * win + i) * 32 + fc * win + j) * 3, b = ((tr * win + i) * 32 + tc * win + j) * 3;
        S.pixels.set(src.subarray(a, a + 3), b);
      }
  });
  pixelsChanged();
};
$("flip").onclick = () => {
  const src = S.pixels.slice();
  for (let y = 0; y < 32; y++)
    for (let x = 0; x < 32; x++) S.pixels.set(src.subarray((y * 32 + 31 - x) * 3, (y * 32 + 32 - x) * 3), (y * 32 + x) * 3);
  pixelsChanged();
};
$("showOverlay").onchange = (e) => { S.showOverlay = e.target.checked; invalidate(); };

// Uploads: center-crop and downsample to 32×32.
function loadFile(file) {
  if (!file || !file.type.startsWith("image/")) return;
  const img = new Image();
  img.onload = () => {
    const c = document.createElement("canvas");
    c.width = c.height = 32;
    const ctx = c.getContext("2d");
    ctx.imageSmoothingQuality = "high";
    const s = Math.min(img.width, img.height);
    ctx.drawImage(img, (img.width - s) / 2, (img.height - s) / 2, s, s, 0, 0, 32, 32);
    const d = ctx.getImageData(0, 0, 32, 32).data;
    const px = new Uint8ClampedArray(3072);
    for (let i = 0; i < 1024; i++) for (let k = 0; k < 3; k++) px[i * 3 + k] = d[i * 4 + k];
    document.querySelectorAll(".gallery canvas.on").forEach((x) => x.classList.remove("on"));
    setPixels(px, null);
    URL.revokeObjectURL(img.src);
  };
  img.src = URL.createObjectURL(file);
}
$("upload").onchange = (e) => loadFile(e.target.files[0]);
let dragDepth = 0;
addEventListener("dragenter", (e) => { e.preventDefault(); dragDepth++; $("drop").classList.add("on"); });
addEventListener("dragleave", () => { if (--dragDepth <= 0) $("drop").classList.remove("on"); });
addEventListener("dragover", (e) => e.preventDefault());
addEventListener("drop", (e) => {
  e.preventDefault();
  dragDepth = 0;
  $("drop").classList.remove("on");
  loadFile(e.dataTransfer.files[0]);
});

// ---------- controls: target class, kept k, signed map ----------
function buildControls() {
  const box = $("controls");
  box.innerHTML = "";
  if (!S.result) return;
  const add = (html, on, onclick, label) => {
    const b = document.createElement("button");
    b.innerHTML = html;
    b.classList.toggle("on", on);
    b.onclick = onclick;
    if (label) b.setAttribute("aria-label", label), (b.title = label);
    box.appendChild(b);
  };
  // Evidence for the prediction, or for the runner-up classes ("why not a cat?").
  const probs = S.result.probs;
  const top3 = [...probs.keys()].sort((a, b) => probs[b] - probs[a]).slice(0, 3);
  top3.forEach((c) =>
    add(`<span class="dot" style="background:${rgba(CLASS_COLORS[c])}"></span>${S.classes[c]} ${(probs[c] * 100).toFixed(0)}%`,
      c === (S.ev.cls ?? predicted(S)), () => { S.ev = { ...S.ev, cls: c === predicted(S) ? null : c, k: null }; ensureCurves(); buildControls(); },
      `evidence for ${S.classes[c]}`));

  const n = S.cfg.grid * S.cfg.grid, k = currentK(S);
  const slider = document.createElement("label");
  slider.className = "slider";
  slider.innerHTML = `keep top <input type="range" min="1" max="${n}" value="${k}" aria-label="patches kept"> <b>${k}</b>`;
  slider.querySelector("input").oninput = (e) => {
    S.ev.k = +e.target.value;
    slider.querySelector("b").textContent = e.target.value;
    invalidate();
  };
  box.appendChild(slider);
  const K = curvePoints(S);
  if (K && S.ev.k != null && S.ev.k !== K.best)
    add(K.enough != null ? "↺ just enough" : "↺ strongest case", false, () => { S.ev.k = null; buildControls(); });
  const spacer = document.createElement("span");
  spacer.className = "spacer";
  box.appendChild(spacer);
  add("signed map", S.ev.heat, () => { S.ev.heat = !S.ev.heat; buildControls(); }, "Color every patch by its evidence, for and against");
  invalidate();
}

addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || !S.cfg) return;
  if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
    const n = S.labels.length;
    selectSample((S.sampleIndex + (e.key === "ArrowRight" ? 1 : n - 1)) % n);
  }
});

// ---------- the figure: hover and click ----------
const figure = $("figure"), tooltip = $("tooltip");
figure.addEventListener("pointermove", (e) => {
  const r = figure.getBoundingClientRect();
  S.pointer = [e.clientX - r.left, e.clientY - r.top];
  invalidate();
});
figure.addEventListener("pointerleave", () => { S.pointer = null; S.hover = null; tooltip.hidden = true; invalidate(); });
figure.addEventListener("click", () => {
  // Clicking a patch keeps everything down to its rank.
  const E = evidence(S);
  if (!S.hover || !E.ready) return;
  S.ev.k = E.order.indexOf(S.hover.p) + 1;
  buildControls();
});
function placeTooltip(html) {
  if (!html || !S.pointer) { tooltip.hidden = true; return; }
  tooltip.hidden = false;
  if (tooltip.innerHTML !== html) tooltip.innerHTML = html;
  const fr = figure.getBoundingClientRect(), st = figure.closest(".stage").getBoundingClientRect();
  const x = fr.left - st.left + S.pointer[0], y = fr.top - st.top + S.pointer[1];
  const w = tooltip.offsetWidth;
  tooltip.style.left = `${Math.max(8, x + w + 30 > st.width ? x - w - 28 : x)}px`;
  tooltip.style.top = `${y}px`;
}

// ---------- the input overlay: support for the target class ----------
function drawOverlay(ctx, W, H) {
  const E = evidence(S);
  if (!S.showOverlay || !E.ready) return;
  const { grid } = S.cfg, t = W / grid, max = Math.max(1e-6, ...E.scores), col = CLASS_COLORS[E.c];
  ctx.fillStyle = "rgba(10,10,15,0.35)";
  ctx.fillRect(0, 0, W, H);
  for (let p = 0; p < grid * grid; p++) {
    const v = Math.max(0, E.scores[p]) / max;
    ctx.fillStyle = rgba(col, 0.6 * v ** 1.3);
    ctx.fillRect((p % grid) * t, Math.floor(p / grid) * t, t, t);
  }
}

// ---------- loop ----------
new ResizeObserver(invalidate).observe(document.body);
let lastAnswer = "", lastHover = "";
function frameLoop() {
  requestAnimationFrame(frameLoop);
  if (!dirty || !S.result) return;
  dirty = false;
  ensureCurves();

  const [ctx, W, H] = fit(figure);
  ctx.clearRect(0, 0, W, H);
  S.hover = drawFigure(ctx, W, H, S);
  placeTooltip(S.hover && tooltipHtml(S, S.hover));
  const hk = JSON.stringify(S.hover);
  if (hk !== lastHover) { lastHover = hk; dirty = true; } // repaint once to outline the new hover

  const answer = answerHtml(S);
  if (answer !== lastAnswer) { $("answer").innerHTML = answer; lastAnswer = answer; }
  const chart = $("chart");
  chart.parentElement.style.visibility = curvePoints(S) ? "visible" : "hidden";
  const [cctx, cW, cH] = fit(chart);
  cctx.clearRect(0, 0, cW, cH);
  drawChart(cctx, cW, cH, S);

  const [octx, oW, oH] = fit($("inputOverlay"));
  octx.clearRect(0, 0, oW, oH);
  drawOverlay(octx, oW, oH);
}

buildSwatches();
requestAnimationFrame(frameLoop);
