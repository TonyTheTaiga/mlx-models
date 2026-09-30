// Lumen — renders the ViT's residual stream and attention as light.
//
// Tower: the scrub accordion, one block open at a time — see tower.js.
// Atlas: every head's attention map from the query, plus the residual stream mosaic.
// Drift: the rotary positions each layer learned (they were left trainable).

import { $, clamp, lerp, ease, frac, oklabToRgb, rgba, fit, HEAD_COLORS, CLASS_COLORS, GOLD, PAPER, layerColor } from "./util.js";
import { drawTower, towerTooltip, readingHtml, tokenName as nameOf } from "./tower.js";

// ---------- state ----------
const S = {
  cfg: null, classes: [], rope: [], pca: null,
  samples: null, labels: [], sampleIndex: 0, source: null,
  pixels: new Uint8ClampedArray(32 * 32 * 3),
  result: null, colors: null,
  query: 0, head: -1, mode: "tower",
  // Tower renders scrub; tower.js also keeps the loom/orbit/stack variants, unused in the UI.
  variant: "scrub", block: 0, blockF: 0, resultId: 0,
  yaw: -0.55, pitch: 0.42, dragging: false,
  hover: null, pointer: null, driftPin: -1,
  brush: 0, paint: [235, 230, 218], showRollout: true,
  t: 0,
};

// Draw on demand: only when something changed, or while a transition is running.
let dirty = true, panelsDirty = true;
function invalidate(panels = false) { dirty = true; if (panels) panelsDirty = true; }

// ---------- worker ----------
const worker = new Worker("worker.js", { type: "module" });
let inflight = false, pending = false, reqId = 0;
function requestForward() {
  if (!S.cfg) return;
  if (inflight) { pending = true; return; }
  inflight = true;
  worker.postMessage({ type: "forward", id: ++reqId, pixels: S.pixels.slice() });
}
worker.onmessage = ({ data }) => {
  if (data.type === "ready") {
    S.cfg = data.cfg; S.classes = data.classes; S.rope = data.rope;
    S.block = S.blockF = data.cfg.layers - 1;
    // Checkpoints trained after the Rope2D fix carry the untouched grid and frequencies.
    const g = data.cfg.grid, nf = data.rope[0].freq.length;
    S.ropeLearned = data.rope.some((L) =>
      L.posIdx.some((v, i) => Math.abs(v - (i % 2 ? Math.floor(i / 2) % g : Math.floor(i / 2 / g))) > 1e-4) ||
      L.freq.some((f, j) => Math.abs(f / Math.pow(100, -j / nf) - 1) > 1e-4));
    const c = S.cfg;
    $("meta").textContent = `ViT · ${c.layers} blocks · ${c.heads} heads · d=${c.dModel} · ${c.grid}×${c.grid} patches of ${c.win}px · 2D RoPE · CIFAR-10`;
    boot();
  } else if (data.type === "result") {
    inflight = false;
    S.result = data.out;
    S.colors = tokenColors(data.out.states);
    S.resultId++;
    renderVerdict();
    invalidate(true);
    if (pending) { pending = false; requestForward(); }
  } else if (data.type === "check") {
    const ok = data.worst < 1e-3;
    $("check").textContent = `${ok ? "✓" : "✗"} browser forward pass vs MLX on ${data.n} images: max logit error ${data.worst.toExponential(1)}`;
  }
};

async function boot() {
  const [pca, meta, buf, ref] = await Promise.all([
    fetch("assets/pca.json").then((r) => r.json()),
    fetch("assets/samples.json").then((r) => r.json()),
    fetch("assets/samples.bin").then((r) => r.arrayBuffer()),
    fetch("assets/reference.json").then((r) => r.json()),
  ]);
  S.pca = pca;
  S.samples = new Uint8Array(buf);
  S.labels = meta.labels;
  buildGallery();
  buildLegend();
  // Start on a random, correctly-framed test image.
  selectSample(Math.floor(Math.random() * meta.count));
  worker.postMessage({
    type: "check",
    images: ref.indices.map((i) => S.samples.slice(i * 3072, (i + 1) * 3072)),
    logits: ref.logits,
  });
}

// PCA false color per token per depth: component 1 → lightness, 2/3 → OKLab a/b.
function tokenColors(states) {
  const { nTokens: T, dModel: D } = S.cfg;
  return states.map((x, d) => {
    const { mean, components, lo, hi } = S.pca[d];
    const out = [];
    for (let t = 0; t < T; t++) {
      const u = components.map((comp, k) => {
        let s = 0;
        for (let i = 0; i < D; i++) s += (x[t * D + i] - mean[i]) * comp[i];
        return clamp((s - lo[k]) / (hi[k] - lo[k]), 0, 1);
      });
      out.push(oklabToRgb(0.38 + 0.46 * u[0], (u[1] - 0.5) * 0.34, (u[2] - 0.5) * 0.34));
    }
    return out;
  });
}

// ---------- input image ----------
const inputCtx = $("input").getContext("2d");
const inputImage = inputCtx.createImageData(32, 32);
function drawInput() {
  for (let i = 0; i < 1024; i++) {
    inputImage.data[i * 4] = S.pixels[i * 3];
    inputImage.data[i * 4 + 1] = S.pixels[i * 3 + 1];
    inputImage.data[i * 4 + 2] = S.pixels[i * 3 + 2];
    inputImage.data[i * 4 + 3] = 255;
  }
  inputCtx.putImageData(inputImage, 0, 0);
}
function setPixels(px, label = null) {
  S.pixels.set(px);
  S.source = { pixels: S.pixels.slice(), label };
  S.label = label;
  drawInput();
  requestForward();
}
function selectSample(i) {
  S.sampleIndex = i;
  setPixels(S.samples.subarray(i * 3072, (i + 1) * 3072), S.labels[i]);
  document.querySelectorAll(".gallery canvas").forEach((c, k) => c.classList.toggle("on", k === i));
}

function buildGallery() {
  const g = $("gallery");
  const n = S.labels.length;
  for (let i = 0; i < n; i++) {
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
  pick.oninput = () => {
    const h = pick.value;
    setPaint([1, 3, 5].map((k) => parseInt(h.slice(k, k + 2), 16)), null);
  };
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
function framePixel(e) {
  const r = frame.getBoundingClientRect();
  return [Math.floor(((e.clientX - r.left) / r.width) * 32), Math.floor(((e.clientY - r.top) / r.height) * 32)];
}
function paintAt(e) {
  const [x, y] = framePixel(e);
  if (e.altKey) {
    if (x < 0 || y < 0 || x > 31 || y > 31) return;
    const i = (y * 32 + x) * 3;
    setPaint([S.pixels[i], S.pixels[i + 1], S.pixels[i + 2]], null);
    return;
  }
  const r = S.brush;
  for (let dy = -r; dy <= r; dy++)
    for (let dx = -r; dx <= r; dx++) {
      if (dx * dx + dy * dy > r * r + r) continue;
      const px = x + dx, py = y + dy;
      if (px < 0 || py < 0 || px > 31 || py > 31) continue;
      S.pixels.set(S.paint, (py * 32 + px) * 3);
    }
  drawInput();
  requestForward();
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
  drawInput();
  requestForward();
};
$("flip").onclick = () => {
  const src = S.pixels.slice();
  for (let y = 0; y < 32; y++)
    for (let x = 0; x < 32; x++) S.pixels.set(src.subarray((y * 32 + 31 - x) * 3, (y * 32 + 32 - x) * 3), (y * 32 + x) * 3);
  drawInput();
  requestForward();
};
$("showRollout").onchange = (e) => { S.showRollout = e.target.checked; invalidate(true); };

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

// ---------- legend & modes ----------
const HINTS = {
  scrub: "Scroll or ↑↓ to move through depth. Only the active block opens: curves are attention into the query, the dotted line is its residual path. Click a tile to follow it, or a closed plate to open it.",
  loom: "Every token is a vertical thread, recolored at each depth. Attention is woven between rows — bright for the active block, a faint trace for the rest. Scroll or ↑↓ to change block.",
  orbit: "Each ring is one depth, the image unwrapped around its center, with CLS at the heart. Threads spiral from one ring into the next. Scroll or ↑↓ to change block.",
  stack: "The whole stack in 3D. Drag to orbit, scroll or ↑↓ to change the active block, click a tile to follow it.",
  atlas: "Rows are blocks, columns are heads: where the query looks at each step. Last column is the residual stream itself. Click a patch to make it the query.",
  drift: "RoPE positions were left trainable, so each block learned its own map of where the patches sit. The image is re-assembled at those learned coordinates.",
  driftFixed: "This checkpoint's RoPE positions and frequencies are fixed, so every block uses the same grid — there is no drift to show.",
};
const tokenName = (t) => nameOf(S.cfg, t);

function buildLegend() {
  if (!S.cfg) return;
  const L = $("legend");
  L.innerHTML = "";
  const add = (html, on, onclick) => {
    const b = document.createElement("button");
    b.innerHTML = html;
    b.classList.toggle("on", on);
    b.onclick = onclick;
    L.appendChild(b);
  };
  if (S.mode === "drift") {
    add("cycle", S.driftPin < 0, () => { S.driftPin = -1; buildLegend(); });
    for (let l = 0; l < S.cfg.layers; l++)
      add(`<span class="dot" style="background:${rgba(layerColor(l, S.cfg.layers))}"></span>block ${l + 1}`, S.driftPin === l, () => { S.driftPin = l; buildLegend(); });
  } else {
    if (S.mode === "tower") {
      const step = document.createElement("span");
      step.className = "stepper";
      step.innerHTML = `<button data-d="-1" aria-label="Previous block">‹</button><span>block <b>${S.block + 1}</b> / ${S.cfg.layers}</span><button data-d="1" aria-label="Next block">›</button>`;
      step.onclick = (e) => { const b = e.target.closest("button"); if (b) setBlock(S.block + +b.dataset.d); };
      L.appendChild(step);
    }
    add("all heads", S.head < 0, () => setHead(-1));
    for (let h = 0; h < S.cfg.heads; h++)
      add(`<span class="dot" style="background:${rgba(HEAD_COLORS[h])}"></span>head ${h + 1}`, S.head === h, () => setHead(S.head === h ? -1 : h));
    const q = document.createElement("span");
    q.className = "query";
    q.innerHTML = `query <b>${tokenName(S.query)}</b>`;
    L.appendChild(q);
  }
  $("hint").textContent = S.mode === "drift" && !S.ropeLearned ? HINTS.driftFixed : HINTS[S.mode === "tower" ? S.variant : S.mode];
  invalidate(true);
}
function setHead(h) { S.head = h; buildLegend(); }
function setQuery(q) { S.query = q; buildLegend(); }
function setBlock(b) {
  b = clamp(b, 0, S.cfg.layers - 1);
  if (b !== S.block) { S.block = b; buildLegend(); }
}
function setMode(m) {
  S.mode = m;
  S.hover = null;
  document.querySelectorAll("#modes button").forEach((b) => b.classList.toggle("on", b.dataset.mode === m));
  buildLegend();
}
$("modes").onclick = (e) => { const b = e.target.closest("button"); if (b) setMode(b.dataset.mode); };

addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || !S.cfg) return;
  if (e.key === "Escape") setQuery(0);
  else if (e.key === "0") setHead(-1);
  else if (e.key >= "1" && e.key <= String(S.cfg.heads)) setHead(+e.key - 1);
  else if (e.key === "t") setMode("tower");
  else if (e.key === "a") setMode("atlas");
  else if (e.key === "d") setMode("drift");
  else if ((e.key === "ArrowUp" || e.key === "ArrowDown") && S.mode === "tower") {
    e.preventDefault();
    setBlock(S.block + (e.key === "ArrowUp" ? 1 : -1));
  } else if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
    const n = S.labels.length;
    selectSample((S.sampleIndex + (e.key === "ArrowRight" ? 1 : n - 1)) % n);
  }
});

// The reading card: the active block's strongest sources, in words.
let readingKey = "";
function updateReading(wide) {
  const el = $("reading");
  el.hidden = S.mode !== "tower" || !wide;
  if (el.hidden) return;
  const key = [S.resultId, S.query, S.head, S.block].join();
  if (key === readingKey) return;
  readingKey = key;
  el.innerHTML = readingHtml(S);
  const { grid, win } = S.cfg;
  el.querySelectorAll("canvas").forEach((c) => {
    const t = +c.dataset.t, cx = c.getContext("2d");
    if (t === 0) { cx.fillStyle = rgba(GOLD); cx.fillRect(1, 1, 2, 2); return; }
    const r0 = Math.floor((t - 1) / grid) * win, c0 = ((t - 1) % grid) * win;
    for (let i = 0; i < win; i++)
      for (let j = 0; j < win; j++) {
        const o = ((r0 + i) * 32 + c0 + j) * 3;
        cx.fillStyle = `rgb(${S.pixels[o]},${S.pixels[o + 1]},${S.pixels[o + 2]})`;
        cx.fillRect(j, i, 1, 1);
      }
  });
}

// ---------- atlas ----------
function drawAtlas(ctx, W, H) {
  const { grid, nTokens: T, layers, heads } = S.cfg;
  const { attn } = S.result;
  const cols = heads + 1;
  const top = 70, bottom = 70, leftPad = 70;
  const cell = Math.min((W - leftPad - 30) / cols, (H - top - bottom) / layers) * 0.9;
  const gap = cell * 0.11;
  const totalW = cols * cell + (cols - 1) * gap;
  const x0 = leftPad + (W - leftPad - 30 - totalW) / 2;
  const y0 = top + (H - top - bottom - (layers * cell + (layers - 1) * gap)) / 2;
  const tile = cell / grid;
  ctx.font = "10px 'JetBrains Mono', monospace";
  ctx.textBaseline = "middle";

  for (let c = 0; c < cols; c++) {
    ctx.textAlign = "center";
    ctx.fillStyle = c < heads ? rgba(HEAD_COLORS[c], S.head < 0 || S.head === c ? 0.95 : 0.35) : "rgba(235,230,218,0.7)";
    ctx.fillText(c < heads ? `head ${c + 1}` : "stream", x0 + c * (cell + gap) + cell / 2, y0 - 16);
  }

  S.hover = null;
  const [mx, my] = S.pointer || [-1, -1];
  for (let l = 0; l < layers; l++) {
    const y = y0 + l * (cell + gap);
    ctx.textAlign = "right";
    ctx.fillStyle = "rgba(235,230,218,0.7)";
    ctx.fillText(`block ${l + 1}`, x0 - 14, y + cell / 2);
    for (let c = 0; c < cols; c++) {
      const x = x0 + c * (cell + gap);
      ctx.globalCompositeOperation = "source-over";
      if (c < heads) {
        // Dim, desaturated image under a glow of attention weights.
        for (let p = 0; p < grid * grid; p++) {
          const r = Math.floor(p / grid), cc = p % grid;
          const o = ((r * 4 + 2) * 32 + cc * 4 + 2) * 3;
          const lum = (S.pixels[o] * 0.3 + S.pixels[o + 1] * 0.59 + S.pixels[o + 2] * 0.11) * 0.22;
          ctx.fillStyle = `rgb(${lum},${lum},${lum + 4})`;
          ctx.fillRect(x + cc * tile, y + r * tile, tile + 0.5, tile + 0.5);
        }
        const row = attn[l][c].subarray(S.query * T, S.query * T + T);
        let max = 1e-6;
        for (let j = 1; j < T; j++) max = Math.max(max, row[j]);
        const dim = S.head >= 0 && S.head !== c ? 0.3 : 1;
        ctx.globalCompositeOperation = "lighter";
        for (let p = 0; p < grid * grid; p++) {
          const v = (row[p + 1] / max) ** 0.8;
          ctx.fillStyle = rgba(HEAD_COLORS[c], v * 0.95 * dim);
          ctx.fillRect(x + (p % grid) * tile, y + Math.floor(p / grid) * tile, tile + 0.5, tile + 0.5);
        }
        // Share of attention spent on CLS: a bar under the cell.
        ctx.globalCompositeOperation = "source-over";
        ctx.fillStyle = "rgba(235,230,218,0.1)";
        ctx.fillRect(x, y + cell + 3, cell, 2);
        ctx.fillStyle = rgba(GOLD, 0.8 * dim);
        ctx.fillRect(x, y + cell + 3, cell * row[0], 2);
      } else {
        const colors = S.colors[l + 1];
        for (let p = 0; p < grid * grid; p++) {
          ctx.fillStyle = rgba(colors[p + 1]);
          ctx.fillRect(x + (p % grid) * tile, y + Math.floor(p / grid) * tile, tile + 0.5, tile + 0.5);
        }
      }
      ctx.globalCompositeOperation = "source-over";
      if (S.query > 0) {
        const q = S.query - 1;
        ctx.strokeStyle = rgba(GOLD, 0.95);
        ctx.lineWidth = 1.2;
        ctx.strokeRect(x + (q % grid) * tile + 0.5, y + Math.floor(q / grid) * tile + 0.5, tile - 1, tile - 1);
      }
      if (mx >= x && mx < x + cell && my >= y && my < y + cell) {
        const p = Math.floor((my - y) / tile) * grid + Math.floor((mx - x) / tile);
        S.hover = { kind: "atlas", layer: l, head: c, patch: p, token: p + 1 };
        ctx.strokeStyle = rgba(PAPER, 0.9);
        ctx.strokeRect(x + (p % grid) * tile + 0.5, y + Math.floor(p / grid) * tile + 0.5, tile - 1, tile - 1);
      }
    }
  }
  ctx.textAlign = "left";
  ctx.fillStyle = "rgba(139,134,118,0.9)";
  ctx.fillText("gold bar = share of attention spent on CLS", x0, y0 + layers * (cell + gap) + 8);

  if (S.hover) {
    const { layer, head, token } = S.hover;
    let text = `<b>block ${layer + 1}</b> · ${tokenName(token)}`;
    if (head < heads) text += ` · head ${head + 1} · ${tokenName(S.query)} → here ${(attn[layer][head][S.query * T + token] * 100).toFixed(1)}%`;
    showTooltip(text);
  } else hideTooltip();
}

// ---------- drift ----------
function drawDrift(ctx, W, H) {
  const { grid, win, layers } = S.cfg;
  const rope = S.rope;
  let lo = 0, hi = grid - 1;
  for (const L of rope) for (const v of L.posIdx) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  lo -= 0.9; hi += 0.9;
  const fw = clamp(W * 0.2, 150, 220), fh = 120;
  const size = Math.max(160, Math.min(W - fw - 110, H - 170));
  const ox = (W - (size + 50 + fw)) / 2, oy = (H - size) / 2 + 14;
  const unit = size / (hi - lo);
  const to = (row, col) => [ox + (col - lo) * unit, oy + (row - lo) * unit];

  // Animate through the blocks (or hold a pinned one).
  let cur, l0, l1, e;
  if (S.driftPin >= 0) { l0 = l1 = S.driftPin; e = 0; cur = S.driftPin; }
  else {
    cur = (S.t * 0.22) % layers;
    l0 = Math.floor(cur); l1 = (l0 + 1) % layers;
    e = ease(clamp((frac(cur) - 0.55) / 0.45, 0, 1));
  }

  // The lattice RoPE was initialized with.
  ctx.fillStyle = "rgba(235,230,218,0.22)";
  for (let r = 0; r < grid; r++)
    for (let c = 0; c < grid; c++) {
      const [x, y] = to(r, c);
      ctx.beginPath();
      ctx.arc(x, y, 1.3, 0, Math.PI * 2);
      ctx.fill();
    }

  // Every block's learned mesh, faint.
  ctx.globalCompositeOperation = "lighter";
  rope.forEach((L, l) => {
    const pos = (p) => to(L.posIdx[2 * p], L.posIdx[2 * p + 1]);
    const active = S.driftPin >= 0 ? l === S.driftPin : l === (e > 0.5 ? l1 : l0);
    ctx.strokeStyle = rgba(layerColor(l, layers), active ? 0.55 : 0.14);
    ctx.lineWidth = active ? 1.1 : 0.8;
    ctx.beginPath();
    for (let r = 0; r < grid; r++)
      for (let c = 0; c < grid; c++) {
        const p = r * grid + c;
        const a = pos(p);
        if (c + 1 < grid) { const b = pos(p + 1); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); }
        if (r + 1 < grid) { const b = pos(p + grid); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); }
      }
    ctx.stroke();
  });
  ctx.globalCompositeOperation = "source-over";

  // The image, re-assembled at the learned coordinates.
  const P0 = rope[l0].posIdx, P1 = rope[l1].posIdx;
  const sq = unit * 0.9;
  const px = sq / win;
  S.hover = null;
  const [mx, my] = S.pointer || [-1, -1];
  for (let p = 0; p < grid * grid; p++) {
    const row = lerp(P0[2 * p], P1[2 * p], e), col = lerp(P0[2 * p + 1], P1[2 * p + 1], e);
    const [cx, cy] = to(row, col);
    const r0 = Math.floor(p / grid) * win, c0 = (p % grid) * win;
    for (let i = 0; i < win; i++)
      for (let j = 0; j < win; j++) {
        const o = ((r0 + i) * 32 + c0 + j) * 3;
        ctx.fillStyle = `rgba(${S.pixels[o]},${S.pixels[o + 1]},${S.pixels[o + 2]},0.93)`;
        ctx.fillRect(cx - sq / 2 + j * px, cy - sq / 2 + i * px, px + 0.4, px + 0.4);
      }
    if (Math.abs(mx - cx) < sq / 2 && Math.abs(my - cy) < sq / 2) {
      S.hover = { kind: "drift", patch: p, row, col };
      ctx.strokeStyle = rgba(PAPER, 0.9);
      ctx.strokeRect(cx - sq / 2, cy - sq / 2, sq, sq);
    }
  }

  // Title.
  const shown = S.driftPin >= 0 ? S.driftPin : e > 0.5 ? l1 : l0;
  ctx.textAlign = "left";
  ctx.textBaseline = "alphabetic";
  ctx.fillStyle = rgba(layerColor(shown, layers));
  ctx.font = "italic 300 30px Fraunces, Georgia, serif";
  ctx.fillText(`block ${shown + 1}`, ox, oy - 18);
  ctx.fillStyle = "rgba(139,134,118,0.9)";
  ctx.font = "10px 'JetBrains Mono', monospace";
  ctx.fillText(S.ropeLearned
    ? "where this block's RoPE believes each patch sits · dots = the lattice it started from"
    : "fixed RoPE: positions sit exactly on the lattice in every block", ox, oy - 4);

  // Learned rotary frequencies vs. the 100^(-j/n) they were initialized to.
  const fx = ox + size + 50, fy = oy + size - fh;
  const nf = rope[0].freq.length;
  const ly = (f) => fy + fh - clamp((Math.log10(Math.max(f, 1e-3)) + 2.2) / 2.5, 0, 1) * fh;
  const lx = (i) => fx + (i / (nf - 1)) * fw;
  ctx.strokeStyle = "rgba(235,230,218,0.1)";
  ctx.strokeRect(fx, fy, fw, fh);
  ctx.setLineDash([3, 3]);
  ctx.strokeStyle = "rgba(235,230,218,0.5)";
  ctx.beginPath();
  for (let i = 0; i < nf; i++) ctx[i ? "lineTo" : "moveTo"](lx(i), ly(Math.pow(100, -i / nf)));
  ctx.stroke();
  ctx.setLineDash([]);
  rope.forEach((L, l) => {
    ctx.strokeStyle = rgba(layerColor(l, layers), l === shown ? 1 : 0.35);
    ctx.lineWidth = l === shown ? 1.8 : 1;
    ctx.beginPath();
    L.freq.forEach((f, i) => ctx[i ? "lineTo" : "moveTo"](lx(i), ly(Math.abs(f))));
    ctx.stroke();
  });
  ctx.fillStyle = "rgba(139,134,118,0.9)";
  ctx.fillText("RoPE freqs (log) · dashed = init", fx, fy - 8);

  if (S.hover) {
    const { patch, row, col } = S.hover;
    showTooltip(`<b>${tokenName(patch + 1)}</b> · learned at (${row.toFixed(2)}, ${col.toFixed(2)})`);
  } else hideTooltip();
}

// ---------- verdict ----------
function renderVerdict() {
  const { probs } = S.result;
  const order = [...probs.keys()].sort((a, b) => probs[b] - probs[a]);
  const top = order[0];
  let html = `<div class="top" style="color:${rgba(CLASS_COLORS[top])}">${S.classes[top]}</div>`;
  html += `<div class="pct">${(probs[top] * 100).toFixed(1)}%</div>`;
  html += `<div class="runners">${order.slice(1, 4).map((c) => `${S.classes[c]} ${(probs[c] * 100).toFixed(0)}%`).join(" · ")}</div>`;
  if (S.label != null) {
    const hit = S.label === top;
    html += `<div class="truth ${hit ? "hit" : "miss"}">label · <b>${S.classes[S.label]}</b> ${hit ? "✓" : "✗"}</div>`;
  }
  $("verdict").innerHTML = html;
}

function drawBloom(ctx, W, H) {
  const probs = S.result.probs;
  const cx = W / 2, cy = H / 2, R = Math.min(W, H) * 0.34;
  const n = probs.length;
  const top = probs.indexOf(Math.max(...probs));
  ctx.strokeStyle = "rgba(235,230,218,0.07)";
  for (const f of [0.25, 0.5, 0.75, 1]) {
    ctx.beginPath();
    ctx.arc(cx, cy, R * (0.1 + 0.9 * Math.sqrt(f)), 0, Math.PI * 2);
    ctx.stroke();
  }
  ctx.globalCompositeOperation = "lighter";
  for (let c = 0; c < n; c++) {
    const p = probs[c];
    const ang = -Math.PI / 2 + (c / n) * Math.PI * 2;
    const len = R * (0.1 + 0.9 * Math.sqrt(p));
    const w = R * 0.2 * (0.35 + 0.65 * Math.sqrt(p));
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(ang);
    const g = ctx.createLinearGradient(0, 0, len, 0);
    g.addColorStop(0, rgba(CLASS_COLORS[c], 0.04));
    g.addColorStop(1, rgba(CLASS_COLORS[c], 0.3 + 0.65 * Math.sqrt(p)));
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.bezierCurveTo(len * 0.35, -w, len * 0.8, -w * 0.7, len, 0);
    ctx.bezierCurveTo(len * 0.8, w * 0.7, len * 0.35, w, 0, 0);
    ctx.fill();
    ctx.restore();
  }
  ctx.globalCompositeOperation = "source-over";
  ctx.font = "10px 'JetBrains Mono', monospace";
  ctx.textBaseline = "middle";
  for (let c = 0; c < n; c++) {
    const ang = -Math.PI / 2 + (c / n) * Math.PI * 2;
    const x = cx + Math.cos(ang) * (R + 14), y = cy + Math.sin(ang) * (R + 14);
    ctx.textAlign = Math.abs(Math.cos(ang)) < 0.2 ? "center" : Math.cos(ang) > 0 ? "left" : "right";
    ctx.fillStyle = c === top ? rgba(CLASS_COLORS[c]) : "rgba(139,134,118,0.85)";
    ctx.fillText(S.classes[c], x, y);
  }
}

function drawLens(ctx, W, H) {
  const lens = S.result.lens;
  const n = lens.length, C = lens[0].length;
  const top = S.result.probs.indexOf(Math.max(...S.result.probs));
  const padL = 4, padR = 4, padT = 4, padB = 22;
  const x = (d) => padL + (d / (n - 1)) * (W - padL - padR);
  const y = (v) => padT + v * (H - padT - padB);
  // Stack classes with the final winner on top so its ribbon reads clearly.
  const order = [...Array(C).keys()].filter((c) => c !== top).concat(top);
  const base = new Array(n).fill(0);
  const bands = order.map((c) => {
    const lo = base.slice();
    for (let d = 0; d < n; d++) base[d] += lens[d][c];
    return { c, lo, hi: base.slice() };
  });
  const smooth = (pts, move) => {
    pts.forEach(([px, py], i) => {
      if (i === 0) { ctx[move ? "moveTo" : "lineTo"](px, py); return; }
      const [ax, ay] = pts[i - 1];
      const mx = (ax + px) / 2;
      ctx.bezierCurveTo(mx, ay, mx, py, px, py);
    });
  };
  for (const { c, lo, hi } of bands) {
    const upper = hi.map((v, d) => [x(d), y(v)]);
    const lower = lo.map((v, d) => [x(d), y(v)]).reverse();
    ctx.beginPath();
    smooth(upper, true);
    smooth(lower, false);
    ctx.closePath();
    ctx.fillStyle = rgba(CLASS_COLORS[c], c === top ? 0.9 : 0.28);
    ctx.fill();
  }
  ctx.font = "10px 'JetBrains Mono', monospace";
  ctx.textBaseline = "alphabetic";
  ctx.fillStyle = "rgba(139,134,118,0.9)";
  for (let d = 0; d < n; d++) {
    ctx.textAlign = d === 0 ? "left" : d === n - 1 ? "right" : "center";
    ctx.fillText(d === 0 ? "embed" : String(d), x(d), H - 6);
  }
}

// ---------- tooltip & stage input ----------
const tooltip = $("tooltip");
function showTooltip(html) {
  if (!S.pointer) return;
  tooltip.hidden = false;
  if (tooltip.innerHTML !== html) tooltip.innerHTML = html;
  const w = tooltip.offsetWidth;
  // Flip to the left of the pointer before running into the stage edge or the reading card.
  let right = $("stageWrap").clientWidth;
  const card = $("reading");
  if (!card.hidden) right = Math.min(right, card.getBoundingClientRect().left - $("stageWrap").getBoundingClientRect().left);
  const x = S.pointer[0] + w + 30 > right ? S.pointer[0] - w - 28 : S.pointer[0];
  tooltip.style.left = `${x}px`;
  tooltip.style.top = `${S.pointer[1]}px`;
}
function hideTooltip() { tooltip.hidden = true; }

const stage = $("stage");
let drag = null;
stage.addEventListener("pointermove", (e) => {
  const r = stage.getBoundingClientRect();
  S.pointer = [e.clientX - r.left, e.clientY - r.top];
  if (drag) {
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    if (Math.abs(dx) + Math.abs(dy) > 3) { S.dragging = true; stage.classList.add("dragging"); }
    if (S.dragging && S.mode === "tower" && S.variant === "stack") {
      S.yaw = drag.yaw + dx * 0.006;
      S.pitch = clamp(drag.pitch + dy * 0.004, 0.12, 1.2);
    }
  }
});
stage.addEventListener("pointermove", () => invalidate());
stage.addEventListener("pointerleave", () => { S.pointer = null; S.hover = null; hideTooltip(); invalidate(); });
let wheelAcc = 0;
stage.addEventListener("wheel", (e) => {
  if (S.mode !== "tower") return;
  e.preventDefault();
  wheelAcc += e.deltaY;
  // Scrolling up climbs the tower toward deeper blocks.
  if (Math.abs(wheelAcc) > 60) { setBlock(S.block - Math.sign(wheelAcc)); wheelAcc = 0; }
}, { passive: false });
stage.addEventListener("pointerdown", (e) => {
  stage.setPointerCapture(e.pointerId);
  const r = stage.getBoundingClientRect();
  S.pointer = [e.clientX - r.left, e.clientY - r.top];
  drag = { x: e.clientX, y: e.clientY, yaw: S.yaw, pitch: S.pitch };
  invalidate();
});
stage.addEventListener("pointerup", () => {
  // Resolved after the next draw, which is where hover hit-testing happens.
  if (!S.dragging) S.click = true;
  drag = null;
  S.dragging = false;
  stage.classList.remove("dragging");
});
function handleClick() {
  S.click = false;
  const h = S.hover;
  if (S.mode === "tower") {
    // A token becomes the query, and the block that produced its depth becomes active.
    if (!h) return;
    if (h.kind === "ghost") return setBlock(Math.max(0, h.k - 2));
    if (h.k >= 1) setBlock(Math.max(0, h.k - 2));
    setQuery(h.t);
  } else if (S.mode === "atlas") {
    if (h) setQuery(h.token === S.query ? 0 : h.token);
    else setQuery(0);
  }
}

// ---------- input overlay: rollout ----------
function drawOverlay(ctx, W, H) {
  if (!S.showRollout) return;
  const { grid, nTokens: T } = S.cfg;
  const R = S.result.rollout;
  let max = 1e-6;
  for (let p = 1; p < T; p++) max = Math.max(max, R[S.query * T + p]);
  const t = W / grid;
  ctx.fillStyle = "rgba(10,10,15,0.35)";
  ctx.fillRect(0, 0, W, H);
  ctx.globalCompositeOperation = "lighter";
  for (let p = 0; p < grid * grid; p++) {
    const v = R[S.query * T + p + 1] / max;
    ctx.fillStyle = rgba(GOLD, 0.55 * v ** 1.5);
    ctx.fillRect((p % grid) * t, Math.floor(p / grid) * t, t, t);
  }
  ctx.globalCompositeOperation = "source-over";
  if (S.query > 0) {
    const q = S.query - 1;
    ctx.strokeStyle = rgba(PAPER, 0.9);
    ctx.lineWidth = 1.5;
    ctx.strokeRect((q % grid) * t + 1, Math.floor(q / grid) * t + 1, t - 2, t - 2);
  }
}

// ---------- loop ----------
const ro = new ResizeObserver(() => invalidate(true));
[stage, $("bloom"), $("lens"), $("frame")].forEach((el) => ro.observe(el));

let last = performance.now(), lastHover = "";
function frameLoop(now) {
  // Schedule first so one bad frame can't stop the loop for good.
  requestAnimationFrame(frameLoop);
  const dt = Math.min(0.05, (now - last) / 1000);
  last = now;
  S.t += dt;
  if (!S.result || !S.colors) return;

  // Ease the scrub accordion toward the active block.
  if (Math.abs(S.blockF - S.block) > 0.002) { S.blockF += (S.block - S.blockF) * Math.min(1, dt * 9); dirty = true; }
  else S.blockF = S.block;
  if (S.mode === "drift" && S.driftPin < 0) dirty = true;

  if (dirty) {
    dirty = false;
    const [ctx, W, H] = fit(stage);
    ctx.clearRect(0, 0, W, H);
    const wide = W >= 760;
    if (S.mode === "tower") {
      // On wide screens the reading card takes a column on the right.
      S.hover = drawTower(ctx, wide ? W - 250 : W, H, S);
      if (S.hover) showTooltip(towerTooltip(S, S.hover));
      else hideTooltip();
    } else if (S.mode === "atlas") drawAtlas(ctx, W, H);
    else drawDrift(ctx, W, H);
    // A new hover target needs one more pass to paint its highlight.
    const key = JSON.stringify(S.hover);
    if (key !== lastHover) { lastHover = key; dirty = true; }
    if (S.click) handleClick();
    updateReading(wide);
  }
  if (panelsDirty) {
    panelsDirty = false;
    const [bctx, bW, bH] = fit($("bloom"));
    bctx.clearRect(0, 0, bW, bH);
    drawBloom(bctx, bW, bH);
    const [lctx, lW, lH] = fit($("lens"));
    lctx.clearRect(0, 0, lW, lH);
    drawLens(lctx, lW, lH);
    const [octx, oW, oH] = fit($("inputOverlay"));
    octx.clearRect(0, 0, oW, oH);
    drawOverlay(octx, oW, oH);
  }
}

buildSwatches();
requestAnimationFrame(frameLoop);
