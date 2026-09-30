// Heatmaps: where the model looked for its prediction.
//
// Two methods, both checked for faithfulness in export.py (hiding the hottest patches first
// collapses the prediction far faster than random):
//   occlusion — drop in log p(class) when each patch is flattened to the image's mean color
//   gradient  — ∂logit/∂pixel · (pixel − mean), a per-pixel first-order estimate of the same thing
// Grad-CAM and last-block CLS attention were scored too and came out near random, so they aren't
// offered as maps.

// Semantic heat ramp (dark red → orange → pale yellow), transparent at zero, always shown with
// a scale legend.
const STOPS = [[92, 18, 16], [179, 38, 30], [232, 89, 12], [245, 166, 35], [255, 232, 163]];
export function heat(t) {
  t = Math.max(0, Math.min(1, t));
  const x = t * (STOPS.length - 1), i = Math.min(STOPS.length - 2, Math.floor(x)), f = x - i;
  const c = STOPS[i].map((a, k) => Math.round(a + (STOPS[i + 1][k] - a) * f));
  return [...c, t < 0.04 ? 0 : Math.round(255 * (0.25 + 0.7 * Math.pow(t, 0.7)))];
}
export const HEAT_CSS = `linear-gradient(90deg, ${STOPS.map((c) => `rgb(${c})`).join(", ")})`;

// Normalized positive map (values 0..1) at its native resolution, or null if unavailable.
export function galleryMap(H, index, method) {
  if (!H) return null;
  if (method === "gradient") {
    const px = H.gradient.subarray(index * 1024, (index + 1) * 1024);
    return { res: 32, values: Float32Array.from(px, (v) => v / 255) };
  }
  return positive(H.occlusion[index], 8);
}
export function positive(scores, res) {
  const max = Math.max(1e-9, ...scores);
  return { res, values: Float32Array.from(scores, (v) => Math.max(0, v) / max) };
}

// Base image (dimmed grayscale, pixel-sharp) under a smooth heat layer.
const tmp = document.createElement("canvas");
export function drawHeat(ctx, pixels, map, x0, y0, size, { dim = 0.45, showHeat = true } = {}) {
  const px = size / 32;
  for (let y = 0; y < 32; y++)
    for (let x = 0; x < 32; x++) {
      const o = (y * 32 + x) * 3;
      const g = (0.3 * pixels[o] + 0.59 * pixels[o + 1] + 0.11 * pixels[o + 2]) * dim;
      const c = showHeat ? `rgb(${g},${g},${g})` : `rgb(${pixels[o]},${pixels[o + 1]},${pixels[o + 2]})`;
      ctx.fillStyle = c;
      ctx.fillRect(x0 + x * px, y0 + y * px, px + 0.4, px + 0.4);
    }
  if (!showHeat || !map) return;
  tmp.width = tmp.height = map.res;
  const tctx = tmp.getContext("2d"), img = tctx.createImageData(map.res, map.res);
  map.values.forEach((v, i) => img.data.set(heat(v), i * 4));
  tctx.putImageData(img, 0, 0);
  ctx.save();
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(tmp, x0, y0, size, size);
  ctx.restore();
}

// ---------- the Heatmaps tab ----------
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

const state = { method: "occlusion", filter: "all", raw: false };

export function renderHeatmaps(root, S, api) {
  const H = S.heatmaps;
  root.replaceChildren();
  if (!H) {
    root.append(h("p", { class: "report-empty" }, "No heatmaps yet. Run ", h("code", {}, "uv run python viz/vit/export.py"), ", then reload."));
    return;
  }
  const F = H.faithfulness, n = S.labels.length;
  const idx = [...Array(n).keys()].filter((i) =>
    state.filter === "all" ? true : state.filter === "mistakes" ? S.preds[i] !== S.labels[i] : S.labels[i] === +state.filter);

  const seg = (items, cur, set) => h("div", { class: "seg" }, items.map(([k, label]) =>
    h("button", { class: k === cur ? "on" : "", "aria-pressed": k === cur, onclick: () => { set(k); renderHeatmaps(root, S, api); } }, label)));
  const methodNote = {
    occlusion: "Each patch hidden in turn; brighter = hiding it costs the prediction more. One value per 4×4 patch.",
    gradient: "Gradient × (pixel − mean): a per-pixel estimate of the same effect, from one backward pass. Sharper, slightly noisier.",
  }[state.method];

  const grid = h("div", { class: "hgrid" }, idx.map((i) => {
    const cv = h("canvas", { width: 128, height: 128 });
    const ctx = cv.getContext("2d");
    const px = S.samples.subarray(i * 3072, (i + 1) * 3072);
    drawHeat(ctx, px, galleryMap(H, i, state.method), 0, 0, 128, { showHeat: !state.raw });
    const right = S.preds[i] === S.labels[i];
    return h("button", { class: `htile ${right ? "" : "wrong"}`, onclick: () => api.inspectSample(i), title: "open in Inspect" },
      cv,
      h("span", { class: "cap" },
        h("span", { class: "mark" }, right ? "✓" : "✗"),
        right ? S.classes[S.preds[i]] : `${S.classes[S.preds[i]]} · is ${S.classes[S.labels[i]]}`));
  }));

  const faithRow = (k, label) => F[k] && h("tr", {}, h("th", {}, label),
    h("td", {}, F[k].deletion.toFixed(2)), h("td", {}, F[k].insertion.toFixed(2)));
  root.append(
    h("header", { class: "rhead" },
      h("h2", {}, "Where does it look?"),
      h("p", {}, `Heat over each gallery image shows which regions the prediction leaned on. ${idx.length} of ${n} images shown; click one to open it.`)),
    h("div", { class: "hcontrols" },
      seg([["occlusion", "occlusion"], ["gradient", "gradient × input"]], state.method, (k) => (state.method = k)),
      seg([["all", "all"], ["mistakes", "mistakes"]], ["all", "mistakes"].includes(state.filter) ? state.filter : "", (k) => (state.filter = k)),
      h("select", { "aria-label": "filter by class", onchange: (e) => { state.filter = e.target.value; renderHeatmaps(root, S, api); } },
        h("option", { value: "all" }, "every class"),
        S.classes.map((c, k) => { const o = h("option", { value: String(k) }, c); if (state.filter === String(k)) o.selected = true; return o; })),
      h("label", { class: "check" }, (() => { const c = h("input", { type: "checkbox" }); c.checked = state.raw; c.onchange = () => { state.raw = c.checked; renderHeatmaps(root, S, api); }; return c; })(), " show images without heat")),
    h("p", { class: "sub" }, methodNote),
    h("div", { class: "scale" }, h("span", {}, "ignored"), h("span", { class: "bar", style: `background:${HEAT_CSS}` }), h("span", {}, "leaned on most")),
    grid,
    h("section", { class: "card" },
      h("h3", {}, "Can these maps be trusted?"),
      h("p", { class: "sub" }, "Each method's ranking tested on these images: flatten patches hottest-first (deletion; a faithful map makes confidence collapse fast, so lower is better) or reveal them hottest-first on a flat image (insertion; higher is better). Values are the mean confidence in the prediction across k = 0…64."),
      h("div", { class: "scroll" }, h("table", { class: "faith" },
        h("thead", {}, h("tr", {}, h("th", {}, "method"), h("th", {}, "deletion ↓"), h("th", {}, "insertion ↑"))),
        h("tbody", {}, faithRow("occlusion", "occlusion"), faithRow("gradient", "gradient × input"),
          faithRow("gradcam", "Grad-CAM (not shown)"), faithRow("cls_attention", "last-block CLS attention (not shown)"), faithRow("random", "random order")))),
      h("p", { class: "sub" }, "Grad-CAM and attention maps score close to random order for this model, so they aren't offered: they'd look plausible without reflecting what drives the prediction.")));
}
