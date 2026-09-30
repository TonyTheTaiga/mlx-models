// Evidence: which patches did the model lean on for its final prediction?
//
// Each patch is scored by occlusion against the prediction itself: how much the log-probability
// of the target class drops when that patch is flattened to the image's mean color (positive =
// the patch supports the class, negative = it argues against it). The ranking is then checked
// two ways: keep only the top-k patches (does the prediction survive?) and remove them (does it
// collapse?). The smallest k that keeps the prediction is the headline.

import { clamp, rgba, oklch, GOLD, PAPER, MUTED, CLASS_COLORS, MONO } from "./util.js";
import { softmax } from "./model.js";

const AGAINST = oklch(0.68, 0.12, 250);
const logSoftmax = (v) => softmax(v).map((p) => Math.log(Math.max(p, 1e-12)));
export const patchName = (cfg, p) => `patch ${Math.floor(p / cfg.grid)},${p % cfg.grid}`;

export const predicted = (S) => S.result.probs.indexOf(Math.max(...S.result.probs));
export const targetClass = (S) => S.ev.cls ?? predicted(S);

// Per-patch scores for the target class, or { ready: false } while the sweep runs.
export function evidence(S) {
  const O = S.occ;
  if (!S.result || !O || O.id !== S.occWant || !O.logits) return { ready: false, done: O?.id === S.occWant ? O.done : 0 };
  const c = targetClass(S), nc = S.classes.length, n = S.cfg.grid * S.cfg.grid;
  const base = logSoftmax(S.result.logits)[c];
  const scores = Float32Array.from({ length: n }, (_, p) => base - logSoftmax(O.logits.subarray(p * nc, p * nc + nc))[c]);
  const order = [...scores.keys()].sort((a, b) => scores[b] - scores[a]);
  return { ready: true, c, scores, order };
}

// Keep/remove curves for the current ranking: p(target) and the winning class at each k.
export function curvePoints(S) {
  const C = S.curves, E = evidence(S);
  if (!E.ready || !C || C.id !== S.curvesWant || !C.keep) return null;
  const nc = S.classes.length, n = S.cfg.grid * S.cfg.grid;
  const at = (arr, k) => {
    const p = softmax(arr.subarray(k * nc, k * nc + nc));
    return { p: p[E.c], top: p.indexOf(Math.max(...p)) };
  };
  const keep = Array.from({ length: n + 1 }, (_, k) => at(C.keep, k));
  const drop = Array.from({ length: n + 1 }, (_, k) => at(C.drop, k));
  // For the prediction: the smallest k whose kept patches alone still give the same answer at
  // ≥ 80% of the confidence. For another class there's no answer to keep, so instead: the
  // smallest k whose patches alone flip the answer to it — or, if none do, the k where they make
  // the strongest case.
  const pBase = S.result.probs[E.c];
  let enough = null, flip = null, peak = 1;
  if (E.c === predicted(S)) {
    enough = n;
    for (let k = 1; k <= n; k++) if (keep[k].top === E.c && keep[k].p >= 0.8 * pBase) { enough = k; break; }
  } else {
    for (let k = 1; k <= n; k++) if (keep[k].top === E.c) { flip = k; break; }
    for (let k = 1; k <= n; k++) if (keep[k].p > keep[peak].p + 1e-6) peak = k;
  }
  return { keep, drop, enough, flip, peak, best: enough ?? flip ?? peak };
}

export const currentK = (S) => {
  const n = S.cfg.grid * S.cfg.grid;
  return clamp(S.ev.k ?? curvePoints(S)?.best ?? 8, 1, n);
};

function meanColor(S) {
  const m = [0, 0, 0];
  for (let i = 0; i < 1024; i++) for (let c = 0; c < 3; c++) m[c] += S.pixels[i * 3 + c] / 1024;
  return m.map(Math.round);
}

// The image, square and centered in the canvas: kept patches in full, the rest flattened to the
// mean color (what the model had to go on), outlined and ranked by support.
export function drawFigure(ctx, W, H, S) {
  const { grid, win } = S.cfg;
  const n = grid * grid;
  const E = evidence(S);
  const size = Math.min(W, H), x0 = (W - size) / 2, y0 = 0, cell = size / grid, px = cell / win;
  const c = targetClass(S), col = CLASS_COLORS[c], m = meanColor(S);
  const k = E.ready ? currentK(S) : n;
  const rank = new Array(n);
  if (E.ready) E.order.forEach((p, i) => (rank[p] = i));

  for (let p = 0; p < n; p++) {
    const x = x0 + (p % grid) * cell, y = y0 + Math.floor(p / grid) * cell;
    const kept = !E.ready || rank[p] < k;
    for (let i = 0; i < win; i++)
      for (let j = 0; j < win; j++) {
        const o = ((Math.floor(p / grid) * win + i) * 32 + (p % grid) * win + j) * 3;
        ctx.fillStyle = kept ? `rgb(${S.pixels[o]},${S.pixels[o + 1]},${S.pixels[o + 2]})` : `rgb(${m[0]},${m[1]},${m[2]})`;
        ctx.fillRect(x + j * px, y + i * px, px + 0.4, px + 0.4);
      }
    if (!kept) { ctx.fillStyle = "rgba(10,10,15,0.55)"; ctx.fillRect(x, y, cell + 0.4, cell + 0.4); }
  }

  if (E.ready) {
    const maxPos = Math.max(1e-6, ...E.scores), maxNeg = Math.max(1e-6, ...E.scores.map((v) => -v));
    for (let p = 0; p < n; p++) {
      const x = x0 + (p % grid) * cell, y = y0 + Math.floor(p / grid) * cell, v = E.scores[p];
      if (S.ev.heat) {
        // Signed evidence map: class color for support, cool blue for patches that argue against.
        ctx.fillStyle = v > 0 ? rgba(col, 0.6 * Math.pow(v / maxPos, 0.8)) : rgba(AGAINST, 0.6 * Math.pow(-v / maxNeg, 0.8));
        ctx.fillRect(x, y, cell, cell);
      }
      if (rank[p] < k && v > 0) {
        ctx.strokeStyle = rgba(col, 0.35 + 0.6 * (v / maxPos));
        ctx.lineWidth = 1 + 2.5 * (v / maxPos);
        ctx.strokeRect(x + 1.5, y + 1.5, cell - 3, cell - 3);
      }
    }
    const r = Math.max(7, cell * 0.12);
    E.order.slice(0, Math.min(5, k)).forEach((p, i) => {
      const bx = x0 + (p % grid) * cell + r + 4, by = y0 + Math.floor(p / grid) * cell + r + 4;
      ctx.fillStyle = rgba(col); ctx.beginPath(); ctx.arc(bx, by, r, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = "#0a0a0f"; ctx.font = `600 ${Math.round(r * 1.25)}px 'JetBrains Mono', monospace`;
      ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(String(i + 1), bx, by + 0.5);
    });
  } else {
    ctx.fillStyle = "rgba(10,10,15,0.55)"; ctx.fillRect(x0, y0, size, size);
    ctx.font = MONO; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillStyle = rgba(GOLD, 0.95);
    ctx.fillText(`testing each patch… ${E.done}/${n}`, x0 + size / 2, y0 + size / 2);
  }
  ctx.strokeStyle = rgba(PAPER, 0.1); ctx.lineWidth = 1; ctx.strokeRect(x0 - 0.5, y0 - 0.5, size + 1, size + 1);

  const hov = S.hover;
  if (hov) {
    ctx.strokeStyle = rgba(PAPER, 0.95); ctx.lineWidth = 1.4;
    ctx.strokeRect(x0 + (hov.p % grid) * cell + 0.5, y0 + Math.floor(hov.p / grid) * cell + 0.5, cell - 1, cell - 1);
  }
  if (!S.pointer) return null;
  const [mx, my] = S.pointer;
  if (mx < x0 || mx >= x0 + size || my < y0 || my >= y0 + size) return null;
  return { p: Math.floor((my - y0) / cell) * grid + Math.floor((mx - x0) / cell) };
}

// p(target) as the top patches are kept (class color) or removed (dashed), best first.
export function drawChart(ctx, W, H, S) {
  const K = curvePoints(S);
  if (!K) return;
  const n = S.cfg.grid * S.cfg.grid, c = targetClass(S), col = CLASS_COLORS[c], k = currentK(S);
  const pad = 6, top = 18, X = (kk) => pad + (kk / n) * (W - 2 * pad), Y = (p) => top + (1 - p) * (H - top - pad - 14);
  ctx.strokeStyle = rgba(PAPER, 0.08); ctx.lineWidth = 1;
  for (const p of [0, 0.5, 1]) { ctx.beginPath(); ctx.moveTo(pad, Y(p)); ctx.lineTo(W - pad, Y(p)); ctx.stroke(); }
  const plot = (pts, color, dash) => {
    ctx.setLineDash(dash); ctx.strokeStyle = color; ctx.lineWidth = 1.8; ctx.beginPath();
    pts.forEach((pt, kk) => ctx[kk ? "lineTo" : "moveTo"](X(kk), Y(pt.p))); ctx.stroke(); ctx.setLineDash([]);
  };
  plot(K.drop, rgba(PAPER, 0.55), [4, 3]);
  plot(K.keep, rgba(col, 0.95), []);
  ctx.strokeStyle = rgba(GOLD, 0.85);
  ctx.beginPath(); ctx.moveTo(X(k), top); ctx.lineTo(X(k), Y(0)); ctx.stroke();
  ctx.fillStyle = rgba(col); ctx.beginPath(); ctx.arc(X(K.best), Y(K.keep[K.best].p), 4, 0, Math.PI * 2); ctx.fill();
  ctx.font = MONO; ctx.fillStyle = rgba(MUTED, 0.9); ctx.textBaseline = "alphabetic";
  ctx.textAlign = "left"; ctx.fillText("0", pad, H - 2);
  ctx.textAlign = "right"; ctx.fillText(`${n} patches`, W - pad, H - 2);
  ctx.textAlign = "left"; ctx.fillText(`p(${S.classes[c]})`, pad, 10);
}

const clsSpan = (S, c) => `<b style="color:${rgba(CLASS_COLORS[c])}">${S.classes[c]}</b>`;

// Model's answer against the ground truth, stated plainly before any evidence.
function verdictHtml(S) {
  const pred = predicted(S), known = S.label != null, right = known && S.label === pred;
  const state = !known ? "unknown" : S.edited ? "edited" : right ? "right" : "wrong";
  const badge = { right: "✓ right", wrong: "✗ wrong", edited: "✎ edited", unknown: "no label" }[state];
  const side = (key, c, note) =>
    `<div class="side"><span class="k">${key}</span>` +
    (c == null ? `<span class="cls none">—</span>` : `<span class="cls" style="color:${rgba(CLASS_COLORS[c])}">${S.classes[c]}</span>`) +
    `<span class="note">${note}</span></div>`;
  let html = `<div class="verdict ${state}"><span class="badge">${badge}</span>` +
    side("model says", pred, `${(S.result.probs[pred] * 100).toFixed(1)}% sure`) +
    side("true label", known ? S.label : null, !known ? "your own image" : S.edited ? "of the image before your edits" : "from CIFAR-10") +
    `</div>`;
  if (known && !right) {
    html += `<p class="miss">It gave ${S.classes[S.label]} only ${(S.result.probs[S.label] * 100).toFixed(1)}%.` +
      (targetClass(S) === S.label ? "" : ` <button data-cls="${S.label}">why not ${S.classes[S.label]}? →</button>`) + `</p>`;
  }
  return html;
}

export function answerHtml(S) {
  if (!S.result) return "";
  const n = S.cfg.grid * S.cfg.grid, c = targetClass(S), E = evidence(S), K = curvePoints(S);
  let html = verdictHtml(S) +
    `<div class="ev-head">evidence for <b style="color:${rgba(CLASS_COLORS[c])}">${S.classes[c]}</b>` +
    (c === predicted(S) ? ", the model's answer" : `, not the model's answer · p = ${(S.result.probs[c] * 100).toFixed(1)}%`) +
    (c !== predicted(S) ? ` <button data-cls="${predicted(S)}">back to ${S.classes[predicted(S)]}</button>` : "") + `</div>`;
  if (!E.ready) return html + `<p class="wait">testing each patch… ${E.done}/${n}</p>`;
  if (!K) return html + `<p class="wait">checking the ranking…</p>`;
  const k = currentK(S);
  if (K.enough != null)
    html += `<div class="enough"><span class="n">${K.enough}</span><span>of ${n} patches were enough` +
      `<small>keep only these and it still says ${S.classes[c]} at ≥80% of its confidence</small></span></div>`;
  else if (K.flip != null)
    html += `<div class="enough"><span class="n">${K.flip}</span><span>patches flip it to ${S.classes[c]}` +
      `<small>keep only these and the model says ${S.classes[c]} (${(K.keep[K.flip].p * 100).toFixed(0)}%) instead of ${S.classes[predicted(S)]}</small></span></div>`;
  else {
    const pk = K.keep[K.peak];
    html += `<div class="enough"><span class="n">${(pk.p * 100).toFixed(0)}%</span><span>its strongest case for ${S.classes[c]}` +
      `<small>no set of its top patches flips the answer; keeping the top ${K.peak} raises p(${S.classes[c]}) from ${(S.result.probs[c] * 100).toFixed(1)}%, but the model still says ${S.classes[pk.top]}</small></span></div>`;
  }
  const test = (label, pt) => `<div class="test"><span class="label">${label}</span>` +
    `<span class="res">${clsSpan(S, pt.top)} <span class="p">· ${S.classes[c]} ${(pt.p * 100).toFixed(0)}%</span></span></div>`;
  html += test(`keep only the top ${k}`, K.keep[k]) + test(`remove the top ${k}`, K.drop[k]);
  return html;
}

export function tooltipHtml(S, h) {
  const E = evidence(S);
  if (!E.ready) return `<b>${patchName(S.cfg, h.p)}</b>`;
  const v = E.scores[h.p], r = E.order.indexOf(h.p) + 1, cls = S.classes[E.c];
  const verb = v >= 0 ? `supports ${cls}: hiding it costs ${v.toFixed(2)}` : `argues against ${cls}: hiding it adds ${(-v).toFixed(2)}`;
  return `<b>${patchName(S.cfg, h.p)}</b> · rank #${r} · ${verb} log-prob · click to keep the top ${r}`;
}
