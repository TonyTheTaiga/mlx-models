// Lineage: which image patches shaped one token?
//
// The target is a token (a patch or CLS) at some depth. Its sources are the 64 image patches,
// scored by one of two methods:
//   occlusion — replace a patch with the image's mean color and measure how far the target's
//               vector moves (causal, includes the MLPs; computed by a sweep in a worker)
//   rollout   — attention paths back to the embedding (heads averaged, residual counted)
// The mosaic sizes and lights each patch by its share; the strip below repeats the question at
// every block, so you can watch the receptive field grow.

import { rgba, GOLD, PAPER, MUTED, MONO, serif } from "./util.js";
import { rolloutFrom, tokenName, depthName } from "./tower.js";

// Shares over the 64 patches (summing to 1 with `cls` for rollout), or { ready: false }.
export function lineage(S, method = S.lin.method, k = S.lin.k, t = S.lin.t) {
  const n = S.cfg.grid * S.cfg.grid;
  if (method === "rollout") {
    const e = rolloutFrom(S, k, t)[1];
    return { ready: true, patches: e.slice(1), cls: e[0] };
  }
  const O = S.occ;
  if (!O || O.id !== S.occWant || !O.data) return { ready: false, done: O?.id === S.occWant ? O.done : 0 };
  const T = S.cfg.nTokens;
  const v = Float32Array.from({ length: n }, (_, p) => O.data[(p * O.depths + (k - 1)) * T + t]);
  const sum = v.reduce((a, b) => a + b, 0) || 1;
  return { ready: true, patches: v.map((x) => x / sum), cls: 0 };
}

export function ranking(patches) {
  return [...patches.keys()].sort((a, b) => patches[b] - patches[a]);
}

const other = (m) => (m === "occlusion" ? "rollout" : "occlusion");

function drawPatch(ctx, S, p, x, y, size, alpha) {
  const { grid, win } = S.cfg;
  const r0 = Math.floor(p / grid) * win, c0 = (p % grid) * win, px = size / win;
  for (let i = 0; i < win; i++)
    for (let j = 0; j < win; j++) {
      const o = ((r0 + i) * 32 + c0 + j) * 3;
      ctx.fillStyle = `rgba(${S.pixels[o]},${S.pixels[o + 1]},${S.pixels[o + 2]},${alpha})`;
      ctx.fillRect(x + j * px, y + i * px, px + 0.35, px + 0.35);
    }
}

export function drawLineage(ctx, W, H, S) {
  const { grid, layers } = S.cfg;
  const n = grid * grid;
  const { k, t } = S.lin;
  const L = lineage(S);
  const top = 70, bottom = 64;
  const size = Math.max(160, Math.min(W - 80, (H - top - bottom - 60) / (1 + 1 / 6.4 + 0.12)));
  const x0 = (W - size) / 2, y0 = top + 58, cell = size / grid;
  const center = (p) => [x0 + ((p % grid) + 0.5) * cell, y0 + (Math.floor(p / grid) + 0.5) * cell];
  const orbAt = [x0 + size / 2, y0 - 30];
  const hovered = S.hover;

  // Title.
  ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
  ctx.font = serif(24, 400); ctx.fillStyle = rgba(PAPER, 0.95);
  ctx.fillText(`${tokenName(S.cfg, t)}`, x0 - 10, top + 8);
  const tw = ctx.measureText(tokenName(S.cfg, t)).width;
  ctx.font = serif(16, 300); ctx.fillStyle = rgba(MUTED, 0.95);
  ctx.fillText(`at ${depthName(k)} — where did it come from?`, x0 - 10 + tw + 10, top + 8);

  // The plate.
  ctx.fillStyle = "rgba(13,13,20,0.9)";
  ctx.beginPath(); ctx.roundRect(x0 - 12, y0 - 12, size + 24, size + 24, 10); ctx.fill();
  ctx.strokeStyle = rgba(PAPER, 0.08); ctx.lineWidth = 1; ctx.stroke();

  let hover = null;
  const [mx, my] = S.pointer || [-1, -1];

  if (!L.ready) {
    for (let p = 0; p < n; p++) drawPatch(ctx, S, p, x0 + (p % grid) * cell, y0 + Math.floor(p / grid) * cell, cell, 0.25);
    ctx.font = MONO; ctx.textAlign = "center"; ctx.fillStyle = rgba(GOLD, 0.9);
    ctx.fillText(`occluding patches… ${L.done}/${n}`, x0 + size / 2, y0 + size / 2);
  } else {
    const vals = L.patches, max = Math.max(...vals) || 1;
    const order = ranking(vals);
    // Glow under the strongest sources.
    for (const p of order.slice(0, 12)) {
      const r = vals[p] / max, [cx, cy] = center(p);
      const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, cell * 1.1);
      g.addColorStop(0, rgba(GOLD, 0.4 * r)); g.addColorStop(1, rgba(GOLD, 0));
      ctx.fillStyle = g; ctx.fillRect(cx - cell * 1.1, cy - cell * 1.1, cell * 2.2, cell * 2.2);
    }
    // Every patch, sized and lit by its share.
    for (let p = 0; p < n; p++) {
      const r = vals[p] / max, s = cell * (0.3 + 0.64 * Math.sqrt(r)), [cx, cy] = center(p);
      drawPatch(ctx, S, p, cx - s / 2, cy - s / 2, s, 0.22 + 0.78 * r);
    }
    // The target.
    if (t > 0) {
      const [cx, cy] = center(t - 1);
      ctx.strokeStyle = rgba(GOLD, 0.95); ctx.lineWidth = 2;
      ctx.strokeRect(cx - cell / 2 + 1, cy - cell / 2 + 1, cell - 2, cell - 2);
    } else {
      const g = ctx.createRadialGradient(...orbAt, 0, ...orbAt, 22);
      g.addColorStop(0, rgba(GOLD, 0.7)); g.addColorStop(1, rgba(GOLD, 0));
      ctx.fillStyle = g; ctx.beginPath(); ctx.arc(...orbAt, 22, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = rgba(GOLD); ctx.beginPath(); ctx.arc(...orbAt, 6, 0, Math.PI * 2); ctx.fill();
      ctx.font = MONO; ctx.textAlign = "left"; ctx.fillStyle = rgba(GOLD, 0.9);
      ctx.fillText("CLS", orbAt[0] + 14, orbAt[1] + 4);
    }
    // Threads from the top five sources into the target, with rank badges.
    const T0 = t > 0 ? center(t - 1) : orbAt;
    ctx.lineCap = "round";
    order.slice(0, 5).forEach((p, i) => {
      const [cx, cy] = center(p);
      if (p !== t - 1) {
        const mxp = (cx + T0[0]) / 2, myp = (cy + T0[1]) / 2, dx = T0[0] - cx, dy = T0[1] - cy;
        const bend = 0.18 * (i % 2 ? 1 : -1);
        ctx.strokeStyle = rgba(GOLD, 0.9 - i * 0.14); ctx.lineWidth = 2.8 - i * 0.4;
        ctx.beginPath(); ctx.moveTo(cx, cy); ctx.quadraticCurveTo(mxp - dy * bend, myp + dx * bend, T0[0], T0[1]); ctx.stroke();
      }
      const bx = cx - cell / 2 + 9, by = cy - cell / 2 + 9;
      ctx.fillStyle = rgba(GOLD); ctx.beginPath(); ctx.arc(bx, by, 7, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = "#0a0a0f"; ctx.font = "600 9px 'JetBrains Mono', monospace"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillText(String(i + 1), bx, by + 0.5);
    });
    ctx.textBaseline = "alphabetic";
    if (hovered?.kind === "lin-patch") {
      const [cx, cy] = center(hovered.p);
      ctx.strokeStyle = rgba(PAPER, 0.9); ctx.lineWidth = 1.2;
      ctx.strokeRect(cx - cell / 2 + 0.5, cy - cell / 2 + 0.5, cell - 1, cell - 1);
    }
  }
  if (mx >= x0 && mx < x0 + size && my >= y0 && my < y0 + size)
    hover = { kind: "lin-patch", p: Math.floor((my - y0) / cell) * grid + Math.floor((mx - x0) / cell) };
  else if (Math.hypot(mx - orbAt[0], my - orbAt[1]) < 16) hover = { kind: "lin-cls" };

  // The receptive field, block by block, for the same token.
  const gap = 10, mw = (size - gap * (layers - 1)) / layers, y1 = y0 + size + 40, mc = mw / grid;
  for (let b = 1; b <= layers; b++) {
    const kk = b + 1, mx0 = x0 + (b - 1) * (mw + gap), Lb = lineage(S, S.lin.method, kk, t);
    for (let p = 0; p < n; p++) drawPatch(ctx, S, p, mx0 + (p % grid) * mc, y1 + Math.floor(p / grid) * mc, mc, 0.22);
    let share5 = null;
    if (Lb.ready) {
      const max = Math.max(...Lb.patches) || 1;
      for (let p = 0; p < n; p++) {
        const r = Lb.patches[p] / max;
        ctx.fillStyle = rgba(GOLD, 0.85 * Math.pow(r, 1.2));
        ctx.fillRect(mx0 + (p % grid) * mc, y1 + Math.floor(p / grid) * mc, mc + 0.3, mc + 0.3);
      }
      share5 = ranking(Lb.patches).slice(0, 5).reduce((s, p) => s + Lb.patches[p], 0);
    }
    const active = kk === k, hov = hovered?.kind === "lin-depth" && hovered.k === kk;
    ctx.strokeStyle = active ? rgba(GOLD, 0.95) : rgba(PAPER, hov ? 0.6 : 0.12);
    ctx.lineWidth = active ? 1.6 : 1;
    ctx.strokeRect(mx0 - 0.5, y1 - 0.5, mw + 1, mw + 1);
    ctx.textAlign = "center"; ctx.font = active ? serif(14, 400) : MONO;
    ctx.fillStyle = active ? rgba(PAPER, 0.95) : rgba(MUTED, 0.9);
    ctx.fillText(`block ${b}`, mx0 + mw / 2, y1 + mw + 16);
    if (share5 != null) {
      ctx.font = MONO; ctx.fillStyle = rgba(MUTED, 0.8);
      ctx.fillText(`top 5 = ${(share5 * 100).toFixed(0)}%`, mx0 + mw / 2, y1 + mw + 30);
    }
    if (mx >= mx0 && mx < mx0 + mw && my >= y1 && my < y1 + mw) hover = { kind: "lin-depth", k: kk };
  }
  ctx.textAlign = "left"; ctx.font = MONO; ctx.fillStyle = rgba(MUTED, 0.85);
  ctx.fillText("the same token after each block — how far its sources reach", x0 - 10, y1 - 10);
  return hover;
}

export function lineageTooltip(S, h) {
  if (h.kind === "lin-cls") return "<b>CLS</b> · click to trace the class token";
  if (h.kind === "lin-depth") return `<b>${depthName(h.k)}</b> · click to trace ${tokenName(S.cfg, S.lin.t)} from here`;
  const L = lineage(S);
  if (!L.ready) return `<b>${tokenName(S.cfg, h.p + 1)}</b>`;
  const rank = ranking(L.patches).indexOf(h.p) + 1;
  return `<b>${tokenName(S.cfg, h.p + 1)}</b> · ${(L.patches[h.p] * 100).toFixed(1)}% · rank #${rank} · click to make it the target`;
}

export function lineageHtml(S) {
  const { k, t, method } = S.lin;
  const L = lineage(S);
  const head = `<div class="rh"><span class="serif">lineage</span> · <b>${tokenName(S.cfg, t)}</b> at ${depthName(k)} is shaped most by</div>`;
  if (!L.ready) return head + `<div class="rf">occluding patches… ${L.done}/${S.cfg.grid * S.cfg.grid}</div>`;
  const order = ranking(L.patches);
  const rows = order.slice(0, 5).map((p) =>
    `<div class="rr"><canvas data-t="${p + 1}" width="4" height="4"></canvas><span class="dot" style="background:${rgba(GOLD)}"></span>` +
    `<span class="nm">${tokenName(S.cfg, p + 1)}${p === t - 1 ? " (itself)" : ""}</span><span class="pct">${(L.patches[p] * 100).toFixed(0)}%</span>` +
    `<span class="bar"><i style="width:${(L.patches[p] / L.patches[order[0]]) * 100}%;background:${rgba(GOLD)}"></i></span></div>`).join("");
  const how = method === "occlusion"
    ? "occlusion: each patch replaced by the image's mean color; share of how far this token moves"
    : `rollout: heads averaged, residual counted, MLPs ignored · ${(L.cls * 100).toFixed(0)}% goes to the CLS embedding`;
  // How the other method ranks this method's top patch.
  const O = lineage(S, other(method));
  let cmp = "";
  if (O.ready) {
    const oOrder = ranking(O.patches), r = oOrder.indexOf(order[0]) + 1;
    cmp = `<div class="rf">${other(method)} ${r === 1 ? "agrees on the top patch" : `ranks it #${r}; its top is ${tokenName(S.cfg, oOrder[0] + 1)}`}</div>`;
  }
  return head + rows + `<div class="rf">${how}</div>` + cmp;
}
