// The Tower view. The page uses scrub; loom, orbit and stack are kept as alternates (set
// S.variant to try one). All of them share one idea: depth k is a layer of tokens
// (0 = pixels, 1 = patch embedding, 2.. = output of block k-1), and one *active block* b moves
// tokens from depth b+1 to depth b+2. Only that block's strongest attention sources are drawn
// in full; the other blocks leave a faint trace.
//
//   scrub  — orthographic accordion; only the active pair of plates is open
//   loom   — flat side view: tokens are warp threads, attention is woven between rows
//   orbit  — concentric rings, the image unwrapped around its center, CLS at the heart
//   stack  — the original 3D tower, calmer: no spin, no overlap, top-k threads

import { clamp, lerp, rgba, GOLD, PAPER, MUTED, HEAD_COLORS, MONO, serif, quadPath, inQuad } from "./util.js";

export const VARIANTS = ["scrub", "loom", "orbit", "stack"];
const ACTIVE_K = 12, TRACE_K = 3;

export function tokenName(cfg, t) {
  if (t === 0) return "CLS";
  return `patch ${Math.floor((t - 1) / cfg.grid)},${(t - 1) % cfg.grid}`;
}
export const depthName = (k) => (k === 0 ? "pixels" : k === 1 ? "embed" : `block ${k - 1}`);

// Attention weight from the query into key j at block b (mean over heads unless one is soloed).
export function weight(S, b, j) {
  const { nTokens: T, heads } = S.cfg;
  const rows = S.result.attn[b];
  if (S.head >= 0) return rows[S.head][S.query * T + j];
  let a = 0;
  for (let h = 0; h < heads; h++) a += rows[h][S.query * T + j];
  return a / heads;
}

// Strongest keys for the query at block b; each carries the head that contributed most.
export function sources(S, b, limit) {
  const { nTokens: T, heads } = S.cfg;
  const rows = S.result.attn[b];
  const out = [];
  for (let j = 0; j < T; j++) {
    let h = S.head;
    if (h < 0) {
      let best = -1;
      for (let k = 0; k < heads; k++) if (rows[k][S.query * T + j] > best) { best = rows[k][S.query * T + j]; h = k; }
    }
    out.push({ j, a: weight(S, b, j), h });
  }
  return out.sort((x, y) => y.a - x.a).slice(0, limit).filter((e) => e.a > 0.012);
}

// Trace: attention rollout from one token at depth k back down to the pixels. Each block's
// heads are averaged and the residual path is counted (0.5·A + 0.5·I); MLPs and values are
// ignored, so this is the standard rollout approximation, not an exact attribution.
// Returns infl[d] for depths 1..k: how much each token at depth d feeds the traced token.
let traceMemo = { key: "", infl: null };
export function traceInfluence(S) {
  const key = [S.resultId, S.traceK, S.query].join();
  if (traceMemo.key === key) return traceMemo.infl;
  const { nTokens: T, heads } = S.cfg;
  const infl = [];
  infl[S.traceK] = Float32Array.from({ length: T }, (_, t) => (t === S.query ? 1 : 0));
  for (let d = S.traceK; d >= 2; d--) {
    const A = S.result.attn[d - 2], up = infl[d], down = new Float32Array(T);
    for (let i = 0; i < T; i++) {
      if (up[i] === 0) continue;
      down[i] += 0.5 * up[i];
      for (let h = 0; h < heads; h++) {
        const row = A[h];
        for (let j = 0; j < T; j++) down[j] += (0.5 / heads) * up[i] * row[i * T + j];
      }
    }
    infl[d - 1] = down;
  }
  traceMemo = { key, infl };
  return infl;
}
// One attention thread: a soft underglow when active, then a crisp core.
function thread(ctx, path, a, color, active, k = 1) {
  if (active) {
    ctx.strokeStyle = rgba(color, 0.08 + 0.3 * a);
    ctx.lineWidth = (3 + 20 * a) * k;
    ctx.beginPath(); path(); ctx.stroke();
  }
  ctx.strokeStyle = rgba(color, active ? clamp(0.5 + 1.5 * a, 0, 0.95) : 0.14 + 0.6 * a);
  ctx.lineWidth = (active ? 0.9 + 6 * a : 0.6 + 1.6 * a) * k;
  ctx.beginPath(); path(); ctx.stroke();
}

function patchPixels(ctx, S, p, x, y, size, alpha = 1) {
  const { grid, win } = S.cfg;
  const r0 = Math.floor(p / grid) * win, c0 = (p % grid) * win, px = size / win;
  for (let i = 0; i < win; i++)
    for (let j = 0; j < win; j++) {
      const o = ((r0 + i) * 32 + c0 + j) * 3;
      ctx.fillStyle = `rgba(${S.pixels[o]},${S.pixels[o + 1]},${S.pixels[o + 2]},${alpha})`;
      ctx.fillRect(x + j * px, y + i * px, px + 0.35, px + 0.35);
    }
}

function label(ctx, text, x, y, { active = false, align = "left", big = false } = {}) {
  ctx.textAlign = align;
  ctx.textBaseline = "middle";
  if (active && text.startsWith("block")) {
    ctx.font = serif(big ? 22 : 16, 400);
    ctx.fillStyle = rgba(PAPER, 0.95);
  } else {
    ctx.font = MONO;
    ctx.fillStyle = rgba(active ? PAPER : MUTED, active ? 0.85 : 0.75);
  }
  ctx.fillText(text, x, y);
}

function orb(ctx, x, y, r, color, { halo = false, ring = null } = {}) {
  if (halo) {
    const g = ctx.createRadialGradient(x, y, 0, x, y, r * 3);
    g.addColorStop(0, rgba(color, 0.55));
    g.addColorStop(1, rgba(color, 0));
    ctx.fillStyle = g;
    ctx.beginPath(); ctx.arc(x, y, r * 3, 0, Math.PI * 2); ctx.fill();
  }
  ctx.fillStyle = rgba(color);
  ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
  if (ring) {
    ctx.strokeStyle = ring;
    ctx.lineWidth = 1.3;
    ctx.beginPath(); ctx.arc(x, y, r + 4, 0, Math.PI * 2); ctx.stroke();
  }
}

export function drawTower(ctx, W, H, S) {
  if (S.variant === "scrub" && S.splitHeads && !S.trace && S.traceF < 0.01) return drawHeadGrid(ctx, W, H, S);
  const fn = { scrub: drawScrub, loom: drawLoom, orbit: drawOrbit, stack: drawStack }[S.variant];
  return fn(ctx, W, H, S);
}

export function towerTooltip(S, h) {
  if (h.kind === "ghost") return `<b>${depthName(h.k)}</b> · click to open`;
  if (h.head !== undefined) S = { ...S, head: h.head };
  const where = h.mid ? `block ${h.k - 1} · after attention` : depthName(h.k);
  let s = `<b>${where}</b> · ${tokenName(S.cfg, h.t)}`;
  if (h.head !== undefined) s += ` · head ${h.head + 1}`;
  if (h.k >= 1) {
    const D = S.cfg.dModel, x = h.mid ? S.result.mids[h.k - 2] : S.result.states[h.k - 1];
    let n = 0;
    for (let i = 0; i < D; i++) n += x[h.t * D + i] ** 2;
    s += ` · ‖x‖ ${Math.sqrt(n).toFixed(1)}`;
  }
  if (S.trace && h.k >= 0 && h.k <= S.traceK) {
    const row = traceInfluence(S)[Math.max(1, h.k)];
    s += ` · feeds the traced token ${(row[h.t] * 100).toFixed(1)}%`;
  } else if (h.k === S.block + 1) s += ` · ${tokenName(S.cfg, S.query)} takes ${(weight(S, S.block, h.t) * 100).toFixed(1)}%`;
  return s;
}

// ---------- scrub: the orthographic accordion ----------
function drawScrub(ctx, W, H, S, { compact = false } = {}) {
  const { grid, layers } = S.cfg;
  const K = layers + 2;
  const src = S.blockF + 1; // fractional while animating
  const b = S.block, fade = clamp(1 - Math.abs(S.blockF - b) * 2.5, 0, 1);
  // With sub-steps on, the open gap also holds the block's state between attention and MLP.
  const steps = !!S.subSteps && !!S.result.mids;
  const yaw = -0.62, pitch = 0.55;
  const cyw = Math.cos(yaw), syw = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  const raw = (x, y, z) => {
    const X = x * cyw - z * syw, Z = x * syw + z * cyw;
    return [X, -(y * cp + Z * sp)];
  };
  const G = steps ? 1.6 : 1.05;
  // Tracing unfolds every plate from the traced depth down to the pixels.
  const tf = compact ? 0 : S.traceF, trK = S.traceK;
  const infl = tf > 0 ? traceInfluence(S) : null;
  const ys = [0];
  for (let i = 0; i < K - 1; i++)
    ys.push(ys[i] + lerp(lerp(0.085, G, clamp(1 - Math.abs(i - src), 0, 1)), i < trK ? 0.5 : 0.085, tf));
  const open = (k) => lerp(clamp(1 - Math.min(Math.abs(k - src), Math.abs(k - src - 1)), 0, 1), k <= trK ? 1 : 0, tf);
  // How strongly token t at depth k feeds the traced token, 0..1 within that depth.
  const inflMax = infl ? infl.map((row) => (row ? Math.max(...row) : 1)) : null;
  const lit = (k, t) => {
    if (!infl || k > trK) return 1;
    // Pixels share the embedding's influence: patch p is embedded as token p + 1.
    const v = infl[Math.max(1, k)][t] / inflMax[Math.max(1, k)];
    return lerp(1, 0.18 + 0.82 * Math.pow(v, 0.6), tf);
  };
  const yMid = ys[b + 1] + (ys[b + 2] - ys[b + 1]) * 0.56;
  const showMid = steps && fade > 0 && tf < 0.01;

  // Fit the accordion to the stage (just the open pair when compact).
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  const extent = compact ? [ys[Math.floor(src)], ys[Math.min(K - 1, Math.ceil(src) + 1)]] : [ys[0], ys[K - 1]];
  for (const y of extent)
    for (const [x, z] of [[-0.85, 0], [0.5, 0.5], [0.5, -0.5], [-0.5, 0.5], [-0.5, -0.5]]) {
      const [a, bb] = raw(x, y, z);
      x0 = Math.min(x0, a); x1 = Math.max(x1, a); y0 = Math.min(y0, bb); y1 = Math.max(y1, bb);
    }
  const scale = compact
    ? Math.min((W - 24) / (x1 - x0), (H - 24) / (y1 - y0))
    : Math.min((W - Math.min(260, W * 0.38)) / (x1 - x0), (H - 150) / (y1 - y0));
  const ox = W / 2 - (compact ? 0 : Math.min(40, W * 0.08)) - (scale * (x0 + x1)) / 2;
  const oy = H / 2 + (compact ? 0 : 22) - (scale * (y0 + y1)) / 2;
  const P = (x, y, z) => { const [a, bb] = raw(x, y, z); return [ox + a * scale, oy + bb * scale]; };
  const tileXZ = (p) => [((p % grid) + 0.5) / grid - 0.5, 0.5 - (Math.floor(p / grid) + 0.5) / grid];
  const posAt = (y, t) => (t === 0 ? P(-0.8, y, 0) : P(tileXZ(t - 1)[0], y, tileXZ(t - 1)[1]));
  const pos = (k, t) => posAt(ys[k], t);

  // Hit targets in draw order: { k, quads, orb: [x, y, r], mid }.
  const drawn = [], ghosts = [];
  const hovered = S.hover;
  const h = 0.5 / grid - 0.06 / grid;
  const tileQuads = (y) => Array.from({ length: grid * grid }, (_, p) => {
    const [x, z] = tileXZ(p);
    return [[x - h, z + h], [x + h, z + h], [x + h, z - h], [x - h, z - h]].map(([a, bb]) => P(a, y, bb));
  });
  const ring = (q, color, w) => { quadPath(ctx, q); ctx.strokeStyle = color; ctx.lineWidth = w; ctx.stroke(); };

  // The block's in-between state: a plate slid into the open gap.
  const drawMid = () => {
    ctx.globalAlpha = fade;
    const corners = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]].map(([x, z]) => P(x * 1.05, yMid, z * 1.05));
    quadPath(ctx, corners);
    ctx.fillStyle = "rgba(13,13,20,0.9)"; ctx.fill();
    ctx.setLineDash([2, 3]);
    ctx.strokeStyle = rgba(PAPER, 0.25); ctx.lineWidth = 1; ctx.stroke();
    ctx.setLineDash([]);
    const quads = tileQuads(yMid);
    quads.forEach((q, p) => { quadPath(ctx, q); ctx.fillStyle = rgba(S.midColors[b][p + 1], 0.9); ctx.fill(); });
    if (S.query > 0) ring(quads[S.query - 1], rgba(GOLD, 0.95), 1.6);
    if (hovered?.kind === "token" && hovered.mid && hovered.t > 0) ring(quads[hovered.t - 1], rgba(PAPER, 0.9), 1.2);
    const [x, y] = posAt(yMid, 0);
    orb(ctx, x, y, 6, S.midColors[b][0], { halo: true, ring: S.query === 0 ? rgba(GOLD, 0.9) : null });
    if (!compact) {
      const lx = P(0.5, yMid, 0.5);
      label(ctx, "after attention", lx[0] + 18, lx[1], { active: true });
    }
    ctx.globalAlpha = 1;
    drawn.push({ k: b + 2, quads: fade > 0.5 ? quads : null, orb: [x, y, 12], mid: true });
  };

  for (let k = 0; k < K; k++) {
    const o = open(k), y = ys[k];
    if (compact && o <= 0.5) { ghosts.push(null); continue; }
    const corners = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]].map(([x, z]) => P(x * 1.05, y, z * 1.05));
    // Closed plates above the open pair sit between it and the eye, so keep them sheer.
    const above = k > src + 1 + 0.5;
    quadPath(ctx, corners);
    ctx.fillStyle = above ? "rgba(13,13,20,0.18)" : "rgba(13,13,20,0.9)";
    ctx.fill();
    const ghostHover = hovered?.kind === "ghost" && hovered.k === k;
    ctx.strokeStyle = rgba(ghostHover ? GOLD : PAPER, ghostHover ? 0.8 : 0.07 + 0.25 * o);
    ctx.lineWidth = 1;
    ctx.stroke();
    ghosts.push(corners);

    const quads = tileQuads(y);
    quads.forEach((q, p) => {
      if (k === 0) {
        // Pixels: shear the patch into the plate with an affine transform.
        const [a, bq, , d] = q;
        ctx.save();
        ctx.transform((bq[0] - a[0]) / 4, (bq[1] - a[1]) / 4, (d[0] - a[0]) / 4, (d[1] - a[1]) / 4, a[0], a[1]);
        patchPixels(ctx, S, p, 0, 0, 4, (0.15 + 0.85 * o) * lit(0, p + 1));
        ctx.restore();
      } else {
        quadPath(ctx, q);
        ctx.fillStyle = rgba(S.colors[k - 1][p + 1], ((above ? 0.03 : 0.07) + 0.88 * o) * lit(k, p + 1));
        ctx.fill();
      }
    });
    if (o > 0.5 && S.query > 0 && (tf < 0.5 || k === trK)) ring(quads[S.query - 1], rgba(GOLD, 0.95), 1.6);
    if (o > 0.5 && hovered?.kind === "token" && !hovered.mid && hovered.k === k && hovered.t > 0) ring(quads[hovered.t - 1], rgba(PAPER, 0.9), 1.2);
    let orbHit = null;
    if (k >= 1) {
      const [x, yy] = pos(k, 0);
      const r = 3 + 4 * o;
      ctx.globalAlpha = lit(k, 0);
      orb(ctx, x, yy, r, S.colors[k - 1][0], { halo: o > 0.5, ring: o > 0.5 && S.query === 0 ? rgba(GOLD, 0.9) : null });
      ctx.globalAlpha = 1;
      if (o > 0.5) orbHit = [x, yy, r + 6];
    }
    if (!compact) {
      const lx = P(0.5, y, 0.5);
      const bigK = tf > 0.5 ? trK : Math.round(src) + 1;
      label(ctx, depthName(k), lx[0] + 18, lx[1], { active: o > 0.5, big: W >= 500 && bigK === k });
    }
    drawn.push({ k, quads: o > 0.5 ? quads : null, orb: orbHit, mid: false });
    if (showMid && k === b + 1) drawMid();
  }

  // Attention, the MLP, and the residual path for the active block, fading in after a scrub.
  if (fade * (1 - tf) > 0.01) {
    ctx.globalAlpha = fade * (1 - tf);
    const Qy = showMid ? yMid : ys[b + 2];
    const reach = showMid ? (yMid - ys[b + 1]) * 0.43 : 0.45;
    const Q = posAt(Qy, S.query), Qc = [Q[0], Q[1] + scale * reach];
    const Kq = pos(b + 1, S.query), Qout = pos(b + 2, S.query);
    ctx.setLineDash([3, 4]);
    ctx.strokeStyle = rgba(GOLD, 0.55); ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.moveTo(Kq[0], Kq[1]); ctx.lineTo(Qout[0], Qout[1]); ctx.stroke();
    ctx.setLineDash([]);
    ctx.lineCap = "round";

    if (showMid) {
      // The MLP works on each token alone: one vertical stroke per token, weighted by how far
      // the MLP moves it.
      const D = S.cfg.dModel, T = S.cfg.nTokens, before = S.result.mids[b], after = S.result.states[b + 1];
      const moved = Array.from({ length: T }, (_, t) => {
        let n = 0;
        for (let i = 0; i < D; i++) n += (after[t * D + i] - before[t * D + i]) ** 2;
        return Math.sqrt(n);
      });
      const most = Math.max(...moved);
      for (let t = 0; t < T; t++) {
        const r = moved[t] / most, a = posAt(yMid, t), c = pos(b + 2, t);
        const isQ = t === S.query;
        ctx.strokeStyle = isQ ? rgba(GOLD, 0.95) : rgba(PAPER, 0.06 + 0.5 * r * r);
        ctx.lineWidth = isQ ? 1.8 : 0.5 + 2.5 * r;
        ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(c[0], c[1]); ctx.stroke();
      }
    }

    for (const { j, a, h: head } of sources(S, b, ACTIVE_K).reverse()) {
      const Kp = pos(b + 1, j), Kc = [Kp[0], Kp[1] - scale * reach];
      thread(ctx, () => { ctx.moveTo(Kp[0], Kp[1]); ctx.bezierCurveTo(Kc[0], Kc[1], Qc[0], Qc[1], Q[0], Q[1]); }, a, HEAD_COLORS[head], true);
    }

    if (!compact) {
      const narrow = W < 500;
      const caption = (y, lines) => {
        const m = P(0.5, y, 0.5);
        ctx.font = MONO; ctx.textAlign = "left";
        lines.forEach(([text, color], i) => { ctx.fillStyle = color; ctx.fillText(text, m[0] + 18, m[1] - 7 + i * 15); });
      };
      const attnCaption = [
        [narrow ? "↑ attention" : `↑ attention into ${tokenName(S.cfg, S.query)}`, rgba(GOLD, 0.8)],
        [narrow ? "┆ residual" : "┆ residual path", rgba(MUTED, 0.9)],
      ];
      if (showMid) {
        caption((ys[b + 1] + yMid) / 2, attnCaption);
        caption((yMid + ys[b + 2]) / 2 + 0.04, [[narrow ? "↑ MLP" : "↑ MLP, each token alone", rgba(PAPER, 0.75)]]);
      } else caption((ys[b + 1] + ys[b + 2]) / 2, attnCaption);
    }
    ctx.globalAlpha = 1;
  }

  // The trace: every token's influence splits at each block into its own residual path
  // (half, drawn as a dashed spine) and attention to other tokens (half, spread by the mean
  // attention). Attention flows are scaled among themselves so the residual can't drown them.
  if (infl) {
    ctx.globalAlpha = tf;
    ctx.lineCap = "round";
    const { nTokens: T, heads } = S.cfg;
    for (let d = 2; d <= trK; d++) {
      const flows = [], A = S.result.attn[d - 2];
      const reach = scale * (ys[d] - ys[d - 1]) * 0.45;
      const upMax = Math.max(...infl[d]);
      for (let i = 0; i < T; i++) {
        const u = infl[d][i];
        if (u < 1e-4) continue;
        // Residual spine for the tokens that matter most at this depth.
        const r = u / upMax;
        if (r > 0.15) {
          const a = pos(d - 1, i), c = pos(d, i);
          ctx.setLineDash([3, 4]);
          ctx.strokeStyle = rgba(GOLD, 0.15 + 0.55 * r); ctx.lineWidth = 0.8 + 2 * r;
          ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(c[0], c[1]); ctx.stroke();
          ctx.setLineDash([]);
        }
        for (let j = 0; j < T; j++) {
          let a = 0;
          for (let h = 0; h < heads; h++) a += A[h][i * T + j];
          if (j !== i) flows.push([i, j, u * 0.5 * (a / heads)]);
        }
      }
      flows.sort((x, y) => y[2] - x[2]);
      const top = flows.slice(0, 24), most = top[0]?.[2] || 1;
      for (const [i, j, f] of top.reverse()) {
        const r = f / most, A0 = pos(d - 1, j), B = pos(d, i);
        ctx.strokeStyle = rgba(GOLD, 0.15 + 0.75 * r);
        ctx.lineWidth = 0.6 + 4 * r;
        ctx.beginPath(); ctx.moveTo(A0[0], A0[1]);
        ctx.bezierCurveTo(A0[0], A0[1] - reach, B[0], B[1] + reach, B[0], B[1]);
        ctx.stroke();
      }
    }
    // Embedding → pixels is one patch to one token: straight shafts.
    const e = infl[1], most = Math.max(...e.subarray(1));
    for (let t = 1; t < T; t++) {
      const r = e[t] / most;
      if (r < 0.2) continue;
      const A = pos(0, t), B = pos(1, t);
      ctx.strokeStyle = rgba(GOLD, 0.1 + 0.7 * r); ctx.lineWidth = 0.6 + 3.5 * r;
      ctx.beginPath(); ctx.moveTo(A[0], A[1]); ctx.lineTo(B[0], B[1]); ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }

  // Hit-testing, nearest first: orbs, then open tiles, then any closed plate.
  if (!S.pointer) return null;
  const [mx, my] = S.pointer;
  for (let i = drawn.length - 1; i >= 0; i--) {
    const d = drawn[i];
    if (d.orb && Math.hypot(mx - d.orb[0], my - d.orb[1]) < d.orb[2]) return { kind: "token", k: d.k, t: 0, mid: d.mid };
  }
  for (let i = drawn.length - 1; i >= 0; i--) {
    const d = drawn[i];
    if (!d.quads) continue;
    const p = d.quads.findIndex((q) => inQuad(mx, my, q));
    if (p >= 0) return { kind: "token", k: d.k, t: p + 1, mid: d.mid };
  }
  for (let k = K - 1; k >= 0; k--) if (ghosts[k] && open(k) <= 0.5 && inQuad(mx, my, ghosts[k])) return { kind: "ghost", k };
  return null;
}

// Every head at once: a compact scrub per head, all on the same block and query.
function drawHeadGrid(ctx, W, H, S) {
  const { heads } = S.cfg;
  const cols = heads <= 2 ? heads : 2, rows = Math.ceil(heads / cols);
  const top = 96, bottom = 60, gap = 16, titleH = 30;
  const cw = (W - gap * (cols + 1)) / cols, ch = (H - top - bottom - gap * (rows - 1)) / rows;
  let hover = null;
  for (let head = 0; head < heads; head++) {
    const cx0 = gap + (head % cols) * (cw + gap), cy0 = top + Math.floor(head / cols) * (ch + gap);
    ctx.save();
    ctx.translate(cx0, cy0);
    ctx.beginPath(); ctx.rect(0, 0, cw, ch); ctx.clip();
    ctx.strokeStyle = rgba(PAPER, 0.07); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.roundRect(0.5, 0.5, cw - 1, ch - 1, 8); ctx.stroke();

    const local = { ...S, head, hover: S.hover?.head === head ? S.hover : null, pointer: null };
    const top1 = sources(local, S.block, 1)[0];
    ctx.textBaseline = "middle"; ctx.textAlign = "left";
    ctx.font = serif(17, 400); ctx.fillStyle = rgba(HEAD_COLORS[head]);
    ctx.fillText(`head ${head + 1}`, 12, 18);
    if (top1) {
      ctx.font = MONO; ctx.fillStyle = rgba(MUTED, 0.95);
      ctx.fillText(`strongest: ${tokenName(S.cfg, top1.j)} · ${(top1.a * 100).toFixed(0)}%`, 84, 19);
    }
    ctx.translate(0, titleH);
    if (S.pointer) {
      const px = S.pointer[0] - cx0, py = S.pointer[1] - cy0 - titleH;
      if (px >= 0 && px < cw && py >= 0 && py < ch - titleH) local.pointer = [px, py];
    }
    const hv = drawScrub(ctx, cw, ch - titleH, local, { compact: true });
    ctx.restore();
    if (hv) hover = { ...hv, head };
  }
  return hover;
}

// ---------- loom: tokens as warp, attention as weft ----------
function drawLoom(ctx, W, H, S) {
  const { grid, layers, nTokens: T } = S.cfg;
  const K = layers + 2;
  const left = 96, right = 28, top = 118, bottom = 104;
  const units = 3 + (T - 2) + (grid - 1);
  const step = (W - left - right) / units;
  const xOf = (t) => (t === 0 ? left : left + step * (3 + (t - 1) + Math.floor((t - 1) / grid)));
  const rowH = (H - top - bottom) / (K - 1);
  const yOf = (k) => H - bottom - k * rowH;
  const b = S.block;
  const beadW = Math.max(2, step * 0.7), beadH = Math.min(10, rowH * 0.18);

  // The active block's band.
  ctx.fillStyle = "rgba(235,230,218,0.035)";
  ctx.fillRect(left - 20, yOf(b + 2) - 16, W - left - right + 36, rowH + 32);

  // Warp: every token is a vertical thread, recolored at each depth.
  ctx.lineWidth = 1.1;
  for (let t = 0; t < T; t++)
    for (let k = 1; k < K - 1; k++) {
      ctx.strokeStyle = rgba(S.colors[k][t], 0.3);
      ctx.beginPath(); ctx.moveTo(xOf(t), yOf(k)); ctx.lineTo(xOf(t), yOf(k + 1)); ctx.stroke();
    }
  if (S.query >= 0) {
    ctx.strokeStyle = rgba(GOLD, 0.7); ctx.lineWidth = 1.6;
    ctx.beginPath(); ctx.moveTo(xOf(S.query), yOf(S.query === 0 ? 1 : 0)); ctx.lineTo(xOf(S.query), yOf(K - 1)); ctx.stroke();
  }

  // Weft: attention woven from each row into the query on the row above.
  ctx.lineCap = "round";
  const weave = (bb, active) => {
    const qx = xOf(S.query), qy = yOf(bb + 2) + beadH / 2 + 1;
    for (const { j, a, h } of sources(S, bb, active ? ACTIVE_K : TRACE_K).reverse()) {
      const kx = xOf(j), ky = yOf(bb + 1) - beadH / 2 - 1;
      thread(ctx, () => { ctx.moveTo(kx, ky); ctx.bezierCurveTo(kx, ky - rowH * 0.55, qx, qy + rowH * 0.55, qx, qy); }, a, HEAD_COLORS[h], active);
    }
  };
  for (let bb = 0; bb < layers; bb++) if (bb !== b) weave(bb, false);
  weave(b, true);

  // Beads.
  for (let k = 1; k < K; k++) {
    const active = k === b + 1 || k === b + 2;
    for (let t = 0; t < T; t++) {
      const x = xOf(t), y = yOf(k), c = S.colors[k - 1][t];
      if (t === 0) { orb(ctx, x, y, 5, c, { halo: active, ring: S.query === 0 && k === b + 2 ? rgba(GOLD, 0.9) : null }); continue; }
      ctx.fillStyle = rgba(c, active ? 1 : 0.78);
      ctx.beginPath(); ctx.roundRect(x - beadW / 2, y - beadH / 2, beadW, beadH, 2); ctx.fill();
    }
    if (S.query > 0 && active) {
      ctx.strokeStyle = rgba(GOLD, 0.95); ctx.lineWidth = 1.4;
      ctx.strokeRect(xOf(S.query) - beadW / 2 - 2, yOf(k) - beadH / 2 - 2, beadW + 4, beadH + 4);
    }
  }
  // Pixels: the patches themselves along the bottom.
  const ps = Math.min(step * 0.92, rowH * 0.62);
  for (let p = 0; p < grid * grid; p++) patchPixels(ctx, S, p, xOf(p + 1) - ps / 2, yOf(0) - ps / 2, ps);
  ctx.font = MONO; ctx.textAlign = "center"; ctx.textBaseline = "top"; ctx.fillStyle = rgba(MUTED, 0.8);
  for (let r = 0; r < grid; r++) ctx.fillText(`row ${r}`, (xOf(r * grid + 1) + xOf(r * grid + grid)) / 2, yOf(0) + ps / 2 + 8);
  ctx.fillText("CLS", xOf(0), yOf(0) + ps / 2 + 8);
  for (let k = 0; k < K; k++) label(ctx, depthName(k), left - 26, yOf(k), { align: "right", active: k === b + 1 || k === b + 2 });

  const hov = S.hover;
  if (hov?.kind === "token") {
    ctx.strokeStyle = rgba(PAPER, 0.9); ctx.lineWidth = 1;
    ctx.strokeRect(xOf(hov.t) - beadW / 2 - 2, yOf(hov.k) - Math.max(beadH, hov.k === 0 ? ps : 0) / 2 - 2, beadW + 4, Math.max(beadH, hov.k === 0 ? ps : 0) + 4);
  }

  if (!S.pointer) return null;
  const [mx, my] = S.pointer;
  const k = Math.round((H - bottom - my) / rowH);
  if (k < 0 || k >= K || Math.abs(yOf(k) - my) > rowH * 0.3) return null;
  let best = -1, bd = step * 0.7;
  for (let t = k === 0 ? 1 : 0; t < T; t++) { const d = Math.abs(xOf(t) - mx); if (d < bd) { bd = d; best = t; } }
  return best >= 0 ? { kind: "token", k, t: best } : null;
}

// ---------- orbit: the image unwrapped into rings ----------
const slotCache = new Map();
function slots(grid) {
  // Patches ordered clockwise by their direction from the image center, so each ring is the
  // image unwrapped around its middle.
  if (slotCache.has(grid)) return slotCache.get(grid);
  const c = (grid - 1) / 2;
  const order = [...Array(grid * grid).keys()].sort((p, q) => {
    const ang = (i) => { const a = Math.atan2(i % grid - c, -(Math.floor(i / grid) - c)); return a < 0 ? a + 2 * Math.PI : a; };
    const rad = (i) => Math.hypot(i % grid - c, Math.floor(i / grid) - c);
    return ang(p) - ang(q) || rad(p) - rad(q);
  });
  const slotOf = new Array(grid * grid);
  order.forEach((p, s) => (slotOf[p] = s));
  const out = { order, slotOf };
  slotCache.set(grid, out);
  return out;
}
let orbitImg = null;
function drawOrbit(ctx, W, H, S) {
  const { grid, layers, nTokens: T } = S.cfg;
  const K = layers + 2, N = grid * grid;
  const cx = W / 2, cy = H / 2 + 18;
  const R = Math.min(W - 80, H - 120) / 2;
  const r0 = R * 0.2, dr = (R - r0) / (K - 1), thick = dr * 0.58;
  const ringR = (k) => r0 + (k - 0.5) * dr;
  const gapA = 0.2, span = (2 * Math.PI - gapA) / N;
  const { order, slotOf } = slots(grid);
  const angOf = (p) => -Math.PI / 2 + gapA / 2 + (slotOf[p] + 0.5) * span;
  const pos = (k, t) => (t === 0 ? [cx, cy] : [cx + Math.cos(angOf(t - 1)) * ringR(k), cy + Math.sin(angOf(t - 1)) * ringR(k)]);
  const b = S.block;

  // The input image at the heart.
  orbitImg = orbitImg || Object.assign(document.createElement("canvas"), { width: 32, height: 32 });
  const ictx = orbitImg.getContext("2d"), img = ictx.createImageData(32, 32);
  for (let i = 0; i < 1024; i++) { for (let c = 0; c < 3; c++) img.data[i * 4 + c] = S.pixels[i * 3 + c]; img.data[i * 4 + 3] = 255; }
  ictx.putImageData(img, 0, 0);
  ctx.save();
  ctx.beginPath(); ctx.arc(cx, cy, r0 * 0.94, 0, Math.PI * 2); ctx.clip();
  ctx.imageSmoothingEnabled = false;
  ctx.globalAlpha = 0.5;
  ctx.drawImage(orbitImg, cx - r0 * 0.94, cy - r0 * 0.94, r0 * 1.88, r0 * 1.88);
  ctx.restore();

  // Rings.
  for (let k = 1; k < K; k++) {
    const active = k === b + 1 || k === b + 2;
    ctx.lineWidth = thick;
    for (let p = 0; p < N; p++) {
      const a = angOf(p);
      ctx.strokeStyle = rgba(S.colors[k - 1][p + 1], active ? 1 : 0.55);
      ctx.beginPath(); ctx.arc(cx, cy, ringR(k), a - span * 0.42, a + span * 0.42); ctx.stroke();
    }
    ctx.font = "9px 'JetBrains Mono', monospace"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillStyle = active ? rgba(GOLD, 0.95) : rgba(MUTED, 0.8);
    ctx.fillText(k === 1 ? "e" : String(k - 1), cx, cy - ringR(k));
  }

  // The query's radial spoke and its segments.
  if (S.query > 0) {
    const a = angOf(S.query - 1);
    ctx.strokeStyle = rgba(GOLD, 0.45); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(cx + Math.cos(a) * r0, cy + Math.sin(a) * r0); ctx.lineTo(cx + Math.cos(a) * (R + 8), cy + Math.sin(a) * (R + 8)); ctx.stroke();
    ctx.strokeStyle = rgba(GOLD, 0.95); ctx.lineWidth = 1.4;
    for (const k of [b + 1, b + 2]) {
      ctx.beginPath();
      ctx.arc(cx, cy, ringR(k) + thick / 2 + 1.5, a - span / 2, a + span / 2);
      ctx.arc(cx, cy, ringR(k) - thick / 2 - 1.5, a + span / 2, a - span / 2, true);
      ctx.closePath(); ctx.stroke();
    }
  }

  // Threads spiral inward: ring → ring, or ring → the CLS heart.
  ctx.lineCap = "round";
  const weave = (bb, active) => {
    const Q = pos(bb + 2, S.query);
    for (const { j, a, h } of sources(S, bb, active ? ACTIVE_K : TRACE_K).reverse()) {
      const Kp = pos(bb + 1, j);
      let c;
      if (j === 0 && S.query === 0) continue; // CLS → CLS stays in the heart
      if (S.query === 0) { const an = angOf(j - 1) + 0.5; c = [cx + Math.cos(an) * ringR(bb + 1) * 0.55, cy + Math.sin(an) * ringR(bb + 1) * 0.55]; }
      else c = [lerp(cx, (Kp[0] + Q[0]) / 2, 0.55), lerp(cy, (Kp[1] + Q[1]) / 2, 0.55)];
      thread(ctx, () => { ctx.moveTo(Kp[0], Kp[1]); ctx.quadraticCurveTo(c[0], c[1], Q[0], Q[1]); }, a, HEAD_COLORS[h], active);
    }
  };
  for (let bb = 0; bb < layers; bb++) if (bb !== b) weave(bb, false);
  weave(b, true);
  orb(ctx, cx, cy, Math.max(5, r0 * 0.2), S.colors[b + 1][0], { halo: true, ring: S.query === 0 ? rgba(GOLD, 0.9) : null });

  const hov = S.hover;
  if (hov?.kind === "token" && hov.t > 0 && hov.k > 0) {
    const a = angOf(hov.t - 1);
    ctx.strokeStyle = rgba(PAPER, 0.9); ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.arc(cx, cy, ringR(hov.k) + thick / 2 + 1, a - span / 2, a + span / 2);
    ctx.arc(cx, cy, ringR(hov.k) - thick / 2 - 1, a + span / 2, a - span / 2, true);
    ctx.closePath(); ctx.stroke();
  }

  if (!S.pointer) return null;
  const dx = S.pointer[0] - cx, dy = S.pointer[1] - cy, r = Math.hypot(dx, dy);
  if (r < r0 * 0.35) return { kind: "token", k: b + 2, t: 0 };
  const k = Math.round((r - r0) / dr + 0.5);
  if (k < 1 || k >= K || Math.abs(r - ringR(k)) > thick / 2 + 2) return null;
  let a = Math.atan2(dy, dx) + Math.PI / 2 - gapA / 2;
  a = ((a % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
  const s = Math.floor(a / span);
  return s < N ? { kind: "token", k, t: order[s] + 1 } : null;
}

// ---------- stack: the 3D tower, calmer ----------
function drawStack(ctx, W, H, S) {
  const { grid, layers, heads } = S.cfg;
  const K = layers + 2, GAP = 0.5, camD = 7;
  const cyw = Math.cos(S.yaw), syw = Math.sin(S.yaw), cp = Math.cos(S.pitch), sp = Math.sin(S.pitch);
  const plateY = (k) => (k - (K - 1) / 2) * GAP;
  const raw = (x, y, z) => {
    const X = x * cyw - z * syw, Z = x * syw + z * cyw;
    const f = camD / (camD - y * sp + Z * cp);
    return [X * f, -(y * cp + Z * sp) * f];
  };
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const k of [0, K - 1])
    for (const [x, z] of [[-0.85, 0], [0.55, 0.55], [0.55, -0.55], [-0.55, 0.55], [-0.55, -0.55]]) {
      const [a, b] = raw(x, plateY(k), z);
      x0 = Math.min(x0, a); x1 = Math.max(x1, a); y0 = Math.min(y0, b); y1 = Math.max(y1, b);
    }
  const scale = Math.min((W - Math.min(220, W * 0.36)) / (x1 - x0), (H - 180) / (y1 - y0));
  const ox = W / 2 - Math.min(30, W * 0.08) - (scale * (x0 + x1)) / 2, oy = H / 2 + 34 - (scale * (y0 + y1)) / 2;
  const P = (x, y, z) => { const [a, b] = raw(x, y, z); return [ox + a * scale, oy + b * scale]; };
  const tileXZ = (p) => [((p % grid) + 0.5) / grid - 0.5, 0.5 - (Math.floor(p / grid) + 0.5) / grid];
  const pos = (k, t) => (t === 0 ? P(-0.78, plateY(k), 0) : P(tileXZ(t - 1)[0], plateY(k), tileXZ(t - 1)[1]));
  const b = S.block;
  const picks = [], orbs = [];
  const hovered = S.hover?.kind === "token" ? S.hover : null;

  for (let k = 0; k < K; k++) {
    const y = plateY(k), active = k === b + 1 || k === b + 2;
    const corners = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]].map(([x, z]) => P(x * 1.04, y, z * 1.04));
    quadPath(ctx, corners);
    ctx.fillStyle = "rgba(13,13,20,0.72)"; ctx.fill();
    ctx.strokeStyle = rgba(PAPER, active ? 0.3 : 0.1); ctx.lineWidth = 1; ctx.stroke();
    const quads = [];
    const h = 0.5 / grid - 0.07 / grid;
    for (let p = 0; p < grid * grid; p++) {
      const [x, z] = tileXZ(p);
      const q = [[x - h, z + h], [x + h, z + h], [x + h, z - h], [x - h, z - h]].map(([a, bb]) => P(a, y, bb));
      quads.push(q);
      if (k === 0) {
        const [a, bq, , d] = q;
        ctx.save();
        ctx.transform((bq[0] - a[0]) / 4, (bq[1] - a[1]) / 4, (d[0] - a[0]) / 4, (d[1] - a[1]) / 4, a[0], a[1]);
        patchPixels(ctx, S, p, 0, 0, 4);
        ctx.restore();
      } else {
        quadPath(ctx, q);
        ctx.fillStyle = rgba(S.colors[k - 1][p + 1], active ? 0.95 : 0.6);
        ctx.fill();
      }
    }
    picks.push(quads);
    if (S.query > 0) { quadPath(ctx, quads[S.query - 1]); ctx.strokeStyle = rgba(GOLD, active ? 0.95 : 0.5); ctx.lineWidth = 1.4; ctx.stroke(); }
    if (hovered && hovered.k === k && hovered.t > 0) { quadPath(ctx, quads[hovered.t - 1]); ctx.strokeStyle = rgba(PAPER, 0.9); ctx.lineWidth = 1.2; ctx.stroke(); }
    if (k >= 1) {
      const [x, yy] = pos(k, 0);
      orb(ctx, x, yy, active ? 6 : 4, S.colors[k - 1][0], { halo: active, ring: S.query === 0 ? rgba(GOLD, active ? 0.9 : 0.4) : hovered?.k === k && hovered.t === 0 ? rgba(PAPER, 0.9) : null });
      orbs.push([k, x, yy]);
    }
    // Threads arriving at this plate.
    if (k >= 2) {
      const bb = k - 2, act = bb === b;
      const Q = pos(k, S.query), Q2 = pos(k, S.query).map((v, i) => (i === 1 ? v + scale * GAP * 0.5 * cp : v));
      ctx.lineCap = "round";
      for (const { j, a, h } of sources(S, bb, act ? ACTIVE_K : TRACE_K).reverse()) {
        const Kp = pos(k - 1, j), K2 = [Kp[0], Kp[1] - scale * GAP * 0.5 * cp];
        thread(ctx, () => { ctx.moveTo(Kp[0], Kp[1]); ctx.bezierCurveTo(K2[0], K2[1], Q2[0], Q2[1], Q[0], Q[1]); }, a, HEAD_COLORS[h], act, act ? 1 : 0.8);
      }
    }
    const lp = P(0.74 * cyw, y, -0.74 * syw);
    label(ctx, depthName(k), lp[0] + 14, lp[1], { active });
  }

  if (!S.pointer || S.dragging) return null;
  const [mx, my] = S.pointer;
  for (let i = orbs.length - 1; i >= 0; i--) { const [k, x, y] = orbs[i]; if (Math.hypot(mx - x, my - y) < 9) return { kind: "token", k, t: 0 }; }
  for (let k = K - 1; k >= 0; k--) { const p = picks[k].findIndex((q) => inQuad(mx, my, q)); if (p >= 0) return { kind: "token", k, t: p + 1 }; }
  return null;
}

// ---------- the "reading" card ----------
export function readingHtml(S) {
  if (S.trace) {
    // The image patches whose embeddings the traced token draws on most.
    const e = traceInfluence(S)[1];
    const top = [...e.keys()].filter((t) => t > 0).sort((x, y) => e[y] - e[x]).slice(0, 5);
    const rows = top.map((t) =>
      `<div class="rr"><canvas data-t="${t}" width="4" height="4"></canvas><span class="dot" style="background:${rgba(GOLD)}"></span>` +
      `<span class="nm">${tokenName(S.cfg, t)}</span><span class="pct">${(e[t] * 100).toFixed(0)}%</span>` +
      `<span class="bar"><i style="width:${Math.min(100, (e[t] / e[top[0]]) * 100)}%;background:${rgba(GOLD)}"></i></span></div>`).join("");
    return `<div class="rh"><span class="serif">trace</span> · <b>${tokenName(S.cfg, S.query)}</b> at ${depthName(S.traceK)} comes most from</div>${rows}` +
      `<div class="rf">${(e[0] * 100).toFixed(0)}% traces back to the CLS embedding, not the image</div>` +
      `<div class="rf">rollout: heads averaged, residual counted, MLPs ignored · Esc to close</div>`;
  }
  const b = S.block;
  const src = sources(S, b, 5);
  const total = src.reduce((s, e) => s + e.a, 0);
  const rows = src.map(({ j, a, h }) =>
    `<div class="rr"><canvas data-t="${j}" width="4" height="4"></canvas>` +
    `<span class="dot" style="background:${rgba(HEAD_COLORS[h])}"></span>` +
    `<span class="nm">${tokenName(S.cfg, j)}</span><span class="pct">${(a * 100).toFixed(0)}%</span>` +
    `<span class="bar"><i style="width:${Math.min(100, a * 200)}%;background:${rgba(HEAD_COLORS[h])}"></i></span></div>`).join("");
  let mlp = "";
  if (S.subSteps && S.result.mids) {
    // How far the MLP moves the query, and how that ranks among all tokens.
    const D = S.cfg.dModel, T = S.cfg.nTokens, before = S.result.mids[b], after = S.result.states[b + 1];
    const moved = Array.from({ length: T }, (_, t) => {
      let n = 0;
      for (let i = 0; i < D; i++) n += (after[t * D + i] - before[t * D + i]) ** 2;
      return Math.sqrt(n);
    });
    const rank = moved.filter((m) => m > moved[S.query]).length + 1;
    mlp = `<div class="rf">then the MLP moves it by ${moved[S.query].toFixed(1)} · ${rank === 1 ? "the most" : `#${rank}`} of ${T} tokens</div>`;
  }
  return `<div class="rh"><span class="serif">block ${b + 1}</span> · <b>${tokenName(S.cfg, S.query)}</b> gathers from</div>${rows}` +
    `<div class="rf">top ${src.length} = ${(total * 100).toFixed(0)}% of ${S.head >= 0 ? `head ${S.head + 1}` : "the mean over heads"}</div>` + mlp;
}
