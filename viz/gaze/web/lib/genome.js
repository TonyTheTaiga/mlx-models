// A constrained drawing vocabulary for evolving images against the classifier (after Tom White's
// Perception Engines): a ground color, a few flat blobs and a handful of curved strokes, in a
// fixed palette of inks. Coordinates are in the unit square.

export const INKS = ["#efe9dc", "#151412", "#d9431e", "#1f4fbf", "#f2c230", "#2f7a4f", "#7fb3d5", "#8b5a2b"];

const rnd = Math.random;
const gauss = () => Math.sqrt(-2 * Math.log(1 - rnd())) * Math.cos(2 * Math.PI * rnd());
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const pick = (n) => Math.floor(rnd() * n);

export function randomGenome(style = "engine") {
  const soft = style === "field";
  return {
    style,
    ground: pick(INKS.length),
    blobs: Array.from({ length: soft ? 7 : 3 }, () => ({ x: rnd(), y: rnd(), r: 0.08 + rnd() * (soft ? 0.35 : 0.22), ink: pick(INKS.length) })),
    strokes: Array.from({ length: soft ? 3 : 8 }, () => {
      const x0 = rnd(), y0 = rnd();
      return { x0, y0, cx: x0 + gauss() * 0.3, cy: y0 + gauss() * 0.3, x1: x0 + gauss() * 0.4, y1: y0 + gauss() * 0.4, w: 0.02 + rnd() * 0.08, ink: pick(INKS.length) };
    }),
    fixed: [], // strokes a person drew; evolution works around them
  };
}

export function clone(g) {
  return { ...g, blobs: g.blobs.map((b) => ({ ...b })), strokes: g.strokes.map((s) => ({ ...s })), fixed: g.fixed };
}

// Nudge one to three genes: positions by a small Gaussian step, occasionally an ink or a size.
export function mutate(g, rate = 1) {
  const m = clone(g);
  const n = 1 + pick(3);
  for (let t = 0; t < n; t++) {
    const r = rnd();
    if (r < 0.06) m.ground = pick(INKS.length);
    else if (r < 0.4 && m.blobs.length) {
      const b = m.blobs[pick(m.blobs.length)];
      if (rnd() < 0.15) b.ink = pick(INKS.length);
      b.x = clamp(b.x + gauss() * 0.06 * rate, -0.1, 1.1);
      b.y = clamp(b.y + gauss() * 0.06 * rate, -0.1, 1.1);
      b.r = clamp(b.r * Math.exp(gauss() * 0.15 * rate), 0.03, 0.6);
    } else if (m.strokes.length) {
      const s = m.strokes[pick(m.strokes.length)];
      if (rnd() < 0.15) s.ink = pick(INKS.length);
      for (const k of ["x0", "y0", "cx", "cy", "x1", "y1"]) if (rnd() < 0.5) s[k] = clamp(s[k] + gauss() * 0.07 * rate, -0.2, 1.2);
      if (rnd() < 0.3) s.w = clamp(s.w * Math.exp(gauss() * 0.2 * rate), 0.008, 0.16);
    }
  }
  return m;
}

// Draw a genome into a size×size context.
export function render(ctx, g, size) {
  ctx.save();
  ctx.fillStyle = INKS[g.ground];
  ctx.fillRect(0, 0, size, size);
  const soft = g.style === "field";
  for (const b of g.blobs) {
    ctx.beginPath();
    if (soft) {
      const gr = ctx.createRadialGradient(b.x * size, b.y * size, 0, b.x * size, b.y * size, b.r * size);
      gr.addColorStop(0, INKS[b.ink]);
      gr.addColorStop(1, INKS[b.ink] + "00");
      ctx.fillStyle = gr;
    } else ctx.fillStyle = INKS[b.ink];
    ctx.arc(b.x * size, b.y * size, b.r * size, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  for (const s of g.strokes) {
    ctx.strokeStyle = INKS[s.ink];
    ctx.lineWidth = s.w * size;
    ctx.beginPath();
    ctx.moveTo(s.x0 * size, s.y0 * size);
    ctx.quadraticCurveTo(s.cx * size, s.cy * size, s.x1 * size, s.y1 * size);
    ctx.stroke();
  }
  for (const f of g.fixed) {
    ctx.strokeStyle = INKS[f.ink];
    ctx.lineWidth = f.w * size;
    ctx.beginPath();
    f.points.forEach(([x, y], i) => ctx[i ? "lineTo" : "moveTo"](x * size, y * size));
    ctx.stroke();
  }
  ctx.restore();
}

// What the model sees: render at 128 px, then average down to 32×32 RGB.
const big = Object.assign(document.createElement("canvas"), { width: 128, height: 128 });
const small = Object.assign(document.createElement("canvas"), { width: 32, height: 32 });
export function toPixels(g) {
  render(big.getContext("2d"), g, 128);
  const sctx = small.getContext("2d");
  sctx.imageSmoothingQuality = "high";
  sctx.clearRect(0, 0, 32, 32);
  sctx.drawImage(big, 0, 0, 32, 32);
  const d = sctx.getImageData(0, 0, 32, 32).data;
  const px = new Uint8ClampedArray(3072);
  for (let i = 0; i < 1024; i++) for (let c = 0; c < 3; c++) px[i * 3 + c] = d[i * 4 + c];
  return px;
}

export function drawPixels(ctx, px, size) {
  const img = ctx.createImageData(32, 32);
  for (let i = 0; i < 1024; i++) {
    for (let c = 0; c < 3; c++) img.data[i * 4 + c] = px[i * 3 + c];
    img.data[i * 4 + 3] = 255;
  }
  small.getContext("2d").putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(small, 0, 0, size, size);
}

export const strokeCount = (g) => g.strokes.length + g.fixed.length;
// Evolved marks (blobs + strokes); a person's strokes don't count, evolution can't remove them.
export const markCount = (g) => g.strokes.length + g.blobs.length;

// Every way to remove exactly one evolved mark.
export function removals(g) {
  const out = [];
  g.strokes.forEach((_, i) => { const m = clone(g); m.strokes.splice(i, 1); out.push(m); });
  g.blobs.forEach((_, i) => { const m = clone(g); m.blobs.splice(i, 1); out.push(m); });
  return out;
}
