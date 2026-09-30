// Shared math, color, and canvas helpers.

export const $ = (id) => document.getElementById(id);
export const dpr = () => Math.min(window.devicePixelRatio || 1, 2);
export const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
export const lerp = (a, b, t) => a + (b - a) * t;
export const ease = (t) => t * t * (3 - 2 * t);
export const frac = (x) => x - Math.floor(x);

export function oklabToRgb(L, a, b) {
  const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3;
  const m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3;
  const s = (L - 0.0894841775 * a - 1.291485548 * b) ** 3;
  const lin = [
    4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s,
  ];
  return lin.map((c) => {
    c = clamp(c, 0, 1);
    return Math.round(255 * (c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055));
  });
}
export const oklch = (L, C, h) => oklabToRgb(L, C * Math.cos((h * Math.PI) / 180), C * Math.sin((h * Math.PI) / 180));
export const rgba = ([r, g, b], a = 1) => `rgba(${r},${g},${b},${a})`;

export const HEAD_COLORS = [oklch(0.74, 0.16, 35), oklch(0.8, 0.12, 205), oklch(0.72, 0.16, 305), oklch(0.84, 0.15, 125)];
export const CLASS_COLORS = Array.from({ length: 10 }, (_, i) => oklch(0.76, 0.13, 20 + i * 36));
export const GOLD = [233, 185, 101];
export const PAPER = [235, 230, 218];
export const MUTED = [139, 134, 118];
export const layerColor = (l, n) => oklch(0.78, 0.13, lerp(200, 70, l / Math.max(1, n - 1)));

export const MONO = "10px 'JetBrains Mono', monospace";
export const serif = (px, weight = 300) => `italic ${weight} ${px}px Fraunces, Georgia, serif`;

// Size a canvas's backing store to its CSS box at device pixel ratio.
export function fit(canvas) {
  const r = canvas.getBoundingClientRect();
  const k = dpr();
  const w = Math.round(r.width * k), h = Math.round(r.height * k);
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(k, 0, 0, k, 0, 0);
  return [ctx, r.width, r.height];
}

export function quadPath(ctx, pts) {
  ctx.beginPath();
  ctx.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
  ctx.closePath();
}

export function inQuad(px, py, q) {
  let sign = 0;
  for (let i = 0; i < 4; i++) {
    const [ax, ay] = q[i], [bx, by] = q[(i + 1) % 4];
    const c = (bx - ax) * (py - ay) - (by - ay) * (px - ax);
    if (c !== 0) {
      if (sign === 0) sign = Math.sign(c);
      else if (Math.sign(c) !== sign) return false;
    }
  }
  return true;
}
