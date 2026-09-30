// Shared math, color, and canvas helpers.

export const $ = (id) => document.getElementById(id);
export const dpr = () => Math.min(window.devicePixelRatio || 1, 2);
export const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

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

export const CLASS_COLORS = Array.from({ length: 10 }, (_, i) => oklch(0.76, 0.13, 20 + i * 36));
export const GOLD = [233, 185, 101];
export const PAPER = [235, 230, 218];
export const MUTED = [139, 134, 118];
export const MONO = "10px 'JetBrains Mono', monospace";

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
