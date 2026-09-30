// Small DOM, text and image helpers shared by the rooms.

export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) if (c != null && c !== false) el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return el;
}

const ONES = ["none", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
  "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"];
const TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty"];
export function word(n) {
  if (n < 20) return ONES[n];
  return TENS[Math.floor(n / 10)] + (n % 10 ? "-" + ONES[n % 10] : "");
}
export const cap = (s) => s[0].toUpperCase() + s.slice(1);
export const pct = (p, d = 0) => `${(p * 100).toFixed(d)}%`;
export const article = (w) => (/^[aeiou]/.test(w) ? "an" : "a");

export async function loadImages(url) {
  return new Uint8Array(await (await fetch(url)).arrayBuffer());
}
export const imageAt = (all, i) => all.subarray(i * 3072, (i + 1) * 3072);

export function meanColor(px) {
  const m = [0, 0, 0];
  for (let i = 0; i < 1024; i++) for (let c = 0; c < 3; c++) m[c] += px[i * 3 + c] / 1024;
  return m.map(Math.round);
}

// Draw a 32×32 image into a canvas context at size, pixel-sharp. Patches with alpha[p] < 1 are
// blended toward the image's mean color (1 = the real pixels, 0 = flat mean).
export function drawImage(ctx, px, size, alpha = null, grid = 8) {
  const win = 32 / grid, s = size / 32, m = meanColor(px);
  for (let y = 0; y < 32; y++)
    for (let x = 0; x < 32; x++) {
      const o = (y * 32 + x) * 3;
      const a = alpha ? alpha[Math.floor(y / win) * grid + Math.floor(x / win)] : 1;
      const c = [0, 1, 2].map((k) => Math.round(m[k] + (px[o + k] - m[k]) * a));
      ctx.fillStyle = `rgb(${c})`;
      ctx.fillRect(x * s, y * s, s + 0.5, s + 0.5);
    }
}

// Heat ramp: transparent → deep red → vermilion → saffron → pale. Always shown with a legend.
const STOPS = [[92, 18, 16], [179, 38, 30], [217, 67, 30], [242, 176, 48], [255, 236, 180]];
export function heatColor(t) {
  t = Math.max(0, Math.min(1, t));
  const x = t * (STOPS.length - 1), i = Math.min(STOPS.length - 2, Math.floor(x)), f = x - i;
  return STOPS[i].map((a, k) => Math.round(a + (STOPS[i + 1][k] - a) * f));
}
export const HEAT_GRADIENT = `linear-gradient(90deg, ${STOPS.map((c) => `rgb(${c})`).join(", ")})`;

// A smooth heat layer (values 0..1 on a res×res grid) over a grayscale copy of the image.
const tmp = document.createElement("canvas");
export function drawHeat(ctx, px, values, res, size) {
  const s = size / 32;
  for (let y = 0; y < 32; y++)
    for (let x = 0; x < 32; x++) {
      const o = (y * 32 + x) * 3, g = Math.round((0.3 * px[o] + 0.59 * px[o + 1] + 0.11 * px[o + 2]) * 0.55 + 60);
      ctx.fillStyle = `rgb(${g},${g},${g - 4})`;
      ctx.fillRect(x * s, y * s, s + 0.5, s + 0.5);
    }
  tmp.width = tmp.height = res;
  const t = tmp.getContext("2d"), img = t.createImageData(res, res);
  values.forEach((v, i) => {
    img.data.set(heatColor(v), i * 4);
    img.data[i * 4 + 3] = v < 0.04 ? 0 : Math.round(255 * (0.3 + 0.68 * Math.pow(v, 0.7)));
  });
  t.putImageData(img, 0, 0);
  ctx.save();
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(tmp, 0, 0, size, size);
  ctx.restore();
}

export function hiDPI(canvas, size) {
  const k = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = canvas.height = Math.round(size * k);
  canvas.style.width = canvas.style.height = `${size}px`;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(k, 0, 0, k, 0, 0);
  return ctx;
}
