// ten dreams — each dream develops out of grey noise, holds, and fades to black.
//
// A dream is the recorded sequence of an optimization (see ../dream.py): frames are spaced densely
// where the image first appears, so playing them at an even pace already has a darkroom rhythm;
// consecutive frames are crossfaded. The page shuffles all dreams so a word never repeats twice
// in a row, and preloads the next dream while the current one plays.

const $ = (id) => document.getElementById(id);
const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
const T = { in: 1600, develop: 12000, hold: 7500, out: 2200 }; // ms
const TOTAL = T.in + T.develop + T.hold + T.out;

const manifest = await fetch("assets/manifest.json").then((r) => r.json());
// The about text names the model that dreamed these, from the manifest.
if (manifest.model) {
  const NUM = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"];
  const m = manifest.model, millions = (manifest.params / 1e6).toFixed(1);
  const line = $("modelline").firstChild;
  line.textContent = line.textContent.replace("The model is a vision transformer trained on",
    `The model is a vision transformer — ${NUM[m.n_layers] ?? m.n_layers} blocks, ${NUM[m.n_heads] ?? m.n_heads} heads, about ${millions} million parameters — trained on`);
}
const canvas = $("dream"), ctx = canvas.getContext("2d");
const word = $("word");
let queue = [], current = null, next = null, start = 0, paused = false, pausedAt = 0;

function shuffled() {
  // Shuffle, then nudge so the same class never plays twice in a row.
  const a = [...manifest.dreams];
  for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
  for (let i = 1; i < a.length; i++)
    if (a[i].class === a[i - 1].class) {
      const k = a.findIndex((d, j) => j > i && d.class !== a[i - 1].class);
      if (k > 0) [a[i], a[k]] = [a[k], a[i]];
    }
  if (current && a[0].class === current.dream.class) a.push(a.shift());
  return a;
}

function load(dream) {
  const frames = Array.from({ length: dream.frames }, (_, i) => {
    const img = new Image();
    img.decoding = "async";
    img.src = `assets/${dream.folder}/${String(i).padStart(2, "0")}.jpg`;
    return img;
  });
  return { dream, frames, ready: Promise.all(frames.map((f) => f.decode().catch(() => {}))) };
}

function pick() {
  if (!queue.length) queue = shuffled();
  return load(queue.shift());
}

async function advance() {
  current = next || pick();
  next = pick(); // preload while this one plays
  await current.ready;
  canvas.setAttribute("aria-label", `the model's dream of ${current.dream.class === "deer" ? "a deer" : `${/^[aeiou]/.test(current.dream.class) ? "an" : "a"} ${current.dream.class}`}`);
  word.classList.remove("on");
  start = performance.now();
}

function size() {
  const r = canvas.getBoundingClientRect(), k = Math.min(devicePixelRatio || 1, 2);
  const w = Math.round(r.width * k);
  if (canvas.width !== w) { canvas.width = canvas.height = w; }
  return w;
}

const ease = (t) => t * t * (3 - 2 * t);
function draw(now) {
  requestAnimationFrame(draw);
  if (!current || paused) return;
  const t = now - start;
  if (t >= TOTAL) { advance(); return; }
  const w = size(), frames = current.frames, n = frames.length;
  // Development: an even walk through the recorded frames.
  const dev = Math.min(1, Math.max(0, (t - T.in) / T.develop));
  const x = dev * (n - 1), i = Math.floor(x), f = x - i;
  ctx.globalAlpha = 1;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(frames[i], 0, 0, w, w);
  if (i + 1 < n && f > 0) { ctx.globalAlpha = f; ctx.drawImage(frames[i + 1], 0, 0, w, w); }
  // Exposure: fade up from black, down to black at the end.
  const fadeIn = ease(Math.min(1, t / T.in)), fadeOut = 1 - ease(Math.max(0, (t - (TOTAL - T.out)) / T.out));
  canvas.style.opacity = (fadeIn * fadeOut).toFixed(3);
  // Focus pulls in as it develops; a slow drift while it holds.
  const blur = (1 - ease(dev)) * 6;
  const drift = reduced ? 1 : 1 + 0.045 * ease(Math.max(0, (t - T.in - T.develop * 0.6) / (T.develop * 0.4 + T.hold + T.out)));
  canvas.style.filter = `blur(${blur.toFixed(2)}px)`;
  canvas.style.transform = `scale(${drift.toFixed(4)})`;
  // The word arrives once the image has, and leaves before it does.
  const showWord = dev > 0.62 && t < TOTAL - T.out - 600;
  if (showWord !== word.classList.contains("on")) {
    if (showWord) word.textContent = `${current.dream.class}.`;
    word.classList.toggle("on", showWord);
  }
}

// Film grain: fresh noise a dozen times a second.
// Fine, faint grain at close to display resolution (coarse grain read as TV static).
const grain = $("grain"), gctx = grain.getContext("2d");
grain.width = grain.height = 512;
const noise = gctx.createImageData(512, 512);
function regrain() {
  for (let i = 0; i < noise.data.length; i += 4) {
    const v = 128 + (Math.random() - 0.5) * 150;
    noise.data[i] = noise.data[i + 1] = noise.data[i + 2] = v;
    noise.data[i + 3] = 255;
  }
  gctx.putImageData(noise, 0, 0);
}
if (!reduced) setInterval(regrain, 83);
regrain();

// ----- controls: click / → next, space pause, i about -----
const about = $("about"), hint = $("hint"), title = $("title");
function skip() { if (!about.hidden) return; start = performance.now() - (TOTAL - T.out); }
function togglePause() {
  paused = !paused;
  if (paused) pausedAt = performance.now();
  else start += performance.now() - pausedAt;
}
function toggleAbout() {
  about.hidden = !about.hidden;
  document.body.classList.toggle("pointer", !about.hidden);
  if (!about.hidden && !paused) togglePause();
  else if (about.hidden && paused) togglePause();
}
addEventListener("keydown", (e) => {
  if (e.key === "ArrowRight") skip();
  else if (e.key === " ") { e.preventDefault(); togglePause(); }
  else if (e.key === "i" || (e.key === "Escape" && !about.hidden)) toggleAbout();
});
$("stage").addEventListener("click", skip);
about.addEventListener("click", toggleAbout);
let idle = 0;
addEventListener("pointermove", () => {
  hint.classList.add("on");
  document.body.classList.add("pointer");
  clearTimeout(idle);
  idle = setTimeout(() => { hint.classList.remove("on"); if (about.hidden) document.body.classList.remove("pointer"); }, 2500);
});

// Title card, then the first dream.
next = pick();
setTimeout(() => title.classList.add("gone"), 5200);
setTimeout(() => { hint.classList.add("on"); setTimeout(() => hint.classList.remove("on"), 4000); }, 6000);
setTimeout(async () => { await advance(); requestAnimationFrame(draw); }, 5600);
