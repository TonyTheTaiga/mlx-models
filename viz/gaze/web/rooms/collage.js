// V. Collage — the most convincing <class> the model can find, assembled from real fragments.
//
// The patch bank is every 4×4 patch of 1,000 test photographs (64,000 fragments). Evolution swaps
// fragments in and out of an 8×8 grid until the model is certain; then it tries to replace each
// fragment cut from a photograph of the target class with one from something else, keeping the
// change only if the model stays ≥99% sure. The question it ends on: how few real cats does a cat
// take?
import { h, word, cap, pct, article, plural, hiDPI } from "../lib/ui.js";
import { evaluate, softmax } from "../lib/pool.js";

const LAMBDA = 12, CERTAIN = 0.995, KEEP = 0.99, PATIENCE = 40;

export async function mount(root, { meta }) {
  const [P, bank] = await Promise.all([
    fetch("assets/pool.json").then((r) => r.json()),
    fetch("assets/pool.bin").then((r) => r.arrayBuffer()).then((b) => new Uint8Array(b)),
  ]);
  const classes = meta.classes, N = P.labels.length, TOTAL = N * 64;
  const S = { alive: true, running: true, target: Math.floor(Math.random() * classes.length), genome: null, fit: -Infinity, probs: null, gen: 0, phase: "compose", stall: 0, prints: 0 };
  const srcImage = (id) => Math.floor(id / 64), srcClass = (id) => P.labels[srcImage(id)];
  const randomId = () => Math.floor(Math.random() * TOTAL);

  // Compose 32×32 pixels from 64 fragment ids.
  function pixels(g) {
    const out = new Uint8ClampedArray(3072);
    for (let t = 0; t < 64; t++) {
      const id = g[t], img = srcImage(id), p = id % 64, sr = Math.floor(p / 8) * 4, sc = (p % 8) * 4;
      const tr = Math.floor(t / 8) * 4, tc = (t % 8) * 4;
      for (let i = 0; i < 4; i++)
        for (let j = 0; j < 4; j++) {
          const s = img * 3072 + ((sr + i) * 32 + sc + j) * 3, d = ((tr + i) * 32 + tc + j) * 3;
          out[d] = bank[s]; out[d + 1] = bank[s + 1]; out[d + 2] = bank[s + 2];
        }
    }
    return out;
  }
  async function score(gs) {
    const logits = await evaluate(gs.map(pixels));
    return logits.map((l) => { const p = softmax(l); return { p, fit: Math.log(Math.max(p[S.target], 1e-12)) }; });
  }
  const own = (g) => [...g].filter((id) => srcClass(id) === S.target).length;
  const photos = (g) => new Set([...g].map(srcImage)).size;

  // ----- layout -----
  const headline = h("h1", { class: "display" });
  const art = h("canvas", { class: "art", role: "img", "aria-label": "the collage" });
  const tip = h("div", { class: "tiletip", hidden: true });
  const plate = h("div", { class: "plate" }, art, tip);
  const caption = h("p", { class: "caption", "aria-live": "polite" });
  const big = h("div", { class: "big" }, "—"), sub = h("div", { class: "sub" });
  const census = h("div", { class: "census" });
  const targets = h("div", { class: "targets", role: "group", "aria-label": "what should it be?" },
    classes.map((c, i) => h("button", { "aria-pressed": i === S.target, onclick: () => setTarget(i) }, c)));
  const runBtn = h("button", { onclick: () => { S.running = !S.running; if (S.running) loop(); sync(); } });
  root.append(h("section", { class: "engine collage" },
    h("div", { class: "wall" },
      h("p", { class: "eyebrow" }, h("b", {}, "V"), " · Collage"),
      headline,
      h("p", {}, `Sixty-four thousand fragments — every 4×4 patch of ${N.toLocaleString()} test photographs. Evolution swaps them in and out of an 8×8 grid until the model is certain.`),
      h("p", {}, "Then it tries to replace every fragment cut from a real photograph of the thing, keeping each swap only if the model stays sure. Hover a tile to see where it came from."),
      h("p", { class: "credit" }, "after the photomontages of Hannah Höch, and ", h("a", { href: "https://aiartists.org/tom-white" }, "Tom White, Perception Engines"))),
    h("div", { class: "work-col" }, plate, caption),
    h("div", { class: "readout" },
      h("p", { class: "eyebrow" }, "Make it"), targets,
      h("p", { class: "eyebrow" }, "Its certainty"), big, sub, census,
      h("div", { class: "actions" }, runBtn, h("button", { onclick: () => restart() }, "start over"), h("button", { class: "accent", onclick: print }, "⤓ print")))));
  function sync() {
    const cls = classes[S.target];
    headline.replaceChildren(`How many ${plural(cls)} does ${article(cls)} `, h("i", {}, cls), " take?");
    runBtn.textContent = S.running ? "❚❚ pause" : "▶ evolve";
    [...targets.children].forEach((b, i) => b.setAttribute("aria-pressed", i === S.target));
  }

  // ----- drawing: tiles with a hairline of paper between them, a collage's seams -----
  let size = 0, hover = -1;
  function paint() {
    size = Math.round(art.getBoundingClientRect().width) || 560;
    const ctx = hiDPI(art, size), cell = size / 8, px = cell / 4, gap = Math.max(1, size / 400);
    ctx.fillStyle = "#f1ede4"; ctx.fillRect(0, 0, size, size);
    if (!S.genome) return;
    const img = pixels(S.genome);
    for (let t = 0; t < 64; t++) {
      const x0 = (t % 8) * cell, y0 = Math.floor(t / 8) * cell;
      for (let i = 0; i < 4; i++)
        for (let j = 0; j < 4; j++) {
          const o = ((Math.floor(t / 8) * 4 + i) * 32 + (t % 8) * 4 + j) * 3;
          ctx.fillStyle = `rgb(${img[o]},${img[o + 1]},${img[o + 2]})`;
          ctx.fillRect(x0 + gap + j * (px - (2 * gap) / 4), y0 + gap + i * (px - (2 * gap) / 4), px, px);
        }
      if (srcClass(S.genome[t]) === S.target) {
        ctx.strokeStyle = "#d9431e"; ctx.lineWidth = 2;
        ctx.strokeRect(x0 + gap + 1, y0 + gap + 1, cell - 2 * gap - 2, cell - 2 * gap - 2);
      }
      if (t === hover) { ctx.strokeStyle = "#151412"; ctx.lineWidth = 2; ctx.strokeRect(x0 + 1, y0 + 1, cell - 2, cell - 2); }
    }
    if (!S.probs) return;
    const p = S.probs[S.target], top = S.probs.indexOf(Math.max(...S.probs)), k = own(S.genome), n = photos(S.genome);
    const cls = classes[S.target];
    big.textContent = pct(p, p > 0.99 ? 1 : 0);
    sub.textContent = `generation ${S.gen} · ${{ compose: "composing", purify: `removing real ${plural(cls)}`, done: "done" }[S.phase]}`;
    census.replaceChildren(
      h("div", {}, h("b", {}, String(n)), ` photographs, cut into 64 fragments`),
      h("div", {}, h("b", { class: "acc" }, String(k)), ` fragment${k === 1 ? "" : "s"} from real ${plural(cls)} (outlined)`));
    caption.textContent = S.phase === "done"
      ? k === 0 ? `${cap(article(cls))} ${cls} made entirely of other things. It is ${pct(p, 1)} sure.` : `${cap(word(k))} fragment${k === 1 ? "" : "s"} of real ${k === 1 ? cls : plural(cls)}, and the rest is everything else. It is ${pct(p, 1)} sure.`
      : top === S.target ? `It sees ${article(cls)} ${cls}.` : `It sees ${article(classes[top])} ${classes[top]} — so far.`;
  }
  art.addEventListener("pointermove", (e) => {
    if (!S.genome) return;
    const r = art.getBoundingClientRect(), t = Math.floor(((e.clientY - r.top) / r.height) * 8) * 8 + Math.floor(((e.clientX - r.left) / r.width) * 8);
    if (t === hover) return;
    hover = t;
    const id = S.genome[t], img = srcImage(id), c = h("canvas", {});
    const ctx = hiDPI(c, 96), im = ctx.createImageData(32, 32);
    for (let i = 0; i < 1024; i++) { for (let k = 0; k < 3; k++) im.data[i * 4 + k] = bank[img * 3072 + i * 3 + k]; im.data[i * 4 + 3] = 255; }
    const tmp = Object.assign(document.createElement("canvas"), { width: 32, height: 32 });
    tmp.getContext("2d").putImageData(im, 0, 0);
    ctx.imageSmoothingEnabled = false; ctx.drawImage(tmp, 0, 0, 96, 96);
    const p = id % 64; ctx.strokeStyle = "#d9431e"; ctx.lineWidth = 2; ctx.strokeRect((p % 8) * 12, Math.floor(p / 8) * 12, 12, 12);
    tip.replaceChildren(c, h("span", {}, `from ${article(classes[P.labels[img]])} ${classes[P.labels[img]]}`));
    tip.hidden = false;
    tip.style.left = `${((t % 8) + 1) * (r.width / 8) + 8}px`;
    tip.style.top = `${Math.floor(t / 8) * (r.height / 8)}px`;
    paint();
  });
  art.addEventListener("pointerleave", () => { hover = -1; tip.hidden = true; paint(); });

  // ----- evolution -----
  async function restart() {
    const starts = Array.from({ length: LAMBDA }, () => Int32Array.from({ length: 64 }, randomId));
    const res = await score(starts);
    const b = res.reduce((bi, r, i) => (r.fit > res[bi].fit ? i : bi), 0);
    Object.assign(S, { genome: starts[b], fit: res[b].fit, probs: res[b].p, gen: 0, phase: "compose", stall: 0 });
    paint();
    if (!S.running) { S.running = true; sync(); loop(); }
  }
  async function setTarget(i) {
    S.target = i; S.gen = 0; S.phase = "compose"; S.stall = 0;
    sync();
    const [r] = await score([S.genome]);
    S.fit = r.fit; S.probs = r.p;
    paint();
    if (!S.running) { S.running = true; sync(); loop(); }
  }
  let looping = false;
  async function loop() {
    if (looping) return;
    looping = true;
    while (S.alive && S.running) {
      let kids;
      if (S.phase === "purify") {
        // Replace one real-<class> fragment with a fragment from something else.
        const mine = [...S.genome.keys()].filter((t) => srcClass(S.genome[t]) === S.target);
        if (!mine.length) { S.phase = "done"; S.running = false; sync(); paint(); break; }
        kids = Array.from({ length: LAMBDA }, () => {
          const g = Int32Array.from(S.genome), t = mine[Math.floor(Math.random() * mine.length)];
          let id; do id = randomId(); while (srcClass(id) === S.target);
          g[t] = id;
          return g;
        });
      } else {
        kids = Array.from({ length: LAMBDA }, () => {
          const g = Int32Array.from(S.genome), n = 1 + Math.floor(Math.random() * 3);
          for (let k = 0; k < n; k++) g[Math.floor(Math.random() * 64)] = randomId();
          return g;
        });
      }
      const res = await score(kids);
      if (!S.alive) break;
      S.gen++;
      if (S.phase === "purify") {
        const ok = res.map((r, i) => [r, i]).filter(([r]) => r.p[S.target] >= KEEP);
        if (ok.length) {
          const [r, i] = ok.reduce((a, b) => (b[0].fit > a[0].fit ? b : a));
          Object.assign(S, { genome: kids[i], fit: r.fit, probs: r.p, stall: 0 });
        } else if (++S.stall > PATIENCE) { S.phase = "done"; S.running = false; sync(); }
      } else {
        const b = res.reduce((bi, r, i) => (r.fit > res[bi].fit ? i : bi), 0);
        if (res[b].fit >= S.fit) Object.assign(S, { genome: kids[b], fit: res[b].fit, probs: res[b].p });
        if (S.probs[S.target] >= CERTAIN) S.phase = "purify";
      }
      paint();
    }
    looping = false;
  }

  function print() {
    const W = 1600, H = 2100, c = document.createElement("canvas");
    c.width = W; c.height = H;
    const ctx = c.getContext("2d");
    ctx.fillStyle = "#f1ede4"; ctx.fillRect(0, 0, W, H);
    ctx.drawImage(art, 200, 200, 1200, 1200);
    const cls = classes[S.target], k = own(S.genome);
    ctx.fillStyle = "#151412"; ctx.font = "italic 400 120px 'Instrument Serif', Georgia, serif";
    ctx.fillText(cap(cls), 200, 1560);
    ctx.font = "400 30px 'IBM Plex Mono', monospace"; ctx.fillStyle = "#4f4b43";
    [`The model is ${pct(S.probs[S.target], 1)} sure this is ${article(cls)} ${cls}.`,
      `64 fragments of ${photos(S.genome)} photographs; ${k} from real ${plural(cls)}.`,
      "Collage · GAZE",
      `Vision transformer, ${meta.cfg.layers} blocks × ${meta.cfg.heads} heads, CIFAR-10 · ${new Date().toISOString().slice(0, 10)}`,
    ].forEach((t, i) => ctx.fillText(t, 200, 1650 + i * 52));
    c.toBlob((blob) => {
      const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(blob), download: `gaze-collage-${cls}-${++S.prints}.png` });
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    });
  }

  const ro = new ResizeObserver(() => paint());
  ro.observe(art);
  sync();
  await restart();
  loop();
  return () => { S.alive = false; ro.disconnect(); };
}
