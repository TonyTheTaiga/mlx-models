// I. Perception Engine — evolve a drawing in a fixed vocabulary until the model is certain.
//
// (1+λ) evolution: each generation proposes λ mutants of the current drawing, the worker pool
// scores them by log p(target), and the best replaces the current drawing only if it's at least
// as convincing. Strokes a person draws are fixed; evolution composes around them.
import { h, word, cap, pct, hiDPI } from "../lib/ui.js";
import { INKS, randomGenome, mutate, render, toPixels, drawPixels, markCount, removals } from "../lib/genome.js";
import { evaluate, softmax } from "../lib/pool.js";

const LAMBDA = 12;
const CERTAIN = 0.995; // phase 1 ends here
const KEEP = 0.99; // phase 2 removes marks while the model stays at least this sure

export async function mount(root, { meta }) {
  const classes = meta.classes;
  const S = {
    target: Math.floor(Math.random() * classes.length), elite: null, fit: -Infinity, probs: null, gen: 0,
    phase: "compose", startMarks: 0, // compose → reduce → done
    running: true, alive: true, drawing: false, ink: 1, prints: 0,
  };

  // ----- layout -----
  const art = h("canvas", { class: "art", role: "img", "aria-label": "the evolving drawing" });
  const seen = h("canvas", { width: 64, height: 64, "aria-label": "what the model sees: 32 by 32 pixels" });
  const plate = h("div", { class: "plate" }, art,
    h("div", { class: "seen" }, seen, h("span", {}, "what the model sees — 32 × 32 pixels")));
  const caption = h("p", { class: "caption", "aria-live": "polite" });
  const big = h("div", { class: "big" }, "—");
  const sub = h("div", { class: "sub" });
  const bars = h("div", { class: "bars" });
  const targets = h("div", { class: "targets", role: "group", "aria-label": "what should it see?" },
    classes.map((c, i) => h("button", { "aria-pressed": i === S.target, onclick: () => setTarget(i) }, c)));
  const runBtn = h("button", { onclick: () => { S.running = !S.running; if (S.running) loop(); syncButtons(); } });
  const drawBtn = h("button", { onclick: () => { S.drawing = !S.drawing; plate.classList.toggle("drawing", S.drawing); syncButtons(); } });
  const inks = h("div", { class: "inks", role: "group", "aria-label": "ink" },
    INKS.map((c, i) => h("button", { style: `background:${c}`, "aria-label": `ink ${i + 1}`, onclick: () => { S.ink = i; syncButtons(); } })));
  const printBtn = h("button", { class: "accent", onclick: print }, "⤓ print");
  const freshBtn = h("button", { onclick: () => restart() }, "new canvas");

  root.append(h("section", { class: "engine" },
    h("div", { class: "wall" },
      h("p", { class: "eyebrow" }, h("b", {}, "I"), " · Perception Engine"),
      h("h1", { class: "display" }, "What does it ", h("i", {}, "need"), " to see?"),
      h("p", {}, "Eight inks, three shapes and eight strokes. An evolutionary loop moves them, one small change at a time, keeping only changes that make the model more certain of the word you choose."),
      h("p", {}, "Once it is certain, the loop starts taking marks away — each removal kept only if the model stays sure. What's left is what it needed."),
      h("p", {}, "The result rarely looks like the thing. It looks like what the model has learned the thing ", h("i", {}, "is"), "."),
      h("p", {}, "Draw into it and evolution composes around your marks."),
      h("p", { class: "credit" }, "after ", h("a", { href: "https://aiartists.org/tom-white" }, "Tom White, Perception Engines"), " and ", h("a", { href: "https://arxiv.org/abs/1712.09665" }, "Brown et al., Adversarial Patch"))),
    h("div", { class: "work-col" }, plate, caption),
    h("div", { class: "readout" },
      h("p", { class: "eyebrow" }, "Make it see"), targets,
      h("p", { class: "eyebrow" }, "Its certainty"), big, sub, bars,
      h("div", { class: "actions" }, runBtn, drawBtn, freshBtn, printBtn),
      inks)));

  function syncButtons() {
    runBtn.textContent = S.running ? "❚❚ pause" : "▶ evolve";
    drawBtn.textContent = S.drawing ? "✓ done drawing" : "✎ draw";
    drawBtn.setAttribute("aria-pressed", S.drawing);
    inks.hidden = !S.drawing;
    [...inks.children].forEach((b, i) => b.classList.toggle("on", i === S.ink));
    [...targets.children].forEach((b, i) => b.setAttribute("aria-pressed", i === S.target));
  }

  // ----- drawing -----
  let artSize = 0;
  function paint() {
    const w = art.getBoundingClientRect().width || 560;
    if (Math.abs(w - artSize) > 1) artSize = w;
    const ctx = hiDPI(art, artSize);
    if (S.elite) render(ctx, S.elite, artSize);
    if (live) {
      ctx.strokeStyle = INKS[S.ink]; ctx.lineWidth = 0.04 * artSize; ctx.lineCap = ctx.lineJoin = "round";
      ctx.beginPath(); live.points.forEach(([x, y], i) => ctx[i ? "lineTo" : "moveTo"](x * artSize, y * artSize)); ctx.stroke();
    }
    if (S.elite) drawPixels(seen.getContext("2d"), toPixels(S.elite), 64);
    if (S.probs) {
      const p = S.probs[S.target], top = S.probs.indexOf(Math.max(...S.probs));
      big.textContent = pct(p, p > 0.99 ? 1 : 0);
      const phase = { compose: "composing", reduce: "taking marks away", done: "done" }[S.phase];
      sub.textContent = `generation ${S.gen} · ${markCount(S.elite)} marks · ${phase}`;
      const order = [...S.probs.keys()].sort((a, b) => S.probs[b] - S.probs[a]).slice(0, 5);
      if (!order.includes(S.target)) order[4] = S.target;
      bars.replaceChildren(...order.map((c) => h("div", { class: `row ${c === S.target ? "target" : ""}` },
        h("span", {}, classes[c]), h("span", { class: "track" }, h("span", { class: "fill", style: `display:block;width:${(S.probs[c] * 100).toFixed(1)}%` })), h("span", {}, pct(S.probs[c])))));
      const n = markCount(S.elite), marks = `${word(n)} mark${n === 1 ? "" : "s"}`;
      caption.textContent = S.phase === "done"
        ? `${cap(marks)} were enough. It sees ${article(classes[S.target])} ${classes[S.target]}.`
        : top === S.target
          ? `You see ${marks}. It sees ${article(classes[S.target])} ${classes[S.target]}${p >= CERTAIN ? ", and it is certain." : "."}`
          : `You see ${marks}. It sees ${article(classes[top])} ${classes[top]} — so far.`;
    }
  }
  const article = (w) => (/^[aeiou]/.test(w) ? "an" : "a");

  let live = null;
  const pos = (e) => { const r = art.getBoundingClientRect(); return [(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height]; };
  art.addEventListener("pointerdown", (e) => { if (!S.drawing) return; art.setPointerCapture(e.pointerId); live = { points: [pos(e)] }; paint(); });
  art.addEventListener("pointermove", (e) => { if (live) { live.points.push(pos(e)); paint(); } });
  art.addEventListener("pointerup", async () => {
    if (!live) return;
    if (live.points.length > 1) {
      S.elite.fixed = [...S.elite.fixed, { points: live.points, ink: S.ink, w: 0.04 }];
      await rescore();
      S.phase = "compose";
      if (!S.running) { S.running = true; syncButtons(); loop(); }
    }
    live = null;
    paint();
  });

  // ----- evolution -----
  async function score(genomes) {
    const logits = await evaluate(genomes.map(toPixels));
    return logits.map((l) => {
      const p = softmax(l);
      return { p, fit: Math.log(Math.max(p[S.target], 1e-12)) };
    });
  }
  async function rescore() {
    const [r] = await score([S.elite]);
    S.fit = r.fit;
    S.probs = r.p;
  }
  async function restart() {
    // Best of a few random starts.
    const starts = Array.from({ length: LAMBDA }, () => randomGenome("engine"));
    const fixed = S.elite?.fixed ?? [];
    starts.forEach((g) => (g.fixed = fixed));
    const res = await score(starts);
    const b = res.reduce((bi, r, i) => (r.fit > res[bi].fit ? i : bi), 0);
    Object.assign(S, { elite: starts[b], fit: res[b].fit, probs: res[b].p, gen: 0, phase: "compose" });
    paint();
    if (!S.running) { S.running = true; syncButtons(); loop(); }
  }
  async function setTarget(i) {
    S.target = i;
    S.gen = 0;
    S.phase = "compose";
    syncButtons();
    await rescore();
    paint();
    if (!S.running) { S.running = true; syncButtons(); loop(); }
  }
  let looping = false;
  async function loop() {
    if (looping) return;
    looping = true;
    while (S.alive && S.running) {
      if (S.phase === "reduce") {
        // Try every single removal; keep the surest one that stays above KEEP.
        const kids = removals(S.elite);
        const res = kids.length ? await score(kids) : [];
        if (!S.alive) break;
        S.gen++;
        const ok = res.map((r, i) => [r, i]).filter(([r]) => r.p[S.target] >= KEEP);
        if (ok.length) {
          const [r, i] = ok.reduce((a, b) => (b[0].fit > a[0].fit ? b : a));
          Object.assign(S, { elite: kids[i], fit: r.fit, probs: r.p });
        } else { S.phase = "done"; S.running = false; syncButtons(); }
        paint();
        continue;
      }
      // Bigger steps while it's far from certain, finer ones as it closes in.
      const rate = S.probs && S.probs[S.target] > 0.9 ? 0.5 : 1;
      const kids = Array.from({ length: LAMBDA }, () => mutate(S.elite, rate));
      const res = await score(kids);
      if (!S.alive) break;
      const b = res.reduce((bi, r, i) => (r.fit > res[bi].fit ? i : bi), 0);
      S.gen++;
      if (res[b].fit >= S.fit) { S.elite = kids[b]; S.fit = res[b].fit; S.probs = res[b].p; }
      if (S.probs[S.target] >= CERTAIN) { S.phase = "reduce"; S.startMarks = markCount(S.elite); }
      paint();
    }
    looping = false;
  }

  // ----- print: the drawing as a labeled poster -----
  function print() {
    const W = 1600, H = 2100, c = document.createElement("canvas");
    c.width = W; c.height = H;
    const ctx = c.getContext("2d");
    ctx.fillStyle = "#f1ede4"; ctx.fillRect(0, 0, W, H);
    ctx.save(); ctx.translate(200, 200);
    render(ctx, S.elite, 1200);
    ctx.restore();
    const cls = classes[S.target], p = S.probs[S.target];
    ctx.fillStyle = "#151412";
    ctx.font = "italic 400 120px 'Instrument Serif', Georgia, serif";
    ctx.fillText(cap(cls), 200, 1560);
    ctx.font = "400 30px 'IBM Plex Mono', monospace";
    ctx.fillStyle = "#4f4b43";
    const lines = [
      `The model is ${pct(p, 1)} sure this is ${article(cls)} ${cls}.`,
      `${cap(word(markCount(S.elite)))} marks, ${S.gen} generations of evolution.`,
      `Perception Engine · GAZE · after Tom White`,
      `Vision transformer, ${meta.cfg.layers} blocks × ${meta.cfg.heads} heads, CIFAR-10 · ${new Date().toISOString().slice(0, 10)}`,
    ];
    lines.forEach((t, i) => ctx.fillText(t, 200, 1650 + i * 52));
    const px = toPixels(S.elite), t = document.createElement("canvas");
    t.width = t.height = 32;
    const img = t.getContext("2d").createImageData(32, 32);
    for (let i = 0; i < 1024; i++) { for (let k = 0; k < 3; k++) img.data[i * 4 + k] = px[i * 3 + k]; img.data[i * 4 + 3] = 255; }
    t.getContext("2d").putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(t, 1400 - 160, 1440, 160, 160);
    ctx.font = "400 22px 'IBM Plex Mono', monospace";
    ctx.fillText("what it sees", 1400 - 160, 1630);
    c.toBlob((blob) => {
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `gaze-perception-engine-${cls}-${++S.prints}.png`;
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    });
  }

  const ro = new ResizeObserver(() => paint());
  ro.observe(art);
  syncButtons();
  await restart();
  loop();
  return () => { S.alive = false; ro.disconnect(); };
}
