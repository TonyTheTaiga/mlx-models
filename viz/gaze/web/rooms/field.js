// IV. Hallucination Field — six images drift between the classes the model knows.
//
// Each cell evolves a soft-focus drawing toward its current class (a small (1+λ) loop, all cells
// scored in one batch across the worker pool) and moves on to another class every so often.
// Certainty brings the image into focus and the class name up out of the dark. Nothing here is
// photographed; every image exists only because the model is asked to see something.
import { h } from "../lib/ui.js";
import { randomGenome, mutate, render, toPixels } from "../lib/genome.js";
import { evaluate, softmax } from "../lib/pool.js";

const CELLS = 6, LAMBDA = 5, DWELL = 40000;

export async function mount(root, { meta }) {
  const classes = meta.classes;
  const S = { alive: true };
  const cells = Array.from({ length: CELLS }, (_, i) => {
    const canvas = h("canvas", { "aria-hidden": "true" });
    const name = h("div", { class: "name" });
    const el = h("div", { class: "cell" }, canvas, name);
    return {
      el, canvas, name,
      target: (i * 3 + Math.floor(Math.random() * 10)) % classes.length,
      elite: randomGenome("field"), fit: -Infinity, p: 0,
      since: performance.now() - (i * DWELL) / CELLS, // staggered, so they don't all change at once
      shown: null, // the drawing on screen, eased toward the elite
    };
  });
  const caption = h("p", { class: "caption" }, "Six images, each evolving toward whatever the model is asked to see. Nothing here was photographed.");
  root.append(h("section", { class: "field", role: "img", "aria-label": "an ambient field of evolving images" }, cells.map((c) => c.el)), caption);

  // Evolution: one batch per generation for all cells.
  async function evolve() {
    while (S.alive) {
      if (document.hidden) { await new Promise((r) => setTimeout(r, 500)); continue; }
      const now = performance.now();
      for (const c of cells)
        if (now - c.since > DWELL) {
          c.target = (c.target + 1 + Math.floor(Math.random() * (classes.length - 1))) % classes.length;
          c.fit = -Infinity; c.since = now;
        }
      const kids = cells.flatMap((c) => Array.from({ length: LAMBDA }, () => mutate(c.elite, 0.7)));
      const logits = await evaluate(kids.map(toPixels));
      if (!S.alive) return;
      cells.forEach((c, i) => {
        for (let j = 0; j < LAMBDA; j++) {
          const p = softmax(logits[i * LAMBDA + j]), fit = Math.log(Math.max(p[c.target], 1e-12));
          if (fit >= c.fit) { c.fit = fit; c.elite = kids[i * LAMBDA + j]; c.p = p[c.target]; }
        }
      });
    }
  }

  // Display: ease blob positions and sizes toward the elite so changes arrive as drift.
  let raf = 0;
  function frame() {
    raf = requestAnimationFrame(frame);
    for (const c of cells) {
      const r = c.el.getBoundingClientRect(), size = Math.round(Math.max(r.width, r.height) * 1.12);
      if (!size) continue;
      if (c.canvas.width !== size) { c.canvas.width = c.canvas.height = size; }
      c.shown = c.shown ? ease(c.shown, c.elite) : structuredClone(c.elite);
      render(c.canvas.getContext("2d"), c.shown, size);
      const blur = 14 - 12 * c.p, sat = 0.5 + 0.9 * c.p;
      c.canvas.style.filter = `blur(${blur.toFixed(1)}px) saturate(${sat.toFixed(2)})`;
      c.name.textContent = classes[c.target];
      c.name.style.opacity = (0.08 + 0.82 * c.p).toFixed(2);
    }
  }
  function ease(a, b) {
    const t = 0.06, mix = (x, y) => x + (y - x) * t;
    if (a.blobs.length !== b.blobs.length || a.strokes.length !== b.strokes.length) return structuredClone(b);
    return {
      ...b,
      blobs: b.blobs.map((bb, i) => ({ ...bb, x: mix(a.blobs[i].x, bb.x), y: mix(a.blobs[i].y, bb.y), r: mix(a.blobs[i].r, bb.r) })),
      strokes: b.strokes.map((s, i) => {
        const o = a.strokes[i], out = { ...s };
        for (const k of ["x0", "y0", "cx", "cy", "x1", "y1", "w"]) out[k] = mix(o[k], s[k]);
        return out;
      }),
    };
  }

  raf = requestAnimationFrame(frame);
  evolve();
  const onKey = (e) => { if (e.key === "Escape") location.hash = "#/"; };
  addEventListener("keydown", onKey);
  return () => { S.alive = false; cancelAnimationFrame(raf); removeEventListener("keydown", onKey); };
}
