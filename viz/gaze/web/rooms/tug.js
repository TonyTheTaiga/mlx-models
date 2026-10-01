// VI. Tug of War — you paint toward one class, the model answers each stroke pulling to another.
//
// A turn: you draw one stroke. The model then searches for its reply — 36 random strokes, then 8
// generations refining the best — scored by log p(its class) − log p(your class) of the whole
// canvas. Eight turns each; whoever's class the model believes more at the end wins. The canvas
// keeps every move.
import { h, pct, article, hiDPI } from "../lib/ui.js";
import { INKS, render, toPixels, drawPixels, clone } from "../lib/genome.js";
import { evaluate, softmax } from "../lib/pool.js";

const TURNS = 8, RANDOM_TRIES = 36, REFINE = 8, LAMBDA = 12;

export async function mount(root, { meta }) {
  const classes = meta.classes;
  const S = { alive: true, you: 3, it: 5, canvas: null, probs: null, turn: 0, busy: false, ink: 1, history: [], over: false };

  const rnd = Math.random, gauss = () => Math.sqrt(-2 * Math.log(1 - rnd())) * Math.cos(2 * Math.PI * rnd());
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const blank = () => ({ style: "engine", ground: 0, blobs: [], strokes: [], fixed: [] });
  const randomStroke = () => {
    const x0 = rnd(), y0 = rnd();
    return { x0, y0, cx: x0 + gauss() * 0.3, cy: y0 + gauss() * 0.3, x1: x0 + gauss() * 0.4, y1: y0 + gauss() * 0.4, w: 0.03 + rnd() * 0.07, ink: Math.floor(rnd() * INKS.length) };
  };
  const nudge = (s) => {
    const m = { ...s };
    for (const k of ["x0", "y0", "cx", "cy", "x1", "y1"]) m[k] = clamp(m[k] + gauss() * 0.06, -0.2, 1.2);
    if (rnd() < 0.3) m.w = clamp(m.w * Math.exp(gauss() * 0.2), 0.015, 0.14);
    if (rnd() < 0.1) m.ink = Math.floor(rnd() * INKS.length);
    return m;
  };
  // Small probabilities still need to read as a tug, so keep a significant digit below 1%.
  const fmt = (p) => (p >= 0.01 ? pct(p) : `${(p * 100).toPrecision(1)}%`);
  const margin = (p) => Math.log(Math.max(p[S.it], 1e-12)) - Math.log(Math.max(p[S.you], 1e-12));

  // ----- layout -----
  const art = h("canvas", { class: "art", role: "img", "aria-label": "the shared canvas" });
  const seen = h("canvas", { width: 64, height: 64, "aria-label": "what the model sees" });
  const plate = h("div", { class: "plate drawing" }, art, h("div", { class: "seen" }, seen, h("span", {}, "what the model sees")));
  const status = h("p", { class: "caption", "aria-live": "polite" });
  const rope = h("div", { class: "rope" });
  const history = h("div", { class: "history" });
  const pickYou = h("select", { "aria-label": "your class", onchange: (e) => { S.you = +e.target.value; newGame(); } });
  const pickIt = h("select", { "aria-label": "the model's class", onchange: (e) => { S.it = +e.target.value; newGame(); } });
  classes.forEach((c, i) => { pickYou.append(h("option", { value: i }, c)); pickIt.append(h("option", { value: i }, c)); });
  const inks = h("div", { class: "inks", role: "group", "aria-label": "your ink" },
    INKS.map((c, i) => h("button", { style: `background:${c}`, "aria-label": `ink ${i + 1}`, onclick: () => { S.ink = i; syncInks(); } })));
  const syncInks = () => [...inks.children].forEach((b, i) => b.classList.toggle("on", i === S.ink));

  root.append(h("section", { class: "engine tug" },
    h("div", { class: "wall" },
      h("p", { class: "eyebrow" }, h("b", {}, "VI"), " · Tug of War"),
      h("h1", { class: "display" }, "Paint against ", h("i", {}, "it"), "."),
      h("p", {}, "You want the model to see one thing; it wants to see another. Each of your strokes gets an answer — one stroke it searched for, the one that pulls hardest its way."),
      h("p", {}, `${TURNS} turns each. Whichever word the model believes more at the end wins. The canvas keeps every move.`),
      h("div", { class: "sides" }, h("label", {}, "you paint a ", pickYou), h("label", {}, "it paints a ", pickIt)),
      h("p", { class: "credit" }, "after the exquisite corpse, and adversarial examples")),
    h("div", { class: "work-col" }, plate, status, history),
    h("div", { class: "readout" },
      h("p", { class: "eyebrow" }, "The rope"), rope,
      h("p", { class: "eyebrow" }, "Your ink"), inks,
      h("div", { class: "actions" }, h("button", { onclick: () => newGame() }, "new game")))));

  // ----- drawing -----
  let live = null;
  function paint() {
    const size = Math.round(art.getBoundingClientRect().width) || 560, ctx = hiDPI(art, size);
    render(ctx, S.canvas, size);
    if (live) {
      ctx.strokeStyle = INKS[S.ink]; ctx.lineWidth = 0.04 * size; ctx.lineCap = ctx.lineJoin = "round";
      ctx.beginPath(); live.points.forEach(([x, y], i) => ctx[i ? "lineTo" : "moveTo"](x * size, y * size)); ctx.stroke();
    }
    drawPixels(seen.getContext("2d"), toPixels(S.canvas), 64);
    if (S.probs) {
      const a = S.probs[S.you], b = S.probs[S.it], share = a / (a + b || 1);
      rope.replaceChildren(
        h("div", { class: "ends" }, h("span", {}, `${classes[S.you]} ${fmt(a)}`), h("span", {}, `${fmt(b)} ${classes[S.it]}`)),
        h("div", { class: "track" }, h("div", { class: "you", style: `width:${(share * 100).toFixed(1)}%` }), h("div", { class: "knot", style: `left:${(share * 100).toFixed(1)}%` })),
        h("div", { class: "note" }, `the model's top guess: ${classes[S.probs.indexOf(Math.max(...S.probs))]}`));
    }
  }
  const pos = (e) => { const r = art.getBoundingClientRect(); return [(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height]; };
  art.addEventListener("pointerdown", (e) => { if (S.busy || S.over) return; art.setPointerCapture(e.pointerId); live = { points: [pos(e)] }; });
  art.addEventListener("pointermove", (e) => { if (live) { live.points.push(pos(e)); paint(); } });
  art.addEventListener("pointerup", async () => {
    if (!live) return;
    const stroke = live; live = null;
    if (stroke.points.length < 2) return paint();
    S.canvas = { ...clone(S.canvas), fixed: [...S.canvas.fixed, { points: stroke.points, ink: S.ink, w: 0.04 }] };
    await rescore();
    snapshot("you");
    await reply();
  });

  async function rescore() {
    const [l] = await evaluate([toPixels(S.canvas)]);
    S.probs = softmax(l);
    paint();
  }
  function snapshot(who) {
    const c = h("canvas", { title: `${who === "you" ? "your" : "its"} stroke ${S.turn + (who === "you" ? 1 : 0)}` });
    render(hiDPI(c, 56), S.canvas, 56);
    history.append(h("figure", { class: who }, c, h("figcaption", {}, who === "you" ? "you" : "it")));
  }

  // The model's reply: random search for a stroke, then refine the best one.
  async function reply() {
    S.busy = true;
    status.textContent = "It's thinking…";
    const withStroke = (s) => ({ ...clone(S.canvas), strokes: [...S.canvas.strokes, s] });
    let cands = Array.from({ length: RANDOM_TRIES }, randomStroke);
    let res = await evaluate(cands.map((s) => toPixels(withStroke(s))));
    let scored = res.map((l, i) => ({ s: cands[i], p: softmax(l) })).map((x) => ({ ...x, m: margin(x.p) }));
    let best = scored.reduce((a, b) => (b.m > a.m ? b : a));
    for (let g = 0; g < REFINE && S.alive; g++) {
      cands = Array.from({ length: LAMBDA }, () => nudge(best.s));
      res = await evaluate(cands.map((s) => toPixels(withStroke(s))));
      for (let i = 0; i < cands.length; i++) {
        const p = softmax(res[i]), m = margin(p);
        if (m > best.m) best = { s: cands[i], p, m };
      }
    }
    if (!S.alive) return;
    S.canvas = withStroke(best.s);
    S.probs = best.p;
    S.turn++;
    snapshot("it");
    S.busy = false;
    paint();
    if (S.turn >= TURNS) return finish();
    const lead = S.probs[S.you] >= S.probs[S.it];
    status.textContent = `Turn ${S.turn + 1} of ${TURNS}. ${lead ? `It still sees more ${classes[S.you]} than ${classes[S.it]}.` : `It's pulling toward ${classes[S.it]}.`} Draw your stroke.`;
  }

  function finish() {
    S.over = true;
    const a = S.probs[S.you], b = S.probs[S.it], won = a > b;
    status.textContent = won
      ? `You won: it sees ${article(classes[S.you])} ${classes[S.you]} (${pct(a)}) over ${article(classes[S.it])} ${classes[S.it]} (${pct(b)}).`
      : `It won: it sees ${article(classes[S.it])} ${classes[S.it]} (${pct(b)}) over ${article(classes[S.you])} ${classes[S.you]} (${pct(a)}).`;
  }

  async function newGame() {
    if (S.you === S.it) S.it = (S.you + 1) % classes.length;
    pickYou.value = S.you; pickIt.value = S.it;
    Object.assign(S, { canvas: blank(), turn: 0, busy: false, over: false });
    history.replaceChildren();
    await rescore();
    const blankTop = classes[S.probs.indexOf(Math.max(...S.probs))];
    status.textContent = `A blank page already looks like ${article(blankTop)} ${blankTop} to it. Turn 1 of ${TURNS}: draw a stroke that makes it see ${article(classes[S.you])} ${classes[S.you]}.`;
  }

  // Start with a random pairing.
  S.you = Math.floor(Math.random() * classes.length);
  S.it = (S.you + 1 + Math.floor(Math.random() * (classes.length - 1))) % classes.length;
  const ro = new ResizeObserver(() => S.canvas && paint());
  ro.observe(art);
  syncInks();
  await newGame();
  return () => { S.alive = false; ro.disconnect(); };
}
