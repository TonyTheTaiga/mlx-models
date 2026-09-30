// III. Plausible Lies — two maps claim to show where the model looked; a deletion test decides.
//
// Each round pairs an occlusion map with a Grad-CAM map for the same photograph (left/right at
// random, unlabeled). You pick the one you'd trust. Then both are tested: patches are flattened
// hottest-first and the model's confidence is replayed from curves computed in MLX. Whichever map
// makes the prediction collapse faster (lower mean confidence over k = 0..64) wins — the answer
// isn't fixed in advance, so Grad-CAM wins the rounds where it is actually right.
import { h, pct, word, cap, article, loadImages, imageAt, drawImage, drawHeat, hiDPI, HEAT_GRADIENT } from "../lib/ui.js";

const ROUNDS = 10;
const NAMES = { occlusion: "Occlusion", gradcam: "Grad-CAM" };
const WHAT = {
  occlusion: "hides each patch and measures what the prediction loses",
  gradcam: "weights the network's last patch features by their gradients — the most-cited heatmap method",
};

export async function mount(root) {
  const [L, imgs] = await Promise.all([fetch("assets/lies.json").then((r) => r.json()), loadImages("assets/lies.bin")]);
  const C = L.classes;
  const S = { alive: true, round: 0, score: 0, rounds: shuffle([...L.rounds]).slice(0, ROUNDS), anim: 0 };

  function shuffle(a) { for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; }
  const auc = (curve) => curve.reduce((a, b) => a + b, 0) / curve.length;
  const halfAt = (curve) => { const k = curve.findIndex((p) => p < curve[0] / 2); return k < 0 ? 64 : k; };
  const patches = (n) => `${word(n)} ${n === 1 ? "patch" : "patches"}`;
  const norm = (v) => { const m = Math.max(1e-12, ...v); return v.map((x) => Math.max(0, x) / m); };

  const stage = h("section", { class: "lies" });
  root.append(stage);

  function play() {
    const r = S.rounds[S.round];
    const px = imageAt(imgs, r.slot), cls = C[r.pred];
    const sides = Math.random() < 0.5 ? ["occlusion", "gradcam"] : ["gradcam", "occlusion"];
    const cards = sides.map((method, i) => {
      const map = h("canvas", { class: "map", role: "img", "aria-label": `map ${"AB"[i]}` });
      const curve = h("canvas", { class: "curve", "aria-hidden": "true" });
      const who = h("div", { class: "who" }, h("b", {}, "AB"[i]), h("span", {}, ""));
      const pickBtn = h("button", { onclick: () => choose(i) }, `Trust ${"AB"[i]}`);
      return { method, map, curve, who, pickBtn, el: h("div", { class: "card" }, who, map, curve, pickBtn) };
    });
    const verdict = h("p", { class: "verdict", "aria-live": "polite" }, "");
    const next = h("button", { class: "accent", hidden: true, onclick: () => { S.round++; S.round < S.rounds.length ? play() : finish(); } },
      S.round + 1 < S.rounds.length ? "Next round →" : "See how you did →");
    stage.replaceChildren(
      h("p", { class: "eyebrow" }, h("b", {}, "III"), " · Plausible Lies"),
      h("p", { class: "progress" }, `round ${S.round + 1} of ${S.rounds.length} · you: ${S.score}`),
      h("h1", { class: "prompt" }, "Which map shows what the model actually used?"),
      h("p", { class: "wall" }, `The model says `, h("b", {}, `${article(cls)} ${cls}`), ` (${pct(r.confidence)}). Both maps claim to show why. Pick the one you'd trust.`),
      h("div", { class: "pair" }, cards.map((c) => c.el)),
      h("div", { class: "scale" }, h("span", {}, "ignored"), h("i", { style: `background:${HEAT_GRADIENT}` }), h("span", {}, "“looked here”")),
      verdict, next);

    const draw = (card, k = 0) => {
      const size = Math.round(card.map.getBoundingClientRect().width) || 360;
      const ctx = hiDPI(card.map, size);
      if (k === 0) { drawHeat(ctx, px, norm(r[card.method]), 8, size); return; }
      // Deletion in progress: the map's top-k patches flattened, the rest intact.
      const alpha = new Array(64).fill(1);
      r[`${card.method}_order`].slice(0, k).forEach((p) => (alpha[p] = 0));
      drawImage(ctx, px, size, alpha);
    };
    const drawCurve = (card, k) => {
      const w = Math.round(card.curve.getBoundingClientRect().width) || 360, H = 90;
      const ctx = card.curve.getContext("2d"), dpr = Math.min(devicePixelRatio || 1, 2);
      card.curve.width = w * dpr; card.curve.height = H * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const c = r[`${card.method}_drop`], X = (i) => (i / 64) * (w - 2) + 1, Y = (p) => 6 + (1 - p) * (H - 20);
      ctx.strokeStyle = "rgba(21,20,18,0.14)"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(0, Y(0)); ctx.lineTo(w, Y(0)); ctx.stroke();
      ctx.strokeStyle = "#151412"; ctx.lineWidth = 2; ctx.lineJoin = "round";
      ctx.beginPath(); c.slice(0, k + 1).forEach((p, i) => ctx[i ? "lineTo" : "moveTo"](X(i), Y(p))); ctx.stroke();
      ctx.fillStyle = "#8a857a"; ctx.font = "11px 'IBM Plex Mono', monospace";
      ctx.fillText(`${cls} ${pct(c[k])} · ${k} patches hidden`, 2, H - 2);
    };
    requestAnimationFrame(() => cards.forEach((c) => draw(c)));

    function choose(i) {
      cards.forEach((c) => (c.pickBtn.disabled = true));
      cards[i].pickBtn.classList.add("on");
      const t0 = performance.now(), dur = 3200;
      const step = (now) => {
        if (!S.alive) return;
        const k = Math.min(64, Math.round(((now - t0) / dur) * 64));
        cards.forEach((c) => { draw(c, Math.max(1, k)); drawCurve(c, k); });
        if (k < 64) S.anim = requestAnimationFrame(step);
        else reveal(i);
      };
      S.anim = requestAnimationFrame(step);
    }

    function reveal(i) {
      const a = cards.map((c) => auc(r[`${c.method}_drop`]));
      const win = a[0] <= a[1] ? 0 : 1, lose = 1 - win;
      const right = i === win;
      if (right) S.score++;
      cards.forEach((c, j) => {
        c.who.querySelector("span").textContent = `${NAMES[c.method]} — ${WHAT[c.method]}`;
        c.el.classList.toggle("winner", j === win);
        draw(c);
      });
      const hw = halfAt(r[`${cards[win].method}_drop`]), hl = halfAt(r[`${cards[lose].method}_drop`]);
      verdict.textContent = `${right ? "You trusted the right one." : "The other map told the truth."} Hiding ${"AB"[win]}'s hottest patches halved the model's confidence after ${patches(hw)}; ${"AB"[lose]}'s took ${hl >= 64 ? "all sixty-four" : patches(hl)}. ${cap(NAMES[cards[win].method])} won this round.`;
      next.hidden = false;
      next.focus();
    }
  }

  function finish() {
    const winPct = Math.round(L.occlusion_wins * 100);
    stage.replaceChildren(h("div", { class: "final" },
      h("p", { class: "eyebrow" }, h("b", {}, "III"), " · Plausible Lies"),
      h("h1", { class: "display" }, `${cap(word(S.score))} of ${word(S.rounds.length)}.`),
      h("div", { class: "wall" },
        h("p", {}, `Across ${L.pool} photographs the model classifies correctly, the occlusion map beat Grad-CAM's in the deletion test ${winPct}% of the time. Mean confidence while deleting (lower means the map found what mattered): ${pct(L.mean_auc.occlusion)} for occlusion, ${pct(L.mean_auc.gradcam)} for Grad-CAM, ${pct(L.mean_auc.random)} for patches deleted in random order.`),
        h("p", {}, "Grad-CAM is the most familiar heatmap in computer vision, and its maps look just as reasonable. For this model they mostly aren't about the prediction. A picture of an explanation is not an explanation until something tests it."),
        h("p", { class: "credit" }, "after ", h("a", { href: "https://distill.pub/2020/communicating-with-interactive-articles/" }, "Distill, Communicating with Interactive Articles"), " · deletion test after Petsiuk et al., RISE (2018)")),
      h("div", { class: "actions", style: "margin-top:24px" },
        h("button", { class: "accent", onclick: () => { Object.assign(S, { round: 0, score: 0, rounds: shuffle([...L.rounds]).slice(0, ROUNDS) }); play(); } }, "Play again"),
        h("button", { onclick: () => (location.hash = "#/field") }, "Next: IV · Hallucination Field →"))));
  }

  play();
  return () => { S.alive = false; cancelAnimationFrame(S.anim); };
}
