// GAZE — four studies in machine sight. A hash router over four rooms plus the catalog.
import { h, loadImages, imageAt, drawImage, drawHeat, hiDPI } from "./lib/ui.js";
import { randomGenome, render } from "./lib/genome.js";
import { meta } from "./lib/pool.js";

const ROOMS = {
  engine: () => import("./rooms/engine.js"),
  enough: () => import("./rooms/enough.js"),
  lies: () => import("./rooms/lies.js"),
  field: () => import("./rooms/field.js"),
};
const root = document.getElementById("room");
let unmount = null;

async function route() {
  const name = location.hash.replace(/^#\/?/, "") || "";
  if (unmount) { unmount(); unmount = null; }
  root.replaceChildren();
  document.body.classList.toggle("dark", name === "field");
  document.querySelectorAll(".bar nav a").forEach((a) => a.classList.toggle("on", a.dataset.room === name));
  window.scrollTo(0, 0);
  if (ROOMS[name]) {
    const mod = await ROOMS[name]();
    unmount = await mod.mount(root, { meta: await meta });
  } else unmount = await landing(root);
  root.focus({ preventScroll: true });
}
addEventListener("hashchange", route);
route();

// ---------- the catalog ----------
async function landing(el) {
  const m = await meta;
  const [enough, enoughImgs, lies, liesImgs] = await Promise.all([
    fetch("assets/enough.json").then((r) => r.json()),
    loadImages("assets/enough.bin"),
    fetch("assets/lies.json").then((r) => r.json()),
    loadImages("assets/lies.bin"),
  ]);
  const previews = {
    engine: (c) => { const ctx = hiDPI(c, 240); render(ctx, randomGenome("engine"), 240); },
    enough: (c) => {
      const it = enough.items.find((x) => x.room === "some" && x.enough > 1) || enough.items[0];
      const alpha = new Array(64).fill(0);
      it.order.slice(0, it.enough).forEach((p) => (alpha[p] = 1));
      drawImage(hiDPI(c, 240), imageAt(enoughImgs, it.slot), 240, alpha);
    },
    lies: (c) => {
      const r = lies.rounds[0], ctx = hiDPI(c, 240), norm = (v) => { const mx = Math.max(1e-9, ...v); return v.map((x) => x / mx); };
      ctx.save(); ctx.beginPath(); ctx.rect(0, 0, 120, 240); ctx.clip();
      drawHeat(ctx, imageAt(liesImgs, r.slot), norm(r.occlusion), 8, 240); ctx.restore();
      ctx.save(); ctx.beginPath(); ctx.rect(120, 0, 120, 240); ctx.clip();
      drawHeat(ctx, imageAt(liesImgs, r.slot), norm(r.gradcam), 8, 240); ctx.restore();
      ctx.fillStyle = "#f1ede4"; ctx.fillRect(119, 0, 2, 240);
    },
    field: (c) => {
      const ctx = hiDPI(c, 240), g = randomGenome("field");
      let raf = 0, t0 = performance.now();
      const tick = (now) => {
        const t = (now - t0) / 1000;
        g.blobs.forEach((b, i) => { b.x = 0.5 + 0.35 * Math.sin(t * 0.13 * (i + 1) + i); b.y = 0.5 + 0.35 * Math.cos(t * 0.11 * (i + 2) + i * 2); });
        ctx.filter = "blur(6px) saturate(1.4)"; render(ctx, g, 240); ctx.filter = "none";
        raf = requestAnimationFrame(tick);
      };
      raf = requestAnimationFrame(tick);
      stops.push(() => cancelAnimationFrame(raf));
    },
  };
  const stops = [];
  const entry = (key, num, title, text, after) => {
    const c = h("canvas", { "aria-hidden": "true" });
    const a = h("a", { class: "entry", href: `#/${key}` },
      h("div", { class: "num" }, num),
      h("div", {}, h("h2", {}, title), h("p", {}, text), h("p", { class: "credit" }, after)),
      c);
    previews[key](c);
    return a;
  };
  const c = m.cfg;
  el.append(
    h("section", { class: "hero" },
      h("p", { class: "eyebrow" }, "An exhibition in four rooms"),
      h("h1", {}, "GAZE"),
      h("p", { class: "lede" }, "Four studies in how a small machine sees — and what it doesn't need to see."),
      h("p", { class: "model" }, `A vision transformer · ${c.layers} blocks × ${c.heads} heads · trained on 32×32 photographs from CIFAR-10 · running live in your browser`)),
    h("section", { class: "catalog" },
      entry("engine", "I", "Perception Engine", "An evolutionary loop composes with eight inks and a handful of strokes until the model is certain it's looking at a ship — or a cat, or a frog. Intervene with your own strokes; print what it made.", "after Tom White, Perception Engines (2018), and Brown et al., Adversarial Patch (2017)"),
      entry("enough", "II", "Enough", `Test photographs reduced to only the patches the model needed. For correctly classified images the median is ${enough.median_enough_correct} of 64; for ${Math.round(enough.color_alone_share * 100)}%, the average color alone was enough.`, "after Anna Ridler, Myriad (Tulips) (2018)"),
      entry("lies", "III", "Plausible Lies", `Two maps claim to show where the model looked. One of them usually lies. Guess which — then a deletion test settles it. Over ${lies.pool} images, one method wins ${Math.round(lies.occlusion_wins * 100)}% of the time.`, "after Distill, Communicating with Interactive Articles (2020)"),
      entry("field", "IV", "Hallucination Field", "A darkened room. Six images drift between the classes the model knows, each evolving toward whatever it is currently asked to see.", "after Trevor Paglen, Adversarially Evolved Hallucinations (2017–), and Refik Anadol, Machine Hallucinations (2016–)")),
    h("section", { class: "colophon" },
      h("div", {}, h("h3", {}, "The model"),
        h("p", {}, `A ${c.layers}-block, ${c.heads}-head vision transformer (d = ${c.dModel}, 2D rotary positions) trained in MLX on CIFAR-10's 32×32 photographs. The rooms that evolve images run it here, in your browser; the others were computed in MLX over 1,000 test images.`)),
      h("div", {}, h("h3", {}, "Honest measures"),
        h("p", {}, "“Hiding” a patch means flattening it to the image's average color. The rooms state their numbers as measured — including when the answer is less dramatic than the art.")),
      h("div", {}, h("h3", {}, "References"),
        h("ul", {},
          h("li", {}, h("a", { href: "https://aiartists.org/tom-white" }, "Tom White — Perception Engines")),
          h("li", {}, h("a", { href: "https://arxiv.org/abs/1712.09665" }, "Brown et al. — Adversarial Patch")),
          h("li", {}, h("a", { href: "https://annaridler.com/works/myriad-tulips" }, "Anna Ridler — Myriad (Tulips)")),
          h("li", {}, h("a", { href: "https://distill.pub/2020/communicating-with-interactive-articles/" }, "Distill — Communicating with Interactive Articles")),
          h("li", {}, h("a", { href: "https://history.siggraph.org/artwork/memo-akten-rebecca-fiebrink-mick-grierson-learning-to-see/" }, "Memo Akten — Learning to See")),
          h("li", {}, h("a", { href: "https://citylights.com/art/trevor-paglen-adversarially-hallucinat/" }, "Trevor Paglen — Adversarially Evolved Hallucinations")),
          h("li", {}, h("a", { href: "https://refikanadol.com/works/unsupervised/" }, "Refik Anadol — Machine Hallucinations"))))));
  return () => stops.forEach((f) => f());
}
