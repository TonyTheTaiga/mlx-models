// II. Enough — photographs reduced to the patches the prediction needed (computed in MLX).
//
// "Enough" is the smallest k such that keeping only the k most important patches (by occlusion),
// with the rest flattened to the image's mean color, still gives the same answer at ≥80% of the
// original confidence. k = 0 means the flat mean color alone already did.
import { h, word, cap, pct, article, loadImages, imageAt, drawImage, hiDPI } from "../lib/ui.js";

export async function mount(root) {
  const [E, imgs] = await Promise.all([fetch("assets/enough.json").then((r) => r.json()), loadImages("assets/enough.bin")]);
  const C = E.classes;
  const works = [];

  // A work: the image with only its needed patches, the rest arriving as you scroll past.
  function work(it, size = "big") {
    const c = h("canvas", { role: "img" });
    const n = it.enough, cls = C[it.pred], truth = C[it.label];
    const lead = n === 0 ? "None." : `${cap(word(n))}.`;
    let line;
    if (it.room === "wrong") line = n === 0
      ? `Its average color was enough to call this ${truth} ${article(cls)} ${cls}.`
      : `${cap(word(n))} ${n === 1 ? "patch" : "patches"} of sixty-four were enough to call this ${truth} ${article(cls)} ${cls}.`;
    else line = n === 0
      ? `Its average color was enough to call this ${article(cls)} ${cls}.`
      : `${n === 1 ? "One patch" : `${cap(word(n))} patches`} of sixty-four were enough to call this ${article(cls)} ${cls}.`;
    c.setAttribute("aria-label", `${line} The photograph is of ${article(truth)} ${truth}.`);
    const facts = h("div", { class: "facts" },
      h("span", {}, n === 0 ? "a flat field of its average color: " : `with ${word(n)} ${n === 1 ? "patch" : "patches"}: `, h("b", {}, `${pct(it.keep[n])} ${cls}`)),
      h("span", {}, "with all sixty-four: ", h("b", {}, `${pct(it.confidence)} ${cls}`)),
      it.room === "wrong" ? h("span", {}, "it is ", h("b", {}, `${article(truth)} ${truth}`)) : null);
    const el = h("article", { class: `work ${size}` },
      h("div", { class: "fig" }, c),
      h("div", {}, h("p", { class: `word ${n === 0 ? "none" : ""}` }, lead), h("p", { class: "line" }, line), facts));
    works.push({ el, c, it, reveal: -1 });
    return el;
  }

  // The index: every curated image as a small mosaic of what it needed.
  function index(items) {
    return h("div", { class: "index" }, items.map((it) => {
      const c = h("canvas", { role: "img", "aria-label": `${C[it.pred]}: ${it.enough} of 64 patches` });
      const alpha = new Array(64).fill(0);
      it.order.slice(0, it.enough).forEach((p) => (alpha[p] = 1));
      requestAnimationFrame(() => drawImage(hiDPI(c, 132), imageAt(imgs, it.slot), 132, alpha));
      return h("figure", {}, c, h("figcaption", {}, h("b", {}, it.enough === 0 ? "none" : word(it.enough)), ` · ${C[it.pred]}`));
    }));
  }

  const some = E.items.filter((x) => x.room === "some");
  const best = C.map((_, c) => some.find((x) => x.pred === c)).filter(Boolean).sort((a, b) => a.enough - b.enough);
  const none = E.items.filter((x) => x.room === "none");
  const wrong = E.items.filter((x) => x.room === "wrong");

  root.append(...[
    h("section", { class: "enough-intro" },
      h("p", { class: "eyebrow" }, h("b", {}, "II"), " · Enough"),
      h("h1", { class: "display" }, "How much of a photograph ", h("i", {}, "does it use?")),
      h("div", { class: "wall", style: "margin-top:22px" },
        h("p", {}, `Each image below is shown with only the patches the model needed — ranked by how much hiding each one hurts, and kept until the answer holds at 80% of its original confidence. Everything else is flattened to the image's average color. Scroll, and the rest of the photograph comes back.`),
        h("p", { class: "credit" }, "after ", h("a", { href: "https://annaridler.com/works/myriad-tulips" }, "Anna Ridler, Myriad (Tulips)"), " · computed in MLX over ", `${E.pool.toLocaleString()} test photographs`)),
      h("div", { class: "stats" },
        h("div", { class: "stat" }, h("div", { class: "v" }, cap(word(Math.round(E.median_enough_correct)))), h("div", { class: "l" }, "patches of sixty-four: the median for a correct answer")),
        h("div", { class: "stat" }, h("div", { class: "v" }, pct(E.color_alone_share)), h("div", { class: "l" }, "of correct answers needed no patch at all — the average color did it")),
        h("div", { class: "stat" }, h("div", { class: "v" }, pct(E.color_alone_share_mistakes)), h("div", { class: "l" }, `of ${E.confident_mistakes} confident mistakes were the same: color alone`)))),
    h("header", { class: "section-head" }, h("h2", {}, "Some"), h("p", { class: "wall" }, "One photograph per class: the one that needed the least.")),
    best.map((it) => work(it)),
    h("header", { class: "section-head" }, h("h2", {}, "Index"), h("p", { class: "wall" }, "The three leanest correct answers in every class, as the model needed them.")),
    index(some),
    h("header", { class: "section-head" }, h("h2", {}, "None"), h("p", { class: "wall" }, "Photographs whose flat average color — no shape, no texture, not one patch — already got the right answer.")),
    none.map((it) => work(it)),
    h("header", { class: "section-head" }, h("h2", {}, "Wrong"), h("p", { class: "wall" }, `Confident mistakes that needed the least, one of each kind. These are the extreme end: across all ${E.confident_mistakes} confident mistakes, ${pct(E.color_alone_share_mistakes)} were made on color alone.`)),
    wrong.map((it) => work(it)),
    h("footer", { class: "section-head" }, h("p", { class: "wall" }, h("a", { href: "#/lies" }, "Next: III · Plausible Lies →"))),
  ].flat()); // append() doesn't flatten the lists of works

  // Scroll-linked reveal: 0 = only the needed patches, 1 = the whole photograph.
  let raf = 0;
  function update() {
    raf = 0;
    const vh = innerHeight;
    for (const w of works) {
      const r = w.c.getBoundingClientRect();
      if (r.bottom < -vh || r.top > 2 * vh) continue;
      // Hold "only what it needed" while the work is centered; reveal as it scrolls away.
      const centeredTop = (vh - r.height) / 2;
      const t = Math.max(0, Math.min(1, (centeredTop - 40 - r.top) / (r.height * 0.6)));
      const q = Math.round(t * 40) / 40;
      const size = Math.round(r.width);
      if (q === w.reveal && size === w.size) continue;
      w.reveal = q; w.size = size;
      const alpha = new Array(64).fill(q);
      w.it.order.slice(0, w.it.enough).forEach((p) => (alpha[p] = 1));
      drawImage(hiDPI(w.c, size), imageAt(imgs, w.it.slot), size, alpha);
    }
  }
  const onScroll = () => { if (!raf) raf = requestAnimationFrame(update); };
  addEventListener("scroll", onScroll, { passive: true });
  addEventListener("resize", onScroll);
  requestAnimationFrame(update);
  return () => { removeEventListener("scroll", onScroll); removeEventListener("resize", onScroll); cancelAnimationFrame(raf); };
}
