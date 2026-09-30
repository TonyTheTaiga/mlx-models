import { loadModel, rollout } from "./model.js";

let model;
const ready = loadModel().then((m) => {
  model = m;
  postMessage({
    type: "ready",
    cfg: m.cfg,
    classes: m.classes,
    rope: m.w.layers.map((L) => ({ posIdx: Array.from(L.posIdx), freq: Array.from(L.freq) })),
  });
});

onmessage = async ({ data }) => {
  await ready;
  if (data.type === "forward") {
    const out = model.forward(data.pixels);
    out.rollout = rollout(out.attn, model.cfg.nTokens);
    postMessage({ type: "result", id: data.id, out });
  } else if (data.type === "check") {
    // Compare against logits MLX produced for the same images at export time.
    let worst = 0;
    data.images.forEach((pixels, i) => {
      model.forward(pixels).logits.forEach((v, k) => {
        worst = Math.max(worst, Math.abs(v - data.logits[i][k]));
      });
    });
    postMessage({ type: "check", worst, n: data.images.length });
  }
};
