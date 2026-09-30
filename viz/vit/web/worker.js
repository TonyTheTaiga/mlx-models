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
  } else if (data.type === "occlude") {
    occlude(data.id, data.pixels, data.patches);
  }
};

// Occlusion sweep: replace each patch in turn with the image's mean color and record how far
// every token at every depth moves (‖Δx‖). One sweep answers "which image patches shaped this
// token?" for all targets at once. A newer request supersedes a running sweep.
let sweepId = 0;
// Yield via MessageChannel rather than setTimeout, which hidden pages throttle heavily.
const channel = new MessageChannel();
const yieldToQueue = () => new Promise((r) => { channel.port1.onmessage = r; channel.port2.postMessage(0); });
async function occlude(id, pixels, patches) {
  sweepId = id;
  const { grid, win, nPatches, nTokens: T, dModel: D, layers, imageSize: N, channels: C } = model.cfg;
  const mean = new Array(C).fill(0);
  for (let i = 0; i < N * N; i++) for (let c = 0; c < C; c++) mean[c] += pixels[i * C + c] / (N * N);
  const base = model.forward(pixels).states;
  const depths = layers + 1;
  const out = new Float32Array(nPatches * depths * T);
  // A pool of workers splits the sweep; each fills in only its own patches.
  const mine = patches ?? [...Array(nPatches).keys()];
  for (const [n, p] of mine.entries()) {
    if (sweepId !== id) return;
    const occ = Uint8ClampedArray.from(pixels);
    const r0 = Math.floor(p / grid) * win, c0 = (p % grid) * win;
    for (let i = 0; i < win; i++)
      for (let j = 0; j < win; j++)
        for (let c = 0; c < C; c++) occ[((r0 + i) * N + c0 + j) * C + c] = mean[c];
    const states = model.forward(occ).states;
    for (let d = 0; d < depths; d++)
      for (let t = 0; t < T; t++) {
        let n = 0;
        for (let k = 0; k < D; k++) n += (states[d][t * D + k] - base[d][t * D + k]) ** 2;
        out[(p * depths + d) * T + t] = Math.sqrt(n);
      }
    postMessage({ type: "occlusion-progress", id, done: n + 1 });
    await yieldToQueue(); // let a newer request in
  }
  postMessage({ type: "occlusion", id, data: out, depths }, [out.buffer]);
}
