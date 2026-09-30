import { loadModel } from "./model.js";

let model;
const ready = loadModel().then((m) => {
  model = m;
  postMessage({
    type: "ready",
    cfg: m.cfg,
    classes: m.classes,
  });
});

onmessage = async ({ data }) => {
  await ready;
  if (data.type === "forward") {
    postMessage({ type: "result", id: data.id, out: model.forward(data.pixels) });
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
  } else if (data.type === "curves") {
    curves(data.id, data.pixels, data.order, data.ks);
  }
};

// Occlusion sweep: flatten each patch in turn to the image's mean color and record the logits.
// A pool of workers splits the patches; a newer request supersedes a running job.
let sweepId = 0;
// Yield via MessageChannel rather than setTimeout, which hidden pages throttle heavily.
const channel = new MessageChannel();
const yieldToQueue = () => new Promise((r) => { channel.port1.onmessage = r; channel.port2.postMessage(0); });
// Replace the given patches with a flat color (the image's mean), as occlusion does.
function imageMean(pixels) {
  const { imageSize: N, channels: C } = model.cfg;
  const mean = new Array(C).fill(0);
  for (let i = 0; i < N * N; i++) for (let c = 0; c < C; c++) mean[c] += pixels[i * C + c] / (N * N);
  return mean;
}
function fillPatches(pixels, patches, mean) {
  const { grid, win, imageSize: N, channels: C } = model.cfg;
  const out = Uint8ClampedArray.from(pixels);
  for (const p of patches) {
    const r0 = Math.floor(p / grid) * win, c0 = (p % grid) * win;
    for (let i = 0; i < win; i++)
      for (let j = 0; j < win; j++)
        for (let c = 0; c < C; c++) out[((r0 + i) * N + c0 + j) * C + c] = mean[c];
  }
  return out;
}

async function occlude(id, pixels, patches) {
  sweepId = id;
  const nPatches = model.cfg.nPatches, nc = model.classes.length, mean = imageMean(pixels);
  const logits = new Float32Array(nPatches * nc);
  for (const [done, p] of patches.entries()) {
    if (sweepId !== id) return;
    logits.set(model.forward(fillPatches(pixels, [p], mean)).logits, p * nc);
    postMessage({ type: "occlusion-progress", id, done: done + 1 });
    await yieldToQueue(); // let a newer request in
  }
  postMessage({ type: "occlusion", id, logits }, [logits.buffer]);
}

// Keep-only / remove curves: for each k, the logits with only the top-k patches kept (the rest
// flattened to the mean) and with the top-k removed. `ks` is this worker's share of k values.
async function curves(id, pixels, order, ks) {
  sweepId = id;
  const nPatches = model.cfg.nPatches, nc = model.classes.length, mean = imageMean(pixels);
  const keep = new Float32Array((nPatches + 1) * nc), drop = new Float32Array((nPatches + 1) * nc);
  for (const k of ks) {
    if (sweepId !== id) return;
    keep.set(model.forward(fillPatches(pixels, order.slice(k), mean)).logits, k * nc);
    drop.set(model.forward(fillPatches(pixels, order.slice(0, k), mean)).logits, k * nc);
    await yieldToQueue();
  }
  postMessage({ type: "curves", id, keep, drop }, [keep.buffer, drop.buffer]);
}
