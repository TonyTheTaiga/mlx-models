// A pool of model workers: evaluate(images) splits a batch across them and resolves to logits.

const size = Math.max(1, Math.min(4, (navigator.hardwareConcurrency || 4) - 1));
const workers = Array.from({ length: size }, () => new Worker(new URL("./worker.js", import.meta.url), { type: "module" }));
const pending = new Map();
let nextId = 0;

export const meta = new Promise((resolve) => {
  workers[0].addEventListener("message", function onReady({ data }) {
    if (data.type === "ready") { workers[0].removeEventListener("message", onReady); resolve(data); }
  });
});

for (const w of workers)
  w.addEventListener("message", ({ data }) => {
    if (data.type !== "eval") return;
    const job = pending.get(data.id);
    if (!job) return;
    pending.delete(data.id);
    job(data.logits);
  });

export function evaluate(images) {
  const shares = workers.map(() => []);
  images.forEach((img, i) => shares[i % size].push(i));
  return Promise.all(workers.map((w, k) => new Promise((resolve) => {
    if (!shares[k].length) return resolve([]);
    const id = nextId++;
    pending.set(id, resolve);
    w.postMessage({ type: "eval", id, images: shares[k].map((i) => images[i]) });
  }))).then((parts) => {
    const out = new Array(images.length);
    parts.forEach((part, k) => part.forEach((l, j) => (out[shares[k][j]] = l)));
    return out;
  });
}

export function softmax(v) {
  const m = Math.max(...v), e = v.map((x) => Math.exp(x - m)), z = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / z);
}
