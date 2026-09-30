// Evaluates batches of 32×32 RGB images with the in-browser ViT and returns their logits.
import { loadModel } from "./model.js";

const ready = loadModel("../assets").then((m) => {
  postMessage({ type: "ready", cfg: m.cfg, classes: m.classes });
  return m;
});

onmessage = async ({ data }) => {
  const model = await ready;
  if (data.type !== "eval") return;
  const logits = data.images.map((px) => model.forward(px).logits);
  postMessage({ type: "eval", id: data.id, logits });
};
