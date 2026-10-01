// VII. Mirror — your camera, at the model's resolution, named in its vocabulary.
//
// Each frame is center-cropped and averaged down to 32×32, the size of every photograph the model
// was trained on, and classified as fast as the worker pool allows. The name changes only once a
// new answer has held for a moment. Every couple of seconds an occlusion sweep (65 forward passes)
// refreshes the heat over what it's leaning on. Nothing leaves the browser.
import { h, cap, pct, article, hiDPI, drawHeat } from "../lib/ui.js";
import { evaluate, softmax } from "../lib/pool.js";

const HOLD = 4; // frames a new top class must hold before the name changes
const HEAT_EVERY = 2000;

export async function mount(root, { meta }) {
  const classes = meta.classes;
  const S = { alive: true, stream: null, source: null, probs: null, smooth: null, name: -1, pending: -1, held: 0, heat: null, lastHeat: 0, frames: 0, still: null };

  const video = h("video", { playsinline: true, muted: true, hidden: true, "aria-label": "your camera" });
  const small = h("canvas", { class: "small", role: "img", "aria-label": "what the model sees: 32 by 32 pixels" });
  const nameEl = h("p", { class: "name", "aria-live": "polite" }, "…");
  const odds = h("div", { class: "bars" });
  const note = h("p", { class: "note" });
  const start = h("button", { class: "accent", onclick: () => startCamera() }, "Turn on the camera");
  const upload = h("label", { class: "file" }, "or use a photo", h("input", { type: "file", accept: "image/*", onchange: (e) => usePhoto(e.target.files[0]) }));
  const intro = h("div", { class: "consent" },
    h("p", {}, "This room uses your camera. Frames are classified here, in your browser — nothing is recorded or sent anywhere."),
    h("div", { class: "actions" }, start, upload), note);

  root.append(h("section", { class: "mirror" },
    h("div", { class: "wall" },
      h("p", { class: "eyebrow" }, h("b", {}, "VII"), " · Mirror"),
      h("h1", { class: "display" }, "It only knows ", h("i", {}, "ten"), " words."),
      h("p", {}, "Airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck. Whatever you show it — a mug, your hand, your face — it must be one of these. Here is you, at the resolution of everything it has ever seen."),
      h("p", { class: "credit" }, "after ", h("a", { href: "https://history.siggraph.org/artwork/memo-akten-rebecca-fiebrink-mick-grierson-learning-to-see/" }, "Memo Akten, Learning to See"))),
    h("div", { class: "glass" }, video, small, intro),
    h("div", { class: "readout" }, h("p", { class: "eyebrow" }, "It sees"), nameEl, odds)));

  const grab = Object.assign(document.createElement("canvas"), { width: 32, height: 32 });
  function framePixels(src, w, hgt) {
    const s = Math.min(w, hgt), g = grab.getContext("2d", { willReadFrequently: true });
    g.imageSmoothingQuality = "high";
    // Mirror the camera, like a mirror; photos stay as they are.
    g.save();
    if (src === video) { g.translate(32, 0); g.scale(-1, 1); }
    g.drawImage(src, (w - s) / 2, (hgt - s) / 2, s, s, 0, 0, 32, 32);
    g.restore();
    const d = g.getImageData(0, 0, 32, 32).data, px = new Uint8ClampedArray(3072);
    for (let i = 0; i < 1024; i++) for (let k = 0; k < 3; k++) px[i * 3 + k] = d[i * 4 + k];
    return px;
  }

  async function startCamera() {
    try {
      S.stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: 640, height: 480 }, audio: false });
      video.srcObject = S.stream;
      await video.play();
      video.hidden = false;
      intro.hidden = true;
      S.source = "camera";
      loop();
    } catch (e) {
      note.textContent = `The camera isn't available (${e.name || "blocked"}). You can use a photo instead.`;
    }
  }
  function usePhoto(file) {
    if (!file) return;
    const img = new Image();
    img.onload = async () => {
      intro.hidden = true;
      S.source = "photo";
      S.still = framePixels(img, img.naturalWidth, img.naturalHeight);
      URL.revokeObjectURL(img.src);
      S.smooth = null; S.name = -1;
      await classify(S.still, true);
      await sweep(S.still);
    };
    img.src = URL.createObjectURL(file);
  }

  async function classify(px, instant = false) {
    const [l] = await evaluate([px]);
    const p = softmax(l);
    // Smooth over frames so the name doesn't flicker.
    S.smooth = S.smooth && !instant ? S.smooth.map((v, i) => 0.6 * v + 0.4 * p[i]) : p;
    const top = S.smooth.indexOf(Math.max(...S.smooth));
    if (instant || S.name < 0) S.name = top;
    else if (top !== S.name) { if (top === S.pending) { if (++S.held >= HOLD) { S.name = top; S.held = 0; } } else { S.pending = top; S.held = 1; } }
    else S.held = 0;
    show(px);
  }
  // Occlusion sweep: what hiding each patch costs the current answer.
  async function sweep(px) {
    const mean = [0, 1, 2].map((k) => { let s = 0; for (let i = 0; i < 1024; i++) s += px[i * 3 + k]; return Math.round(s / 1024); });
    const batch = [px];
    for (let p = 0; p < 64; p++) {
      const o = Uint8ClampedArray.from(px), r0 = Math.floor(p / 8) * 4, c0 = (p % 8) * 4;
      for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) o.set(mean, ((r0 + i) * 32 + c0 + j) * 3);
      batch.push(o);
    }
    const logits = await evaluate(batch), t = S.name;
    const lp = logits.map((l) => Math.log(Math.max(softmax(l)[t], 1e-12)));
    const drop = lp.slice(1).map((v) => Math.max(0, lp[0] - v)), m = Math.max(1e-9, ...drop);
    S.heat = drop.map((v) => v / m);
    S.lastHeat = performance.now();
    show(px);
  }

  function show(px) {
    const size = Math.round(small.getBoundingClientRect().width) || 360, ctx = hiDPI(small, size);
    if (S.heat) drawHeat(ctx, px, S.heat, 8, size);
    else {
      const img = ctx.createImageData(32, 32);
      for (let i = 0; i < 1024; i++) { for (let k = 0; k < 3; k++) img.data[i * 4 + k] = px[i * 3 + k]; img.data[i * 4 + 3] = 255; }
      grab.getContext("2d").putImageData(img, 0, 0);
      ctx.imageSmoothingEnabled = false; ctx.drawImage(grab, 0, 0, size, size);
    }
    const c = classes[S.name];
    nameEl.textContent = `${cap(article(c))} ${c}.`;
    const order = [...S.smooth.keys()].sort((a, b) => S.smooth[b] - S.smooth[a]).slice(0, 5);
    odds.replaceChildren(...order.map((k) => h("div", { class: `row ${k === S.name ? "target" : ""}` },
      h("span", {}, classes[k]), h("span", { class: "track" }, h("span", { class: "fill", style: `display:block;width:${(S.smooth[k] * 100).toFixed(1)}%` })), h("span", {}, pct(S.smooth[k])))));
  }

  async function loop() {
    while (S.alive && S.source === "camera") {
      if (document.hidden || video.readyState < 2) { await new Promise((r) => setTimeout(r, 200)); continue; }
      const px = framePixels(video, video.videoWidth, video.videoHeight);
      await classify(px);
      if (performance.now() - S.lastHeat > HEAT_EVERY) await sweep(px);
    }
  }

  return () => {
    S.alive = false;
    if (S.stream) S.stream.getTracks().forEach((t) => t.stop()); // release the camera
  };
}
