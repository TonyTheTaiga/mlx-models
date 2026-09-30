// A plain-JS port of networks/transformers/vit/model.py + the CIFAR Classifier head,
// instrumented to keep every residual state and attention map.

export async function loadModel(base = "assets") {
  const [manifest, buf] = await Promise.all([
    fetch(`${base}/weights.json`).then((r) => r.json()),
    fetch(`${base}/weights.bin`).then((r) => r.arrayBuffer()),
  ]);
  const all = new Float32Array(buf);
  const t = {};
  for (const [name, [offset, shape]] of Object.entries(manifest.tensors)) {
    const size = shape.reduce((a, b) => a * b, 1);
    t[name] = all.subarray(offset, offset + size);
  }
  const m = manifest.config.model;
  const d = manifest.config.dataset;
  const cfg = {
    win: m.win_size,
    layers: m.n_layers,
    dModel: m.d_model,
    heads: m.n_heads,
    headDim: m.d_model / m.n_heads,
    imageSize: d.image_size,
    channels: d.channels,
    grid: d.image_size / m.win_size,
  };
  cfg.nPatches = cfg.grid * cfg.grid;
  cfg.nTokens = cfg.nPatches + 1;
  const layers = [];
  for (let i = 0; i < cfg.layers; i++) {
    const p = `vit.encoder_stack.layers.${i}.`;
    layers.push({
      ln1w: t[p + "pn1.weight"], ln1b: t[p + "pn1.bias"],
      ln2w: t[p + "pn2.weight"], ln2b: t[p + "pn2.bias"],
      qkv: t[p + "proj.weight"],
      out: t[p + "atten_proj.weight"],
      ff1: t[p + "feed_forward.layers.0.weight"],
      ff2: t[p + "feed_forward.layers.2.weight"],
      freq: t[p + "roper.freq"],
      posIdx: t[p + "roper.pos_idx"],
    });
  }
  return new Vit(cfg, {
    embed: t["vit.proj.weight"],
    cls: t["class_token.weight"],
    headW: t["classifier.weight"],
    headB: t["classifier.bias"],
    layers,
  }, manifest.classes);
}

// y[T, out] = x[T, in] @ W[out, in]^T
function linear(x, T, inDim, W, outDim, bias) {
  const y = new Float32Array(T * outDim);
  for (let t = 0; t < T; t++) {
    const xo = t * inDim;
    for (let o = 0; o < outDim; o++) {
      const wo = o * inDim;
      let s = bias ? bias[o] : 0;
      for (let i = 0; i < inDim; i++) s += x[xo + i] * W[wo + i];
      y[t * outDim + o] = s;
    }
  }
  return y;
}

function layerNorm(x, T, D, w, b, eps = 1e-5) {
  const y = new Float32Array(T * D);
  for (let t = 0; t < T; t++) {
    const o = t * D;
    let mean = 0;
    for (let i = 0; i < D; i++) mean += x[o + i];
    mean /= D;
    let v = 0;
    for (let i = 0; i < D; i++) v += (x[o + i] - mean) ** 2;
    const inv = 1 / Math.sqrt(v / D + eps);
    for (let i = 0; i < D; i++) y[o + i] = (x[o + i] - mean) * inv * w[i] + b[i];
  }
  return y;
}

// Abramowitz & Stegun 7.1.26, |err| < 1.5e-7 — close enough to MLX's exact GELU.
function erf(x) {
  const s = Math.sign(x);
  x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return s * y;
}

export function softmax(v) {
  let m = -Infinity;
  for (const x of v) m = Math.max(m, x);
  const e = Array.from(v, (x) => Math.exp(x - m));
  const z = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / z);
}

class Vit {
  constructor(cfg, w, classes) {
    this.cfg = cfg;
    this.w = w;
    this.classes = classes;
    // Rope2D: radians = [row * freq, col * freq] per patch. Both freq and pos_idx are
    // per-layer *learned* values in the checkpoint, not the arange/pow they were initialized to.
    const nFreq = cfg.headDim / 4;
    const half = cfg.headDim / 2;
    for (const L of w.layers) {
      L.cos = new Float32Array(cfg.nPatches * half);
      L.sin = new Float32Array(cfg.nPatches * half);
      for (let p = 0; p < cfg.nPatches; p++) {
        const row = L.posIdx[2 * p], col = L.posIdx[2 * p + 1];
        for (let i = 0; i < half; i++) {
          const r = (i < nFreq ? row : col) * L.freq[i % nFreq];
          L.cos[p * half + i] = Math.cos(r);
          L.sin[p * half + i] = Math.sin(r);
        }
      }
    }
  }

  // Rotate q or k in place; token 0 (CLS) is left alone, as in EncoderBlock.apply_rope.
  rope(x, offset, L) {
    const { nTokens, heads, headDim, dModel } = this.cfg;
    const half = headDim / 2;
    for (let t = 1; t < nTokens; t++) {
      const p = t - 1;
      for (let h = 0; h < heads; h++) {
        const o = t * 3 * dModel + offset + h * headDim;
        for (let i = 0; i < half; i++) {
          const c = L.cos[p * half + i], s = L.sin[p * half + i];
          const a = x[o + 2 * i], b = x[o + 2 * i + 1];
          x[o + 2 * i] = a * c - b * s;
          x[o + 2 * i + 1] = a * s + b * c;
        }
      }
    }
  }

  // pixels: Uint8 or float array of length H*W*C in HWC order, 0..255.
  patchify(pixels) {
    const { grid, win, channels, imageSize, nPatches } = this.cfg;
    const P = win * win * channels;
    const out = new Float32Array(nPatches * P);
    for (let r = 0; r < grid; r++)
      for (let c = 0; c < grid; c++)
        for (let i = 0; i < win; i++)
          for (let j = 0; j < win; j++)
            for (let ch = 0; ch < channels; ch++) {
              const src = ((r * win + i) * imageSize + (c * win + j)) * channels + ch;
              out[(r * grid + c) * P + (i * win + j) * channels + ch] = pixels[src] / 255;
            }
    return out;
  }

  forward(pixels) {
    const { nTokens: T, dModel: D, heads, headDim, win, channels, nPatches } = this.cfg;
    const w = this.w;
    const patchEmb = linear(this.patchify(pixels), nPatches, win * win * channels, w.embed, D);
    let x = new Float32Array(T * D);
    x.set(w.cls, 0);
    x.set(patchEmb, D);

    const states = [x.slice()];
    const mids = []; // each block's state after attention, before the MLP
    const attn = [];
    const scale = 1 / Math.sqrt(headDim);
    for (const L of w.layers) {
      const qkv = linear(layerNorm(x, T, D, L.ln1w, L.ln1b), T, D, L.qkv, 3 * D);
      this.rope(qkv, 0, L);
      this.rope(qkv, D, L);
      const mixed = new Float32Array(T * D);
      const layerAttn = [];
      for (let h = 0; h < heads; h++) {
        const A = new Float32Array(T * T);
        const ho = h * headDim;
        for (let i = 0; i < T; i++) {
          const qo = i * 3 * D + ho;
          let m = -Infinity;
          for (let j = 0; j < T; j++) {
            const ko = j * 3 * D + D + ho;
            let s = 0;
            for (let k = 0; k < headDim; k++) s += qkv[qo + k] * qkv[ko + k];
            s *= scale;
            A[i * T + j] = s;
            if (s > m) m = s;
          }
          let z = 0;
          for (let j = 0; j < T; j++) z += A[i * T + j] = Math.exp(A[i * T + j] - m);
          for (let j = 0; j < T; j++) {
            const a = (A[i * T + j] /= z);
            const vo = j * 3 * D + 2 * D + ho;
            const oo = i * D + ho;
            for (let k = 0; k < headDim; k++) mixed[oo + k] += a * qkv[vo + k];
          }
        }
        layerAttn.push(A);
      }
      attn.push(layerAttn);
      const proj = linear(mixed, T, D, L.out, D);
      for (let i = 0; i < x.length; i++) x[i] += proj[i];
      mids.push(x.slice());

      const hidden = linear(layerNorm(x, T, D, L.ln2w, L.ln2b), T, D, L.ff1, 4 * D);
      for (let i = 0; i < hidden.length; i++) {
        const v = hidden[i];
        hidden[i] = 0.5 * v * (1 + erf(v / Math.SQRT2));
      }
      const ff = linear(hidden, T, 4 * D, L.ff2, D);
      for (let i = 0; i < x.length; i++) x[i] += ff[i];
      states.push(x.slice());
    }

    // "Depth lens": the classifier head read out from the CLS token at every depth.
    const lens = states.map((s) => softmax(linear(s.subarray(0, D), 1, D, w.headW, 10, w.headB)));
    const logits = Array.from(linear(x.subarray(0, D), 1, D, w.headW, 10, w.headB));
    return { states, mids, attn, logits, probs: softmax(logits), lens };
  }
}

// Attention rollout (Abnar & Zuidema): mean over heads, add identity for the residual, chain.
export function rollout(attn, T) {
  let R = null;
  for (const layer of attn) {
    const A = new Float32Array(T * T);
    for (const H of layer) for (let i = 0; i < T * T; i++) A[i] += H[i] / layer.length;
    for (let i = 0; i < T; i++) {
      for (let j = 0; j < T; j++) A[i * T + j] *= 0.5;
      A[i * T + i] += 0.5;
    }
    if (!R) { R = A; continue; }
    const N = new Float32Array(T * T);
    for (let i = 0; i < T; i++)
      for (let k = 0; k < T; k++) {
        const a = A[i * T + k];
        if (a === 0) continue;
        for (let j = 0; j < T; j++) N[i * T + j] += a * R[k * T + j];
      }
    R = N;
  }
  return R;
}
