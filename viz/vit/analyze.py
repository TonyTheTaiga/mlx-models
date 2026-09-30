"""
Model-builder analysis of the CIFAR-10 ViT, computed in MLX over the full test set.

Writes viz/vit/web/assets/analysis.json (+ examples.bin), which the Lumen page renders.

Shortcut learning
  - counterfactual tests: accuracy per class when the image is changed in a controlled way
    (border only, center only, grayscale, patches shuffled, mirrored)
  - occlusion evidence per class: average evidence map and the share that falls on the border
  - the most border-reliant correctly classified images, for drill-down

Internals health
  - class read-out after every block: the trained head applied to CLS, and linear probes on CLS
    and on the mean patch token
  - knock-outs: accuracy with each block skipped and with each attention head zeroed
  - attention statistics per head: entropy, spatial reach, attention to CLS
  - residual stream: norm per block and the relative size of each attention / MLP update

Usage: uv run python viz/vit/analyze.py [--checkpoint ...] [--evidence-per-class 200]
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import yaml

from training.vit.main import Classifier, load_checkpoint

ROOT = Path(__file__).parents[2]
OUT = Path(__file__).parent / "web" / "assets"


def fingerprint(path: Path) -> str:
    """Short content hash of a checkpoint, so the page can tell if its files disagree."""
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def load_images(split_dir: Path, per_class: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    images, labels = [], []
    for label in sorted(int(d.name) for d in split_dir.iterdir() if d.name.isdigit()):
        paths = sorted((split_dir / str(label)).glob("*.png"))[:per_class]
        for path in paths:
            images.append(cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
            labels.append(label)
    return np.stack(images), np.array(labels)


class Probe:
    """Instrumented forward pass mirroring EncoderBlock, with knock-outs and statistics."""

    def __init__(self, model: Classifier):
        self.model = model
        self.blocks = model.vit.encoder_stack.layers
        self.grid = int(round(model.vit.encoder_stack.layers[0].roper.pos_idx.shape[0] ** 0.5))

    def __call__(self, images, skip_block=None, zero_head=None, collect=False):
        m = self.model
        patches = m.vit.preprocess(images)
        cls = m.class_token(mx.zeros((images.shape[0], 1), dtype=mx.int32))
        x = mx.concatenate([cls, m.vit.proj(patches)], axis=1)
        cls_states, patch_means, stats = [x[:, 0]], [x[:, 1:].mean(1)], []
        for i, blk in enumerate(self.blocks):
            if skip_block == i:
                cls_states.append(x[:, 0])
                patch_means.append(x[:, 1:].mean(1))
                continue
            q, k, v = mx.split(blk.proj(blk.pn1(x)), 3, axis=-1)
            q, k, v = blk.apply_rope(blk.reshape(q)), blk.apply_rope(blk.reshape(k)), blk.reshape(v)
            scale = (blk.d_model // blk.n_heads) ** -0.5
            attn = mx.softmax((q @ k.transpose(0, 1, 3, 2)) * scale, axis=-1)  # (B, H, T, T)
            heads_out = attn @ v
            if zero_head is not None and zero_head[0] == i:
                mask = mx.ones((1, blk.n_heads, 1, 1)).at[0, zero_head[1]].add(-1.0)
                heads_out = heads_out * mask
            attn_out = blk.atten_proj(heads_out.transpose(0, 2, 1, 3).reshape(*x.shape))
            mid = x + attn_out
            mlp_out = blk.feed_forward(blk.pn2(mid))
            if collect:
                stats.append(self._stats(attn, x, attn_out, mid, mlp_out))
            x = mid + mlp_out
            cls_states.append(x[:, 0])
            patch_means.append(x[:, 1:].mean(1))
        return m.classifier(x[:, 0]), cls_states, patch_means, stats

    def _stats(self, attn, x_in, attn_out, mid, mlp_out):
        B, H, T, _ = attn.shape
        g = self.grid
        # Normalized entropy of each query's attention row (1 = uniform over all tokens).
        ent = -(attn * mx.log(attn + 1e-12)).sum(-1) / np.log(T)  # (B, H, T)
        # Spatial reach: expected distance (in patches) from a patch query to the patches it
        # attends to, renormalized over patch keys.
        pos = mx.array([[r, c] for r in range(g) for c in range(g)], dtype=mx.float32)
        dist = mx.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1))  # (P, P)
        pp = attn[:, :, 1:, 1:]
        pp = pp / (pp.sum(-1, keepdims=True) + 1e-12)
        reach = (pp * dist).sum(-1)  # (B, H, P)
        norm = lambda a: mx.sqrt((a**2).sum(-1))  # noqa: E731
        return {
            "entropy": ent.sum((0, 2)),  # summed; averaged by the caller
            "reach": reach.sum((0, 2)),
            "to_cls": attn[:, :, 1:, 0].sum((0, 2)),  # patch queries' attention to CLS
            "cls_self": attn[:, :, 0, 0].sum(0),
            "count_q": B * T,
            "count_p": B * (T - 1),
            "count_b": B,
            # Summed norms; ratios are taken of the sums so tiny-norm tokens can't blow them up.
            "x_norm": norm(x_in).sum(),
            "attn_norm": norm(attn_out).sum(),
            "mid_norm": norm(mid).sum(),
            "mlp_norm": norm(mlp_out).sum(),
            "tokens": B * T,
        }


def batches(n, size):
    for s in range(0, n, size):
        yield slice(s, min(n, s + size))


def to_input(images: np.ndarray) -> mx.array:
    return mx.array(images.astype(np.float32) / 255.0)


def predict(probe, images, labels, bs=500, **kw):
    preds = []
    for sl in batches(len(images), bs):
        logits = probe(to_input(images[sl]), **kw)[0]
        preds.append(np.array(mx.argmax(logits, -1)))
    return np.concatenate(preds)


def per_class_acc(preds, labels, n_classes):
    return [float((preds[labels == c] == c).mean()) for c in range(n_classes)]


def false_alarms(preds, labels, n_classes):
    """P(predict c | true class isn't c): a class the model falls back on scores well by default."""
    return [float((preds[labels != c] == c).mean()) for c in range(n_classes)]


# ---------- counterfactual images ----------
def ring_mask(grid, win, border=True):
    """Pixel mask (H, W) of the outer ring of patches (or its complement)."""
    m = np.zeros((grid, grid), bool)
    m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = True
    if not border:
        m = ~m
    return np.kron(m, np.ones((win, win), bool))


def flatten_where(images, mask):
    """Replace masked pixels with each image's mean color, as the occlusion page does."""
    out = images.copy()
    mean = images.reshape(len(images), -1, 3).mean(1).round().astype(np.uint8)
    out[:, mask] = mean[:, None, :]
    return out


def shuffle_patches(images, grid, win, seed=0):
    rng = np.random.default_rng(seed)
    n = len(images)
    p = images.reshape(n, grid, win, grid, win, 3).transpose(0, 1, 3, 2, 4, 5).reshape(n, grid * grid, win, win, 3)
    for i in range(n):
        p[i] = p[i][rng.permutation(grid * grid)]
    return p.reshape(n, grid, grid, win, win, 3).transpose(0, 1, 3, 2, 4, 5).reshape(n, grid * win, grid * win, 3)


# ---------- occlusion evidence ----------
def occlusion_scores(probe, images, grid, win, bs_images=8):
    """Δ log p(predicted class) for flattening each patch, per image: (N, P), plus preds."""
    n, P = len(images), grid * grid
    scores, preds = np.zeros((n, P), np.float32), np.zeros(n, int)
    for sl in batches(n, bs_images):
        imgs = images[sl]
        k = len(imgs)
        base = nn.log_softmax(probe(to_input(imgs))[0], -1)
        pred = np.array(mx.argmax(base, -1))
        mean = imgs.reshape(k, -1, 3).mean(1).round().astype(np.uint8)
        occ = np.repeat(imgs[:, None], P, axis=1)  # (k, P, H, W, 3)
        for p in range(P):
            r, c = divmod(p, grid)
            occ[:, p, r * win : (r + 1) * win, c * win : (c + 1) * win] = mean[:, None, None, :]
        lp = nn.log_softmax(probe(to_input(occ.reshape(k * P, *imgs.shape[1:])))[0], -1)
        lp = np.array(lp).reshape(k, P, -1)
        b = np.array(base)
        for j in range(k):
            scores[sl.start + j] = b[j, pred[j]] - lp[j, :, pred[j]]
        preds[sl] = pred
    return scores, preds


# ---------- linear probes ----------
def fit_probe(train_x, train_y, test_x, test_y, n_classes, epochs=400):
    mu, sd = train_x.mean(0), train_x.std(0) + 1e-6
    xtr, xte = mx.array((train_x - mu) / sd), mx.array((test_x - mu) / sd)
    ytr = mx.array(train_y)
    head = nn.Linear(train_x.shape[1], n_classes)
    opt = optim.Adam(learning_rate=1e-2)

    def loss(h):
        return nn.losses.cross_entropy(h(xtr), ytr, reduction="mean") + 1e-4 * (h.weight**2).sum()

    step = nn.value_and_grad(head, loss)
    for _ in range(epochs):
        _, g = step(head)
        opt.update(head, g)
        mx.eval(head.parameters(), opt.state)
    return float((np.array(mx.argmax(head(xte), -1)) == test_y).mean())


def collect_features(probe, images, bs=500):
    cls_feats, patch_feats = None, None
    for sl in batches(len(images), bs):
        _, cls_states, patch_means, _ = probe(to_input(images[sl]))
        c = [np.array(s) for s in cls_states]
        p = [np.array(s) for s in patch_means]
        cls_feats = c if cls_feats is None else [np.concatenate([a, b]) for a, b in zip(cls_feats, c)]
        patch_feats = p if patch_feats is None else [np.concatenate([a, b]) for a, b in zip(patch_feats, p)]
    return cls_feats, patch_feats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("training/vit/checkpoints/cifar10"))
    parser.add_argument("--weights", default="best.npz")
    parser.add_argument("--evidence-per-class", type=int, default=200)
    parser.add_argument("--probe-train-per-class", type=int, default=1000)
    args = parser.parse_args()
    t0 = time.time()
    log = lambda msg: print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)  # noqa: E731

    ckpt = ROOT / args.checkpoint
    config = yaml.safe_load(open(ckpt / "config.yaml"))
    dcfg, mcfg = config["dataset"], config["model"]
    data_root = ROOT / dcfg["path"]
    classes = (data_root / "classes.txt").read_text().split()
    nc, win = len(classes), mcfg["win_size"]
    grid = dcfg["image_size"] // win

    model = Classifier(num_classes=nc, image_size=dcfg["image_size"], channels=dcfg["channels"], **mcfg)
    load_checkpoint(model, ckpt / args.weights, learned_rope=True)
    model.eval()
    probe = Probe(model)

    test_x, test_y = load_images(data_root / "test")
    log(f"loaded {len(test_x)} test images")

    # The instrumented pass must reproduce the model exactly.
    check = to_input(test_x[:64])
    assert mx.allclose(probe(check)[0], model(check), atol=1e-4).item(), "instrumented forward diverges"

    # ----- accuracy and counterfactual tests -----
    base_pred = predict(probe, test_x, test_y)
    result = {
        "checkpoint": str(args.checkpoint / args.weights),
        "fingerprint": fingerprint(ckpt / args.weights),
        "config": mcfg,
        "classes": classes,
        "n_test": int(len(test_x)),
        "accuracy": {"overall": float((base_pred == test_y).mean()), "per_class": per_class_acc(base_pred, test_y, nc)},
    }
    result["accuracy"]["false_alarm"] = false_alarms(base_pred, test_y, nc)
    conf = np.zeros((nc, nc), int)
    np.add.at(conf, (test_y, base_pred), 1)
    result["confusion"] = conf.tolist()
    log(f"accuracy {result['accuracy']['overall']:.1%}")

    tests = {
        "border_only": ("Border only", "the center 6×6 patches flattened to the image's mean color; only the outer ring of patches is left", flatten_where(test_x, ring_mask(grid, win, border=False))),
        "center_only": ("Center only", "the outer ring of patches flattened; only the center 6×6 is left", flatten_where(test_x, ring_mask(grid, win, border=True))),
        "grayscale": ("Grayscale", "color removed (channel mean)", np.repeat(test_x.mean(-1, keepdims=True).round().astype(np.uint8), 3, -1)),
        "shuffled": ("Patches shuffled", "the 64 patches randomly rearranged: local texture kept, global shape destroyed", shuffle_patches(test_x, grid, win)),
        "mirrored": ("Mirrored", "flipped left–right; should cost almost nothing", test_x[:, :, ::-1].copy()),
    }
    result["tests"] = {}
    for key, (name, desc, imgs) in tests.items():
        p = predict(probe, imgs, test_y)
        result["tests"][key] = {
            "name": name,
            "description": desc,
            "overall": float((p == test_y).mean()),
            "per_class": per_class_acc(p, test_y, nc),
            "false_alarm": false_alarms(p, test_y, nc),
        }
        log(f"test {key}: {result['tests'][key]['overall']:.1%}")

    # ----- occlusion evidence per class -----
    sel = np.concatenate([np.flatnonzero(test_y == c)[: args.evidence_per_class] for c in range(nc)])
    scores, preds = occlusion_scores(probe, test_x[sel], grid, win)
    labels = test_y[sel]
    border = ring_mask(grid, 1, border=True).reshape(-1)  # patch-level ring
    pos = np.clip(scores, 0, None)
    share = pos / (pos.sum(1, keepdims=True) + 1e-9)
    border_share = share[:, border].sum(1)
    correct = preds == labels
    ev = {"n_images": int(len(sel)), "border_baseline": float(border.mean()), "per_class": []}
    for c in range(nc):
        m = (labels == c) & correct
        mw = (labels == c) & ~correct
        ev["per_class"].append({
            "n_correct": int(m.sum()),
            "map": share[m].mean(0).tolist() if m.any() else [0.0] * grid * grid,
            "border_share": float(border_share[m].mean()) if m.any() else None,
            "border_share_wrong": float(border_share[mw].mean()) if mw.any() else None,
        })
    ev["border_share_overall"] = float(border_share[correct].mean())
    result["evidence"] = ev
    log(f"occlusion evidence on {len(sel)} images; border share {ev['border_share_overall']:.1%} (uniform {ev['border_baseline']:.1%})")

    # Drill-down examples: per class, the most border-reliant correct images (and one centered).
    examples, ex_imgs = [], []
    for c in range(nc):
        idx = np.flatnonzero((labels == c) & correct)
        if not len(idx):
            continue
        order = idx[np.argsort(-border_share[idx])]
        for kind, j in [("border", order[0]), ("border", order[1]), ("center", order[-1])]:
            examples.append({"slot": len(ex_imgs), "label": int(c), "pred": int(preds[j]), "border_share": float(border_share[j]), "kind": kind})
            ex_imgs.append(test_x[sel[j]])
    np.stack(ex_imgs).astype(np.uint8).tofile(OUT / "examples.bin")
    result["examples"] = examples

    # ----- internals: read-out per block -----
    train_x, train_y = load_images(data_root / "train", args.probe_train_per_class)
    tr_cls, tr_patch = collect_features(probe, train_x)
    te_cls, te_patch = collect_features(probe, test_x)
    lens = []
    for s in te_cls:
        logits = model.classifier(mx.array(s))
        lens.append(float((np.array(mx.argmax(logits, -1)) == test_y).mean()))
    probe_cls = [fit_probe(a, train_y, b, test_y, nc) for a, b in zip(tr_cls, te_cls)]
    probe_patch = [fit_probe(a, train_y, b, test_y, nc) for a, b in zip(tr_patch, te_patch)]
    depths = ["embed"] + [f"block {i + 1}" for i in range(len(probe.blocks))]
    result["readout"] = {"depths": depths, "head_on_cls": lens, "probe_cls": probe_cls, "probe_patch_mean": probe_patch}
    log(f"read-out probe on CLS: {' '.join(f'{a:.0%}' for a in probe_cls)}")

    # ----- internals: knock-outs -----
    base_acc = result["accuracy"]["overall"]
    result["block_knockout"] = []
    for i in range(len(probe.blocks)):
        acc = float((predict(probe, test_x, test_y, skip_block=i) == test_y).mean())
        result["block_knockout"].append({"block": i + 1, "accuracy": acc, "delta": acc - base_acc})
    log("block knock-outs: " + " ".join(f"{b['delta']:+.1%}" for b in result["block_knockout"]))

    # ----- internals: per-head statistics and knock-outs -----
    agg = None
    for sl in batches(len(test_x), 500):
        stats = probe(to_input(test_x[sl]), collect=True)[3]
        stats = [{k: (np.array(v) if isinstance(v, mx.array) else v) for k, v in s.items()} for s in stats]
        agg = stats if agg is None else [{k: a[k] + b[k] for k in a} for a, b in zip(agg, stats)]
    heads, residual = [], []
    for i, s in enumerate(agg):
        for h in range(len(s["entropy"])):
            acc = float((predict(probe, test_x, test_y, zero_head=(i, h)) == test_y).mean())
            heads.append({
                "block": i + 1, "head": h + 1,
                "entropy": float(s["entropy"][h] / s["count_q"]),
                "reach": float(s["reach"][h] / s["count_p"]),
                "to_cls": float(s["to_cls"][h] / s["count_p"]),
                "cls_self": float(s["cls_self"][h] / s["count_b"]),
                "knockout_delta": acc - base_acc,
            })
        residual.append({
            "block": i + 1,
            "input_norm": float(s["x_norm"] / s["tokens"]),
            "attn_update": float(s["attn_norm"] / s["x_norm"]),
            "mlp_update": float(s["mlp_norm"] / s["mid_norm"]),
        })
        log(f"block {i + 1}: head knock-outs " + " ".join(f"{hh['knockout_delta']:+.1%}" for hh in heads[-len(s['entropy']):]))
    result["heads"] = heads
    result["residual"] = residual

    with open(OUT / "analysis.json", "w") as f:
        json.dump(result, f)
    log(f"wrote {OUT / 'analysis.json'}")


if __name__ == "__main__":
    main()
