"""
Export the CIFAR-10 ViT and precomputed studies for GAZE, the four-room exhibition.

Writes to viz/gaze/web/assets/:
  weights.bin / weights.json   the browser model (rooms I and IV evolve images against it live)
  enough.json / enough.bin     room II: images reduced to the patches the prediction needed
  lies.json / lies.bin         room III: occlusion vs. Grad-CAM maps with their deletion curves

Everything is computed over a pool of test images (100 per class by default).

Usage: uv run python viz/gaze/export.py [--checkpoint ...] [--pool-per-class 100]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "vit"))
from export import (  # noqa: E402
    ROOT,
    fingerprint,
    gradcam,
    load_model,
    load_split,
    occlusion,
    write_weights,
)

OUT = Path(__file__).parent / "web" / "assets"


def probs(model, images: np.ndarray, bs: int = 520) -> np.ndarray:
    out = []
    for s in range(0, len(images), bs):
        out.append(
            np.array(mx.softmax(model(mx.array(images[s : s + bs].astype(np.float32) / 255.0)), -1))
        )
    return np.concatenate(out)


def flatten(img: np.ndarray, patches, grid: int, win: int) -> np.ndarray:
    out = img.copy()
    mean = img.reshape(-1, 3).mean(0).round().astype(np.uint8)
    mask = np.zeros(grid * grid, bool)
    mask[list(patches)] = True
    out[np.kron(mask.reshape(grid, grid), np.ones((win, win), bool))] = mean
    return out


def curves(model, img, order, grid, win):
    """Class probabilities with only the top-k kept, and with the top-k removed, k = 0..P."""
    P = grid * grid
    keep = [flatten(img, order[k:], grid, win) for k in range(P + 1)]
    drop = [flatten(img, order[:k], grid, win) for k in range(P + 1)]
    p = probs(model, np.stack(keep + drop))
    return p[: P + 1], p[P + 1 :]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("training/vit/checkpoints/cifar10"))
    parser.add_argument("--weights", default="best.npz")
    parser.add_argument("--pool-per-class", type=int, default=100)
    parser.add_argument("--lies-rounds", type=int, default=20)
    args = parser.parse_args()
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:6.1f}s] {m}", flush=True)  # noqa: E731

    ckpt = ROOT / args.checkpoint
    model, config, classes = load_model(ckpt, args.weights)
    win = config["model"]["win_size"]
    grid = config["dataset"]["image_size"] // win
    P = grid * grid
    OUT.mkdir(parents=True, exist_ok=True)
    write_weights(model, config, classes, ckpt / args.weights, OUT)
    fp = fingerprint(ckpt / args.weights)

    images, labels = load_split(
        ROOT / config["dataset"]["path"] / "test", args.pool_per_class, seed=7
    )
    pr = probs(model, images)
    preds, conf = pr.argmax(1), pr.max(1)
    log(f"pool of {len(images)} images, accuracy {(preds == labels).mean():.1%}")

    # Occlusion ranking (drop in log p(pred) per patch) and keep/remove curves for every image.
    occ = occlusion(model, images, preds, grid, win)
    order = np.argsort(-occ, axis=1)
    keep_p, drop_p = [], []
    for i in range(len(images)):
        k, d = curves(model, images[i], order[i], grid, win)
        keep_p.append(k)
        drop_p.append(d)
    keep_p, drop_p = np.stack(keep_p), np.stack(drop_p)  # (N, P+1, C)
    log("occlusion rankings and keep/remove curves")

    # Enough: the smallest k whose kept patches alone still give the same answer at ≥80% of the
    # original confidence. k = 0 counts: with no patch kept the image is a flat field of its mean
    # color, and sometimes that alone already gets the same answer.
    idx = np.arange(len(images))
    same = keep_p.argmax(-1) == preds[:, None]
    keep_pred = keep_p[idx, :, preds]  # (N, P+1): p(prediction) with only the top-k kept
    enough_ok = same & (keep_pred >= 0.8 * conf[:, None])
    enough = np.where(enough_ok.any(1), enough_ok.argmax(1), P)

    correct = preds == labels
    fewest = lambda c: c[np.lexsort((-conf[c], enough[c]))]  # noqa: E731  fewest patches, surest
    chosen = []
    # "Some": per class, the correct images that needed the fewest patches (at least one).
    for c in range(len(classes)):
        chosen += [
            (int(i), "some") for i in fewest(idx[correct & (labels == c) & (enough > 0)])[:3]
        ]
    # "None": correct images whose mean color alone was enough, one per class where it happens.
    for c in range(len(classes)):
        chosen += [
            (int(i), "none") for i in fewest(idx[correct & (labels == c) & (enough == 0)])[:1]
        ]
    # "Wrong": confident mistakes that needed the fewest patches, one per (label, prediction) pair.
    confident_wrong = idx[~correct & (conf >= 0.8)]
    seen = set()
    for i in fewest(confident_wrong):
        if (labels[i], preds[i]) not in seen and len(seen) < 10:
            seen.add((labels[i], preds[i]))
            chosen.append((int(i), "wrong"))
    items = []
    for slot, (i, room) in enumerate(chosen):
        items.append(
            {
                "slot": slot,
                "room": room,
                "label": int(labels[i]),
                "pred": int(preds[i]),
                "confidence": float(conf[i]),
                "enough": int(enough[i]),
                "order": order[i].tolist(),
                "keep": keep_p[i, :, preds[i]].round(4).tolist(),
                "enough_confidence": float(keep_p[i, enough[i], preds[i]]),
            }
        )
    np.stack([images[i] for i, _ in chosen]).astype(np.uint8).tofile(OUT / "enough.bin")
    with open(OUT / "enough.json", "w") as f:
        json.dump(
            {
                "fingerprint": fp,
                "classes": classes,
                "pool": int(len(images)),
                "median_enough_correct": float(np.median(enough[correct])),
                "color_alone_share": float((enough[correct] == 0).mean()),
                "confident_mistakes": int(len(confident_wrong)),
                "color_alone_share_mistakes": float((enough[confident_wrong] == 0).mean()),
                "items": items,
            },
            f,
        )
    log(
        f"enough: median {np.median(enough[correct]):.0f} patches for correct images; "
        f"mean color alone enough for {(enough[correct] == 0).mean():.0%}; {len(items)} works"
    )

    # Plausible Lies: occlusion vs Grad-CAM on randomly drawn, correctly classified images. Each
    # map is judged by its own deletion curve — no answer is fixed in advance.
    gcam = np.concatenate(
        [
            gradcam(model, images[s : s + 250], preds[s : s + 250])
            for s in range(0, len(images), 250)
        ]
    )
    g_order = np.argsort(-gcam, axis=1)
    g_drop = np.stack(
        [
            curves(model, images[i], g_order[i], grid, win)[1][:, preds[i]]
            for i in range(len(images))
        ]
    )
    o_drop = drop_p[idx, :, preds]
    auc_o, auc_g = o_drop.mean(1), g_drop.mean(1)
    # Baseline: the same deletion test with patches in a random order.
    r_order = np.stack([np.random.default_rng(100 + i).permutation(P) for i in range(len(images))])
    auc_r = np.array(
        [
            curves(model, images[i], r_order[i], grid, win)[1][:, preds[i]].mean()
            if correct[i]
            else np.nan
            for i in range(len(images))
        ]
    )
    rng = np.random.default_rng(11)
    rounds = rng.choice(idx[correct], size=args.lies_rounds, replace=False)
    lies = []
    for slot, i in enumerate(rounds):
        lies.append(
            {
                "slot": slot,
                "label": int(labels[i]),
                "pred": int(preds[i]),
                "confidence": float(conf[i]),
                "occlusion": np.clip(occ[i], 0, None).round(5).tolist(),
                "gradcam": gcam[i].round(6).tolist(),
                "occlusion_order": order[i].tolist(),
                "gradcam_order": g_order[i].tolist(),
                "occlusion_drop": o_drop[i].round(4).tolist(),
                "gradcam_drop": g_drop[i].round(4).tolist(),
            }
        )
    images[rounds].astype(np.uint8).tofile(OUT / "lies.bin")
    with open(OUT / "lies.json", "w") as f:
        json.dump(
            {
                "fingerprint": fp,
                "classes": classes,
                "pool": int(correct.sum()),
                "occlusion_wins": float((auc_o[correct] < auc_g[correct]).mean()),
                "mean_auc": {
                    "occlusion": float(auc_o[correct].mean()),
                    "gradcam": float(auc_g[correct].mean()),
                    "random": float(np.nanmean(auc_r)),
                },
                "rounds": lies,
            },
            f,
        )
    log(
        f"lies: occlusion's deletion beats Grad-CAM's on {(auc_o[correct] < auc_g[correct]).mean():.0%} of {correct.sum()} correct images"
    )


if __name__ == "__main__":
    main()
