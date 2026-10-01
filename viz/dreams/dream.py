"""
Ten Dreams: what the CIFAR-10 ViT imagines each of its ten classes looks like.

Feature visualization after Olah, Mordvintsev & Schubert, "Feature Visualization" (Distill, 2017):
an S×S image, parameterized in a decorrelated Fourier basis, is optimized so that random crops of
it — flipped, scaled and averaged down to the model's 32×32 — maximize one class logit (with a
softmax term so it's that class, not just a strong image). Every dream's development is recorded
as a sequence of frames, so the page can show it emerging from noise.

Writes viz/dreams/web/assets/: <class>-<seed>/NN.jpg frames and manifest.json.

Usage: uv run python viz/dreams/dream.py [--size 256] [--seeds 4] [--steps 768]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

sys.path.insert(0, str(Path(__file__).parents[1] / "vit"))
from export import ROOT, fingerprint, load_model  # noqa: E402

OUT = Path(__file__).parent / "web" / "assets"

# Color decorrelation (from Lucid; ImageNet statistics), normalized.
_C = np.array([[0.26, 0.09, 0.02], [0.27, 0.00, -0.05], [0.27, -0.09, 0.03]], dtype=np.float32)
COLOR = mx.array(_C / np.max(np.linalg.norm(_C, axis=0)))


def frame_steps(steps: int, n: int = 24) -> list[int]:
    """n checkpoints, dense early (where the image appears) and sparse late."""
    return sorted({int(round(steps * (i / (n - 1)) ** 2.2)) for i in range(n)})


class Dreamer:
    def __init__(self, model, size: int):
        self.model, self.S = model, size
        fy = np.fft.fftfreq(size)[:, None]
        fx = np.fft.rfftfreq(size)[None, :]
        self.scale = mx.array(
            1.0 / np.maximum(np.sqrt(fx**2 + fy**2), 1.0 / size), dtype=mx.float32
        )

    def image(self, p):
        x = mx.fft.irfft2((p["re"] + 1j * p["im"]) * self.scale, s=(self.S, self.S)) / 2.5
        return mx.sigmoid(x.transpose(1, 2, 0) @ COLOR.T)  # (S, S, 3) in 0..1

    def views(self, img, n, rng):
        """
        n random glimpses at the model's 32×32: a crop of continuous size and position, area-
        averaged with fractional box edges (an integral image sampled bilinearly), so no pooling
        grid lines up from one step to the next and none survives into the image.
        """
        S = self.S
        integral = mx.pad(mx.cumsum(mx.cumsum(img, axis=0), axis=1), [(1, 0), (1, 0), (0, 0)])
        out = []
        for _ in range(n):
            c = rng.uniform(0.5, 1.0) * S
            y0, x0 = rng.uniform(0, S - c, 2)
            edges = np.arange(33) * (c / 32)
            v = self._boxes(integral, y0 + edges, x0 + edges) / (c / 32) ** 2
            out.append(v[:, ::-1] if rng.random() < 0.5 else v)
        return mx.stack(out)

    @staticmethod
    def _sample(integral, ys, xs):
        """Bilinear lookup of the integral image at fractional coordinates: (len(ys), len(xs), 3)."""
        top = integral.shape[0] - 1
        y0 = np.clip(np.floor(ys).astype(np.int32), 0, top - 1)
        x0 = np.clip(np.floor(xs).astype(np.int32), 0, top - 1)
        fy = mx.array((ys - y0)[:, None, None], dtype=mx.float32)
        fx = mx.array((xs - x0)[None, :, None], dtype=mx.float32)
        Y0, X0 = mx.array(y0)[:, None], mx.array(x0)[None, :]
        a, b = integral[Y0, X0], integral[Y0, X0 + 1]
        c, d = integral[Y0 + 1, X0], integral[Y0 + 1, X0 + 1]
        return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy

    def _boxes(self, integral, ye, xe):
        A = self._sample(integral, ye, xe)
        return A[1:, 1:] - A[:-1, 1:] - A[1:, :-1] + A[:-1, :-1]

    def dream(self, cls: int, steps: int, seed: int, record: list[int]):
        rng = np.random.default_rng(seed)
        mx.random.seed(seed)
        shape = (3, self.S, self.S // 2 + 1)
        p = {"re": mx.random.normal(shape) * 0.01, "im": mx.random.normal(shape) * 0.01}
        opt = optim.Adam(learning_rate=0.08)

        def loss(p, s):
            logits = self.model(self.views(self.image(p), 8, np.random.default_rng(s)))
            return -(logits[:, cls].mean() - 0.3 * mx.logsumexp(logits, axis=-1).mean())

        step = mx.value_and_grad(loss)
        frames = []
        for t in range(steps + 1):
            if t in record:
                frames.append(np.array(self.image(p)))
            if t == steps:
                break
            _, g = step(p, int(rng.integers(1 << 30)))
            opt.update(p, g)
            mx.eval(p, opt.state)
        final = mx.array(frames[-1])
        belief = mx.softmax(self.model(self.views(final, 32, np.random.default_rng(1))), -1)
        return frames, float(belief[:, cls].mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("training/vit/checkpoints/cifar10"))
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--steps", type=int, default=768)
    parser.add_argument("--only", help="comma-separated classes to render (for quick tests)")
    args = parser.parse_args()
    t0 = time.time()

    model, config, classes = load_model(ROOT / args.checkpoint)
    model.freeze()
    dreamer = Dreamer(model, args.size)
    record = frame_steps(args.steps)
    OUT.mkdir(parents=True, exist_ok=True)
    dreams = []
    for seed in range(args.seeds):
        for c, name in enumerate(classes):
            if args.only and name not in args.only.split(","):
                continue
            frames, belief = dreamer.dream(c, args.steps, seed, record)
            # One contrast stretch per dream (from its final frame), applied to every frame.
            lo, hi = np.percentile(frames[-1], [0.5, 99.5])
            folder = OUT / f"{name}-{seed}"
            folder.mkdir(exist_ok=True)
            for i, f in enumerate(frames):
                img = np.clip((f - lo) / max(hi - lo, 1e-6), 0, 1)
                cv2.imwrite(
                    str(folder / f"{i:02d}.jpg"),
                    cv2.cvtColor((img * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 88],
                )
            dreams.append(
                {
                    "class": name,
                    "seed": seed,
                    "folder": folder.name,
                    "frames": len(frames),
                    "belief": belief,
                }
            )
            print(
                f"[{time.time() - t0:6.1f}s] {name:<11} seed {seed}  belief {belief:.3f}",
                flush=True,
            )
    with open(OUT / "manifest.json", "w") as f:
        json.dump(
            {
                "fingerprint": fingerprint(ROOT / args.checkpoint / "best.npz"),
                "model": config["model"],
                "params": int(sum(v.size for _, v in tree_flatten(model.parameters()))),
                "size": args.size,
                "steps": args.steps,
                "record": record,
                "classes": classes,
                "dreams": dreams,
            },
            f,
        )


if __name__ == "__main__":
    main()
