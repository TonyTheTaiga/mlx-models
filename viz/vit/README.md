# Lumen — what is your ViT actually using?

A model-builder's report on the CIFAR-10 ViT in `networks/transformers/vit/`, plus a per-image
drill-down. The heavy lifting runs in MLX over the full test set (`analyze.py`); the page renders it.

**Shortcuts** — is it looking at the right thing?
- Controlled tests on all 10,000 test images: *border only*, *center only*, *grayscale*,
  *patches shuffled*, *mirrored*. Each class is scored by how much of its above-chance accuracy
  survives (accuracy − how often other images get that label, relative to clean images), so a
  class the model falls back on can't masquerade as robust. Flags: background shortcut, color
  reliance, texture over shape.
- Occlusion evidence per class: the average evidence map and the share on the border, with the
  most border-reliant images one click from the drill-down.

**Internals** — is every part of the network earning its keep?
- Where class information appears: linear probes on CLS and on the mean patch token after every
  block, and the trained head applied early.
- Knock-outs: accuracy with each block skipped, and with each head zeroed.
- Per-head attention statistics (entropy, spatial reach, attention to CLS) and residual-stream
  update sizes. Findings are written from the numbers at the top.

**Inspect an image** — the drill-down. The model runs in the browser (a Web Worker port of the MLX
forward pass, checked against MLX on load): each patch is hidden in turn and scored by the drop in
the prediction's log-probability, then the ranking is tested by keeping or removing the top k.
A verdict card sets the prediction against the true label; paint on the image (✎ paint), drop in
your own, and watch the answer change.

## Run

```bash
uv run python viz/vit/export.py     # browser model + gallery → viz/vit/web/assets/
uv run python viz/vit/analyze.py    # full-test-set analysis (~3 min) → assets/analysis.json
python3 viz/vit/serve.py
```

Then open http://localhost:8765. Re-run both scripts after retraining; the page warns if they
were produced from different checkpoints.
