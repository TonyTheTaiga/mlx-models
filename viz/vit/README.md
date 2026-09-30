# Lumen — ViT visualizer

An in-browser, artistic view of the CIFAR-10 ViT in `networks/transformers/vit/`. The model runs
entirely in JavaScript (a Web Worker port of the MLX forward pass), so you can paint on an image
and watch attention and the residual stream change live.

- **Tower** — one plate per depth (pixels → embedding → block 1…6), folded like an accordion so
  only the active block's input and output plates are open. Tiles are colored by a fixed per-depth
  PCA of the residual stream; threads are the block's strongest attention into the query token
  (CLS by default, click any tile to follow a patch), and a dotted line is its residual path.
  Scroll or ↑/↓ to move through depth; a card lists the top sources in words. Optional controls:
  **play** (space) walks the query up through every block; **attention → MLP** opens each block
  into its state between the two, with one stroke per token for how far the MLP moves it; and
  **split heads** shows a compact scrub per head, side by side. (`tower.js` also
  keeps three unused alternates — loom, orbit, stack — selectable via `S.variant`.)
- **Atlas** — every head's attention map from the query, block by block, next to the residual stream.
- **Drift** — Rope2D's `freq`/`pos_idx` were trainable, so each block learned its own patch
  coordinates; the image is re-assembled at those learned positions.
- **Verdict / Depth lens** — class probabilities, and the classifier head read out from CLS after
  every block.

## Run

```bash
uv run python viz/vit/export.py            # writes viz/vit/web/assets/ from the checkpoint
python3 viz/vit/serve.py
```

Then open http://localhost:8765. The footer reports the max logit difference between the browser
forward pass and MLX on a few reference images (≈2e-5).

Keys: `t`/`a`/`d` switch views, `1`–`4` solo a head, `0` all heads, `Esc` resets the query to CLS,
`←`/`→` step through the gallery.
