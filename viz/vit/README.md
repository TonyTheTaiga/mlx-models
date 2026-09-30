# Lumen — which patches did the model lean on?

An in-browser, image-first answer to one question about the CIFAR-10 ViT in
`networks/transformers/vit/`: **which patches did the model lean on for its final prediction?**
The model runs entirely in JavaScript (a Web Worker port of the MLX forward pass, checked against
MLX on load), so you can paint on an image, drop in your own, and watch the answer change.

**How it answers.** Each of the 64 patches is hidden in turn — flattened to the image's average
color — and scored by how much the log-probability of the predicted class drops (negative scores
mean the patch argued *against* it). The ranking is then put to the test:

- **keep only the top k** patches (the rest flattened): does the prediction survive?
- **remove the top k**: does it collapse?

The headline is the smallest k that keeps the answer at ≥80% of its confidence — "6 of 64 patches
were enough". The large image shows exactly those patches, with the rest flattened; drag k to see
more or fewer, click a patch to keep everything down to its rank, and toggle **signed map** to color
every patch for and against. Pick a runner-up class to ask "why not a cat?": the page then shows
the strongest case its best patches can make.

The sweeps (64 + 130 forward passes per image) run on a small worker pool, so they take a second
or two and never block painting.

## Run

```bash
uv run python viz/vit/export.py            # writes viz/vit/web/assets/ from the checkpoint
python3 viz/vit/serve.py
```

Then open http://localhost:8765. Keys: `←`/`→` step through the gallery.
