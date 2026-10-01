# ten dreams

A small vision transformer, trained on ten kinds of 32-pixel photograph, imagines each of them.

Each dream is a feature visualization (after Olah, Mordvintsev & Schubert, *Feature Visualization*,
Distill 2017, and Mordvintsev's *DeepDream*, 2015): a 256 px image, parameterized in a decorrelated
Fourier basis, optimized for 768 steps so that random glimpses of it — continuous-scale crops,
flips, area-averaged down to the model's 32×32 — leave the CIFAR-10 ViT certain of one class. Every
optimization is recorded as 24 frames, dense where the image first appears, and the page plays
them back like a print developing in a darkroom: grey fog, a shape, then a deer.

Four seeds per class, forty dreams, shuffled so a word never repeats twice in a row. Each image's
contrast is stretched once; nothing else is retouched. Click or → for the next dream, space to
pause, i for the note on how they were made.

## Run

```bash
uv run python viz/dreams/dream.py                          # ~10 min → viz/dreams/web/assets/
python3 viz/vit/serve.py --dir viz/dreams/web --port 8767
```

Re-render after retraining; the dreams belong to the checkpoint that made them.
