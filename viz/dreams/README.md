# ten dreams

A small vision transformer, trained on ten kinds of 32-pixel photograph, imagines each of them.

Each dream is a feature visualization (after Olah, Mordvintsev & Schubert, *Feature Visualization*,
Distill 2017, and Mordvintsev's *DeepDream*, 2015): a 256 px image, parameterized in a decorrelated
Fourier basis, optimized for 768 steps so that random glimpses of it — continuous-scale crops,
flips, area-averaged down to the model's 32×32 — leave the CIFAR-10 ViT certain of one class. Every
optimization is recorded as 24 frames, dense where the image first appears.

The page is **a field of forty**: every dream as a WebGL point cloud (three.js), a column per class
— vehicles on the left, animals on the right — and a row per seed, so you can compare how the model
imagines a cat with how it imagines a dog. Every pixel is a point, set back or brought forward by
*when the model decided it*: computed in the browser from the recorded frames, the fraction of the
optimization during which each pixel was still far from its final color, smoothed. What settled
first (usually ground and sky) sits furthest away; the last details to resolve come nearest.

Dreams arrive as they load, each condensing out of its own fog. Drag to look around, scroll to come
closer; click a dream to fly up to it and watch it develop again, each point flying in at the moment
its pixel settled, like a print coming up in a darkroom tray. Click the dark or press Esc to step
back; arrow keys walk between dreams; i for the note on how they were made. Each image's contrast is
stretched once; nothing else is retouched.

## Run

```bash
uv run python viz/dreams/dream.py                          # ~10 min → viz/dreams/web/assets/
python3 viz/vit/serve.py --dir viz/dreams/web --port 8767
```

Re-render after retraining; the dreams belong to the checkpoint that made them.
