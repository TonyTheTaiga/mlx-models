# GAZE — seven studies in machine sight

An online exhibition built on the CIFAR-10 ViT in `networks/transformers/vit/`, grown out of what
[Lumen](../vit/README.md) measured about it. Every room says its numbers as measured.

| room | after | what it is |
|---|---|---|
| **I · Perception Engine** | Tom White, *Perception Engines*; Brown et al., *Adversarial Patch* | an evolutionary loop composes with 8 inks, 3 shapes and 8 strokes until the model is certain of a class, then removes marks while it stays certain; draw into it, print the result |
| **II · Enough** | Anna Ridler, *Myriad (Tulips)* | test photographs shown with only the patches the prediction needed (k = 0 counts: sometimes the flat average color was enough); scroll to bring the rest back |
| **III · Plausible Lies** | Distill, *Communicating with Interactive Articles*; Petsiuk et al., *RISE* | pick which of two unlabeled heatmaps (occlusion vs. Grad-CAM) to trust; a replayed deletion test decides each round |
| **IV · Hallucination Field** | Trevor Paglen; Refik Anadol | a dark, full-bleed field of six images drifting between classes, each evolving toward what the model is asked to see |
| **V · Collage** | Hannah Höch's photomontage | evolution assembles the most convincing \<class\> from 64,000 real 4×4 fragments of 1,000 photographs, then replaces every fragment cut from a real photo of that class while the model stays ≥99% sure; hover a tile for its source |
| **VI · Tug of War** | the exquisite corpse; adversarial examples | you paint toward one class; after each stroke the model searches for the stroke that pulls hardest toward another; eight turns each, the canvas keeps the fight |
| **VII · Mirror** | Memo Akten, *Learning to See* | your camera (or a photo), downsampled to 32×32 and named live in the model's ten words, with an occlusion heatmap every two seconds; frames never leave the browser |

Rooms I, IV, V, VI and VII run the model live in the browser (a worker pool over the JS port of the MLX
forward pass). Rooms II and III are precomputed in MLX over 1,000 test photographs.

## Run

```bash
uv run python viz/gaze/export.py                              # ~40 s → viz/gaze/web/assets/
python3 viz/vit/serve.py --dir viz/gaze/web --port 8766
```

Then open http://localhost:8766. Re-run the export after retraining.
