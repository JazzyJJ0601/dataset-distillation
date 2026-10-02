# Dataset Distillation (CIFAR-10)

Compress the 50,000 CIFAR-10 training images into 10, 100 or 500 synthetic images that train
a network almost as well as a much larger real subset.

The method is **distribution matching** (DM, Zhao & Bilen, WACV 2023): synthetic images start
as real ones and are optimised so that, through many freshly initialised random ConvNets, their
mean feature embedding per class matches the real images' embedding. Real and synthetic batches
get the same random augmentation (differentiable siamese augmentation: colour, flip, scale,
rotate, translate, cutout) at every step, so the images learn content rather than one view.

## Results

ConvNet-3 (width 128, instance norm), trained from scratch for 1,000 epochs on each set,
tested on the full 10,000-image CIFAR-10 test set. Five training runs per row (mean ± std).
The baseline is the fair one: a class-balanced **random subset of real images of the same size**
(a different subset each run), trained the same way with the same augmentation.

| Images per class | Total images | Random real subset | Distilled (DM) | Gain |
|---:|---:|---:|---:|---:|
| 1  | 10  | 15.7% ± 1.8 | **28.3% ± 0.6** | +12.6 pts |
| 10 | 100 | 34.2% ± 1.8 | **50.8% ± 0.7** | +16.6 pts |
| 50 | 500 | 51.1% ± 0.9 | **63.5% ± 0.4** | +12.4 pts |

Reference: the same network trained on all 50,000 images for 30 epochs reaches 75.7% / 75.9%
(two runs).

The distilled sets beat same-size real subsets at every size, and 100 distilled images
(50.8%) match roughly 500 random real ones (51.1%), a 5x reduction. The numbers are in line
with, and slightly above, those published for DM (26.0 / 48.9 / 63.0%).

**Honest limits**

- This is a clean, measured reimplementation of a published method, not a new method.
- One distillation run per size (seed 0); the ± is across the five evaluation trainings.
- The full-data reference used 30 epochs, so it understates what full data can reach.
- An earlier version of this repo used gradient matching with a bug that left the synthetic
  images' labels random; it scored 12.9%, barely above chance. That code was replaced.

## Reproduce

Needs a CUDA GPU and the CIFAR-10 python batches in `data/cifar-10-batches-py/`
(from https://www.cs.toronto.edu/~kriz/cifar.html, not included).

```bash
pip install -r requirements.txt
python dm.py --ipc 10            # distil, evaluate, write results/ipc10.json (~15 min on an RTX 3090 Ti)
bash run_all.sh                  # every number above (~50 min)
pytest -q tests
```

`results/` holds the JSON for each run, the logs, and the distilled images (`syn_ipc*.pt`).

## License

MIT
