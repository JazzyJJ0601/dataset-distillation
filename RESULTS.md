# Dataset Distillation Results

## Experiment Overview

This experiment evaluates gradient-matching dataset distillation for CIFAR-10. The goal is to synthesize a small set of images that capture the essential information from the full training set, enabling models trained on the synthetic data to perform competitively.

## Method
- Distillation technique: Gradient matching (matching model gradients on real vs synthetic data)
- Target dataset: CIFAR-10 (10 classes, 50K train / 10K test images)
- Synthetic dataset size: 100 images (10 per class)
- Architecture: SimpleCNN (convolutional network)

## Results

| Dataset Type | Test Accuracy |
|--------------|---------------|
| Real CIFAR-10 | 11.79% |
| Synthetic Images | 12.89% |

## Honest Analysis

The synthetic dataset achieves 12.89% accuracy, which is only **~1 percentage point higher** than random real samples at 11.79%. This gain is marginal and barely above the 10% random chance baseline. The gradient matching approach shows some signal, but the effect is very weak with only 100 synthetic images on CIFAR-10.

### Setup Summary
- Synthetic budget: 100 images total (10 per class)
- Evaluation: SimpleCNN trained 10 epochs on synthetic vs sampled real data
- Test set: 10K CIFAR-10 images

### Limitations
- Small synthetic dataset relative to task complexity
- Simple architecture may limit representation capacity
- Training noise could obscure true performance differences