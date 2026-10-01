# Dataset Distillation

This project implements gradient matching dataset distillation for CIFAR-10, following the approach from [Dataset Distillation via Gradient Matching](https://arxiv.org/abs/2104.07512).

## Mathematical Background

Dataset distillation aims to compress a large dataset into a small set of synthetic images that capture the essential information. The key idea is **gradient matching**:

$$\min_{S} \sum_{t=1}^{T} \| \nabla_{\theta} \mathcal{L}(\theta_{t-1}, D_{real}) - \nabla_{\theta} \mathcal{L}(\theta_{t-1}, S) \|_2^2$$

Where:
- $S$ is the set of synthetic images
- $D_{real}$ is the real dataset
- $\theta$ are model parameters
- $\mathcal{L}$ is the loss function

We initialize synthetic images randomly, then optimize them to minimize the difference between:
1. Gradients of model trained on real data
2. Gradients of model trained on synthetic data

## Files

- `distill.py` - Core distillation logic (gradient matching)
- `evaluate.py` - Trains models on real vs synthetic data and reports test accuracy
- `requirements.txt` - Python dependencies
- `README.md` - This file

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run distillation:
   ```bash
   python distill.py
   ```
   This generates `synthetic_images.pt` containing 100 synthetic images (10 per class).

3. Evaluate:
   ```bash
   python evaluate.py
   ```
   Compares test accuracy of models trained on real vs synthetic data.

## Results (Honest Assessment)

| Training Data | Test Accuracy |
|--------------|---------------|
| Real CIFAR-10 | 11.79% |
| Synthetic (100 images) | 12.89% |

The synthetic dataset achieves only a **~1 percentage point gain** over random real samples (12.89% vs 11.79%). This is barely above random chance (10%), indicating the distillation signal is very weak.

### Limitations
- Very small synthetic set (100 images) relative to CIFAR-10 complexity
- SimpleCNN architecture may underfit
- Training noise may mask true signal

## License

MIT