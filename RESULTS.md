# Results

CIFAR-10 test accuracy, ConvNet-3, 1,000 epochs, 5 training runs each. Raw numbers: `results/ipc*.json`, `results/full.json`.

| Images per class | Total images | Random real subset | Distilled (DM) | Gain |
|---:|---:|---:|---:|---:|
| 1  | 10  | 15.7% ± 1.8 | **28.3% ± 0.6** | +12.6 pts |
| 10 | 100 | 34.2% ± 1.8 | **50.8% ± 0.7** | +16.6 pts |
| 50 | 500 | 51.1% ± 0.9 | **63.5% ± 0.4** | +12.4 pts |

Full training set (30 epochs, 2 runs): 75.7%, 75.9%.
