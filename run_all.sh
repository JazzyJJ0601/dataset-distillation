#!/usr/bin/env bash
# Reproduce every number in the README (about 50 minutes on one RTX 3090 Ti).
set -e
cd "$(dirname "$0")"
PY=${PY:-python}
mkdir -p results
for ipc in 1 10 50; do $PY dm.py --ipc $ipc > results/run_ipc$ipc.log 2>&1; done
$PY -c "
import json
from dm import load_cifar, train_eval
x, y, xt, yt = load_cifar()
acc = [train_eval(x, y, xt, yt, seed=k, epochs=30) for k in range(2)]
json.dump({'full_data': acc, 'epochs': 30}, open('results/full.json', 'w'))
print(acc)" > results/run_full.log 2>&1
