#!/usr/bin/env python3
"""Dataset distillation on CIFAR-10 by distribution matching (Zhao & Bilen, WACV 2023),
evaluated against class-balanced random real subsets of the same size.

  python dm.py --ipc 10                 # distil 10 images/class, evaluate, write results/ipc10.json
  python dm.py --ipc 10 --iters 0       # baseline only (random subsets)
"""
import argparse, json, os, pickle, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "cuda"
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)


def load_cifar(root="data/cifar-10-batches-py"):
    def batch(name):
        with open(os.path.join(root, name), "rb") as f:
            d = pickle.load(f, encoding="bytes")
        x = torch.tensor(d[b"data"], dtype=torch.float32).view(-1, 3, 32, 32) / 255.0
        return (x - MEAN) / STD, torch.tensor(d[b"labels"])
    xs, ys = zip(*[batch(f"data_batch_{i}") for i in range(1, 6)])
    xt, yt = batch("test_batch")
    return torch.cat(xs), torch.cat(ys), xt, yt


class ConvNet(nn.Module):
    """The standard ConvNet-3 of the dataset-distillation literature (width 128, instance norm, avg pool)."""
    def __init__(self, width=128, classes=10):
        super().__init__()
        layers, c = [], 3
        for _ in range(3):
            layers += [nn.Conv2d(c, width, 3, padding=1), nn.GroupNorm(width, width, affine=True),
                       nn.ReLU(inplace=True), nn.AvgPool2d(2)]
            c = width
        self.features = nn.Sequential(*layers, nn.Flatten())
        self.classifier = nn.Linear(width * 4 * 4, classes)

    def forward(self, x):
        return self.classifier(self.features(x))


def dsa(x, gen):
    """Differentiable siamese augmentation: one random draw (from `gen`) applied to the whole batch,
    so real and synthetic batches given the same generator state get the same transform."""
    r = lambda *s: torch.rand(*s, generator=gen, device="cpu").item()
    # colour: brightness, saturation, contrast
    x = x + (r(1) - 0.5)
    m = x.mean(dim=1, keepdim=True); x = (x - m) * (r(1) * 2) + m
    m = x.mean(dim=[1, 2, 3], keepdim=True); x = (x - m) * (r(1) + 0.5) + m
    # geometry: flip, scale, rotate, translate in one affine grid
    s = 1 + (r(1) - 0.5) * 0.4
    a = (r(1) - 0.5) * 2 * 15 / 180 * np.pi
    fx = -1.0 if r(1) < 0.5 else 1.0
    tx, ty = (r(1) - 0.5) * 0.25, (r(1) - 0.5) * 0.25
    theta = torch.tensor([[fx * s * np.cos(a), -s * np.sin(a), tx],
                          [fx * s * np.sin(a), s * np.cos(a), ty]], dtype=x.dtype, device=x.device)
    grid = F.affine_grid(theta.expand(x.size(0), 2, 3), x.shape, align_corners=True)
    x = F.grid_sample(x, grid, align_corners=True)
    # cutout: a 16x16 hole
    cx, cy = int(r(1) * 32), int(r(1) * 32)
    mask = torch.ones_like(x[:1, :1])
    mask[..., max(cy - 8, 0):cy + 8, max(cx - 8, 0):cx + 8] = 0
    return x * mask


def train_eval(x, y, xt, yt, seed, epochs=1000, lr=0.01, bs=256):
    """Train a fresh ConvNet on (x, y) with augmentation, return test accuracy."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    net = ConvNet().to(DEV)
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    x, y = x.to(DEV), y.to(DEV)
    for ep in range(epochs):
        if ep == epochs // 2:
            for g in opt.param_groups: g["lr"] = lr * 0.1
        perm = torch.randperm(len(x), device=DEV)
        for i in range(0, len(x), bs):
            idx = perm[i:i + bs]
            loss = F.cross_entropy(net(dsa(x[idx], gen)), y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval(); correct = 0
    with torch.no_grad():
        for i in range(0, len(xt), 1000):
            correct += (net(xt[i:i + 1000].to(DEV)).argmax(1).cpu() == yt[i:i + 1000]).sum().item()
    return correct / len(xt)


def random_subset(by_class, ipc, gen):
    xs, ys = [], []
    for c, imgs in enumerate(by_class):
        idx = torch.randperm(len(imgs), generator=gen)[:ipc]
        xs.append(imgs[idx]); ys.append(torch.full((ipc,), c))
    return torch.cat(xs), torch.cat(ys)


def distill(by_class, ipc, iters, seed, batch_real=256, lr_img=1.0):
    gen = torch.Generator().manual_seed(seed)
    syn, syn_y = random_subset(by_class, ipc, gen)       # initialise from real images
    syn = syn.to(DEV).requires_grad_(True)
    opt = torch.optim.SGD([syn], lr=lr_img, momentum=0.5)
    real_gpu = [c.to(DEV) for c in by_class]
    aug_gen = torch.Generator().manual_seed(seed + 1)
    t0 = time.time()
    for it in range(iters):
        net = ConvNet().to(DEV)                           # fresh random network each step
        for p in net.parameters(): p.requires_grad_(False)
        loss = 0.0
        for c in range(len(by_class)):
            xr = real_gpu[c][torch.randint(len(real_gpu[c]), (batch_real,), device=DEV)]
            xs = syn[c * ipc:(c + 1) * ipc]
            state = aug_gen.get_state()
            er = net.features(dsa(xr, aug_gen)).mean(0)
            aug_gen.set_state(state)                      # same augmentation for the synthetic batch
            es = net.features(dsa(xs, aug_gen)).mean(0)
            loss = loss + ((er - es) ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 1000 == 0 or it == iters - 1:
            print(f"  iter {it:5d}  loss {loss.item():.2f}  {time.time() - t0:.0f}s", flush=True)
    return syn.detach().cpu(), syn_y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ipc", type=int, default=10)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--evals", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    x, y, xt, yt = load_cifar()
    by_class = [x[y == c] for c in range(10)]
    out = {"ipc": a.ipc, "iters": a.iters, "eval_epochs": a.epochs, "random": [], "dm": []}

    for k in range(a.evals):                              # baseline: a different random subset each time
        xs, ys = random_subset(by_class, a.ipc, torch.Generator().manual_seed(1000 + k))
        out["random"].append(train_eval(xs, ys, xt, yt, seed=k, epochs=a.epochs))
        print(f"random subset {k}: {out['random'][-1]:.4f}", flush=True)

    if a.iters > 0:
        syn, syn_y = distill(by_class, a.ipc, a.iters, a.seed)
        os.makedirs("results", exist_ok=True)
        torch.save({"images": syn, "labels": syn_y}, f"results/syn_ipc{a.ipc}.pt")
        for k in range(a.evals):
            out["dm"].append(train_eval(syn, syn_y, xt, yt, seed=k, epochs=a.epochs))
            print(f"distilled eval {k}: {out['dm'][-1]:.4f}", flush=True)

    for key in ("random", "dm"):
        if out[key]:
            out[key + "_mean"], out[key + "_std"] = float(np.mean(out[key])), float(np.std(out[key]))
    os.makedirs("results", exist_ok=True)
    with open(f"results/ipc{a.ipc}.json", "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k.endswith(("mean", "std"))}))


if __name__ == "__main__":
    main()
