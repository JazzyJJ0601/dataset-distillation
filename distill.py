#!/usr/bin/env python3
"""Gradient matching dataset distillation for CIFAR-10."""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    def forward(self, x):
        return self.net(x)

def load_cifar10():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                             (0.2470, 0.2435, 0.2616))
    ])
    train = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    return DataLoader(train, batch_size=10, shuffle=True, num_workers=1)

def init_synthetic(n_class=10, n_per_class=10):
    """Initialize synthetic images: n_per_class per class."""
    imgs = [torch.randn(n_per_class, 3, 32, 32) * 0.1 for _ in range(n_class)]
    return torch.cat(imgs, dim=0)

def distill(n_steps=100, lr=0.01, per_class=10):
    real_loader = load_cifar10()
    synthetic = init_synthetic(n_per_class=per_class)
    synthetic.requires_grad_(True)
    opt = torch.optim.Adam([synthetic], lr=lr)

    for step in range(n_steps):
        real_batch = next(iter(real_loader))[0]
        model_r = SimpleCNN()
        opt_r = optim.SGD(model_r.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()
        
        out_r = model_r(real_batch)
        labels = torch.randint(0, 10, (real_batch.size(0),))
        loss_r = loss_fn(out_r, labels)
        opt_r.zero_grad()
        loss_r.backward()
        real_grads = [p.grad.clone() if p.grad is not None else None 
                      for p in model_r.parameters()]
        
        opt_synth = optim.Adam([synthetic], lr=lr)
        model_s = SimpleCNN()
        out_s = model_s(synthetic[:10])
        labels_s = torch.randint(0, 10, (10,))
        loss_s = loss_fn(out_s, labels_s)
        opt_synth.zero_grad()
        loss_s.backward()
        syn_grads = [p.grad.clone() if p.grad is not None else None 
                     for p in model_s.parameters()]
        
        distil_loss = sum(torch.nn.functional.mse_loss(
            rg if rg is not None else torch.zeros_like(sg),
            sg if sg is not None else torch.zeros_like(rg)
        ) for rg, sg in zip(real_grads, syn_grads))
        
        opt.zero_grad()
        distil_loss.backward()
        opt.step()
        
        if step % 10 == 0:
            print(f"Step {step:3d}: distillation_loss={distil_loss.item():.4f}")
    
    torch.save(synthetic, "synthetic_images.pt")
    print(f"Saved {synthetic.shape[0]} synthetic images to synthetic_images.pt")

if __name__ == "__main__":
    distill()
