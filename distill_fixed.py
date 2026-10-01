#!/usr/bin/env python3
"""Gradient matching dataset distillation for CIFAR-10 (fixed version)."""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    return DataLoader(train, batch_size=5000, shuffle=True, num_workers=2)

def train_cifar10():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                             (0.2470, 0.2435, 0.2616))
    ])
    train = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    return DataLoader(train, batch_size=64, shuffle=True, num_workers=2)

def test_cifar10():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                             (0.2470, 0.2435, 0.2616))
    ])
    test = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    return DataLoader(test, batch_size=64, num_workers=2)

def distill(n_steps=250, lr=0.01, per_class=10):
    real_loader = load_cifar10()
    
    # Initialize synthetic images as leaf tensors with requires_grad=True
    synthetic = torch.randn(per_class*10, 3, 32, 32, device=device, requires_grad=True)
    
    opt = torch.optim.Adam([synthetic], lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    
    for step in range(n_steps):
        # Get real batch
        real_batch, _ = next(iter(real_loader))
        real_batch = real_batch.to(device)
        real_labels = torch.randint(0, 10, (real_batch.size(0),), device=device)
        
        # Compute real gradients (w.r.t model params)
        model_r = SimpleCNN().to(device)
        out_r = model_r(real_batch)
        loss_r = loss_fn(out_r, real_labels)
        model_r.zero_grad()
        loss_r.backward()
        real_grads = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in model_r.parameters()]
        
        # Compute synthetic gradients (w.r.t model params)
        model_s = SimpleCNN().to(device)
        out_s = model_s(synthetic)
        syn_labels = torch.randint(0, 10, (synthetic.size(0),), device=device)
        loss_s = loss_fn(out_s, syn_labels)
        model_s.zero_grad()
        loss_s.backward()
        
        # Compute distil_loss without cloning - use .detach() on real_grads
        # but NOT on syn_grads so they remain connected to synthetic
        distil_loss = sum(torch.nn.functional.mse_loss(rg.detach(), sg) 
                          for rg, sg in zip(real_grads, model_s.parameters()))
        
        # Now distil_loss has a path to synthetic through model_s's weights
        # and through the forward pass of synthetic
        model_s.zero_grad()
        distil_loss.backward()
        opt.step()
        
        if step % 50 == 0:
            print(f"Step {step:3d}: distillation_loss={distil_loss.item():.4f}")
    
    torch.save(synthetic.cpu(), "synthetic_images.pt")
    print(f"Saved {synthetic.shape[0]} synthetic images to synthetic_images.pt")

def train_and_evaluate(loader, epochs=5, lr=0.01):
    model = SimpleCNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_cifar10():
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

def main():
    distill(n_steps=250, lr=0.01, per_class=10)
    
    synth_imgs = torch.load("synthetic_images.pt")
    
    # Train on synthetic
    synth_dataset = TensorDataset(synth_imgs, torch.randint(0, 10, (synth_imgs.shape[0],)))
    synth_loader = DataLoader(synth_dataset, batch_size=64, shuffle=True)
    acc_syn = train_and_evaluate(synth_loader)
    
    # Train on 100 random real images
    train_loader = train_cifar10()
    batch = next(iter(train_loader))
    x, y = batch[0][:100], batch[1][:100]
    random_dataset = TensorDataset(x, y)
    random_loader = DataLoader(random_dataset, batch_size=64, shuffle=True)
    acc_real = train_and_evaluate(random_loader)
    
    # Write results
    results = {"synthetic_acc": acc_syn, "random_real_acc": acc_real}
    with open("results.json", "w") as f:
        json.dump(results, f)
    print(f"Synthetic accuracy: {acc_syn * 100:.2f}%")
    print(f"Random real accuracy: {acc_real * 100:.2f}%")

if __name__ == "__main__":
    main()
