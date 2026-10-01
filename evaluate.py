#!/usr/bin/env python3
"""Evaluate model trained on synthetic vs real data."""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import sys

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
    test = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    return (DataLoader(train, batch_size=64, shuffle=True, num_workers=1),
            DataLoader(test, batch_size=64, num_workers=1))

def train(model, loader, epochs=5, lr=0.01):
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        total_loss, n = 0.0, 0
        for x, y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
    return total_loss / n

def evaluate(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

def main():
    try:
        synth_imgs = torch.load("synthetic_images.pt")
    except FileNotFoundError:
        print("Error: synthetic_images.pt not found. Run distill.py first.")
        sys.exit(1)

    train_loader, test_loader = load_cifar10()
    synth_dataset = TensorDataset(
        synth_imgs, torch.randint(0, 10, (synth_imgs.shape[0],))
    )
    synth_loader = DataLoader(synth_dataset, batch_size=64, shuffle=True)

    print("=" * 50)
    print("Training CNN on Real CIFAR-10...")
    print("=" * 50)
    model_real = SimpleCNN()
    train(model_real, train_loader, epochs=5)
    acc_real = evaluate(model_real, test_loader)
    print(f"Real data test accuracy: {acc_real * 100:.2f}%")

    print("=" * 50)
    print("Training CNN on Synthetic Images...")
    print("=" * 50)
    model_syn = SimpleCNN()
    train(model_syn, synth_loader, epochs=5)
    acc_syn = evaluate(model_syn, test_loader)
    print(f"Synthetic data test accuracy: {acc_syn * 100:.2f}%")

    print("=" * 50)
    print(f"Accuracy drop: {(acc_real - acc_syn) * 100:.2f}%")
    print("=" * 50)

if __name__ == "__main__":
    main()
