import os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dm import ConvNet, dsa, random_subset


def test_siamese_augmentation_is_shared():
    x = torch.randn(4, 3, 32, 32)
    g = torch.Generator().manual_seed(0)
    state = g.get_state()
    a = dsa(x, g)
    g.set_state(state)
    b = dsa(x[:2], g)
    assert torch.allclose(a[:2], b, atol=1e-6)


def test_random_subset_is_class_balanced():
    by_class = [torch.full((20, 3, 32, 32), float(c)) for c in range(10)]
    x, y = random_subset(by_class, 3, torch.Generator().manual_seed(0))
    assert x.shape == (30, 3, 32, 32)
    assert torch.equal(torch.bincount(y), torch.full((10,), 3))
    assert torch.equal(x[:, 0, 0, 0], y.float())  # each image carries its own class label


def test_convnet_shapes():
    net = ConvNet()
    x = torch.randn(2, 3, 32, 32)
    assert net.features(x).shape == (2, 2048)
    assert net(x).shape == (2, 10)
