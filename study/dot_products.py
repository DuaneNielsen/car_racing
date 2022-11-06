import torch
from torch import sin, cos, pi
from matplotlib import pyplot as plt

basis = torch.tensor([
    [1., 0.],
    [0., 1.]
])


angles = torch.stack([
    cos(torch.linspace(0, pi, steps=8)),
    sin(torch.linspace(0, pi, steps=8))
])

plt.plot(angles[0], angles[1])
plt.show()

dot = cos(torch.linspace(0, pi, 8))

