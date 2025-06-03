# models.py
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, noise_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        dims = [noise_dim] + hidden_dims + [output_dim]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.LeakyReLU(0.2))
            else:
                layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims + [1]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.LeakyReLU(0.2))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)