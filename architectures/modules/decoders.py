import torch
import torch.nn as nn
import numpy as np
from .base import BaseINR

class MLP(BaseINR):
    def __init__(self, in_features=1, out_features=1, hidden_features=256, num_layers=4):
        super().__init__()
        layers = []
        last = in_features
        for _ in range(num_layers):
            layers.append(nn.Linear(last, hidden_features))
            layers.append(nn.ReLU())
            last = hidden_features
        layers.append(nn.Linear(last, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class SineLayer(nn.Module):
    def __init__(self, in_f, out_f, omega_0=30.0, is_first=False):
        super().__init__()
        self.linear = nn.Linear(in_f, out_f)
        self.omega_0 = omega_0
        self.is_first = is_first
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1/self.linear.in_features,
                                            1/self.linear.in_features)
            else:
                bound = np.sqrt(6/self.linear.in_features)/self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

class SIREN(BaseINR):
    def __init__(self, in_features=1, out_features=1,
                 hidden_features=256, num_layers=4):
        super().__init__()
        layers = [SineLayer(in_features, hidden_features, is_first=True)]
        for _ in range(num_layers - 1):
            layers.append(SineLayer(hidden_features, hidden_features))
        layers.append(nn.Linear(hidden_features, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)