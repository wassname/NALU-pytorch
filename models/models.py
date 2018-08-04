import math
import torch
import torch.nn as nn

from .nac import NAC
from .nalu import NALU
from .utils import str2act


class MultiLayerNet(nn.Module):
    def __init__(self, activation, num_layers, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.activation = str2act(activation)

        layers = []
        if self.activation is not None:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                self.activation,
            ])
        else:
            layers.append(nn.Linear(in_dim, hidden_dim))
        for i in range(num_layers - 2):
            if self.activation is not None:
                layers.extend([
                    nn.Linear(hidden_dim, hidden_dim),
                    self.activation,
                ])
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.Linear(hidden_dim, out_dim))

        self.model = nn.Sequential(*layers)

        # init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x):
        out = self.model(x)
        return out


class MultiLayerNAC(nn.Module):
    def __init__(self, num_layers, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        layers = []
        layers.append(NAC(in_dim, hidden_dim))
        for i in range(num_layers - 2):
            layers.append(NAC(hidden_dim, hidden_dim))
        layers.append(NAC(hidden_dim, out_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        return out


class MultiLayerNALU(nn.Module):
    def __init__(self, num_layers, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        layers = []
        layers.append(NALU(in_dim, hidden_dim))
        for i in range(num_layers - 2):
            layers.append(NALU(hidden_dim, hidden_dim))
        layers.append(NALU(hidden_dim, out_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        return out
