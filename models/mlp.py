import math
import torch.nn as nn

from .utils import str2act


class MLP(nn.Module):
    def __init__(self, activation, input_dim=1, encoding_dim=8):
        super().__init__()
        self.activation = str2act(activation)

        self.i2h = nn.Linear(input_dim, encoding_dim)
        self.h2h1 = nn.Linear(encoding_dim, encoding_dim)
        self.h2h2 = nn.Linear(encoding_dim, encoding_dim)
        self.h2h3 = nn.Linear(encoding_dim, encoding_dim)
        self.h2o = nn.Linear(encoding_dim, input_dim)

        # init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x):
        out = self.activation(self.i2h(x))
        out = self.activation(self.h2h1(out))
        out = self.activation(self.h2h2(out))
        out = self.activation(self.h2h3(out))
        out = self.h2o(out)
        return out
