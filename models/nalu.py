import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from .nac import NAC
from torch.nn.parameter import Parameter


class NALU(nn.Module):
    """A Neural Arithmetic Logic Unit [1].

    NALU uses 2 NACs with tied weights to support
    multiplicative extrapolation.

    Attributes:
        in_features: size of the input sample.
        out_features: size of the output sample.

    Sources:
        [1]: https://arxiv.org/abs/1808.00508
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = 1e-10

        self.G = Parameter(torch.Tensor(out_features, in_features))
        self.W = Parameter(torch.Tensor(out_features, in_features))
        self.register_parameter('bias', None)
        self.nac = NAC(in_features, out_features)

        init.kaiming_uniform_(self.G, a=math.sqrt(5))
        init.kaiming_uniform_(self.W, a=math.sqrt(5))

    def forward(self, input):
        a = self.nac(input)
        g = F.sigmoid(F.linear(input, self.G, self.bias))
        add_sub = g * a
        log_input = torch.log(torch.abs(input) + self.eps)
        m = torch.exp(F.linear(log_input, self.W, self.bias))
        mul_div = (1 - g) * m
        y = add_sub + mul_div
        return y

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )
