import torch
import torch.nn as nn
from .cca import CrissCrossAttention

class RCCA(nn.Module):
    def __init__(self, in_ch, inter_ch, R=2):
        super().__init__()

        self.R = R
        self.cc = CrissCrossAttention(in_ch, inter_ch)

    def forward(self, x):
        for _ in range(self.R):
            x = self.cc(x)
        return x
