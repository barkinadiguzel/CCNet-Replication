import torch
import torch.nn as nn

class FusionBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch * 2, in_ch, 1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        return self.fuse(x)
