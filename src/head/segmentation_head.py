import torch
import torch.nn as nn

class SegmentationHead(nn.Module):
    def __init__(self, in_ch, num_classes):
        super().__init__()

        self.classifier = nn.Conv2d(in_ch, num_classes, 1)

    def forward(self, x):
        return self.classifier(x)
