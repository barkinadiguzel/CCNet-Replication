import torch
import torch.nn as nn

from ..backbone.resnet_dilated import ResNetDilated
from ..blocks.conv_block import ConvBlock
from ..blocks.rcca import RCCA
from ..blocks.fusion import FusionBlock
from ..head.segmentation_head import SegmentationHead

class CCNet(nn.Module):
    def __init__(self, num_classes, inter_ch=64):
        super().__init__()

        self.backbone = ResNetDilated()

        self.reduce = ConvBlock(2048, inter_ch)

        self.rcca = RCCA(inter_ch, inter_ch // 2, R=2)

        self.fusion = FusionBlock(inter_ch)

        self.head = SegmentationHead(inter_ch, num_classes)

    def forward(self, x):
        feat = self.backbone(x)

        local = self.reduce(feat)

        context = self.rcca(local)

        fused = self.fusion(local, context)

        out = self.head(fused)

        return out
