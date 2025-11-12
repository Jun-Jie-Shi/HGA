from monai.networks.nets import SwinUNETR
from torch import nn
import torch
import numpy as np
import torch.nn.functional as F


softmax_helper = lambda x: F.softmax(x, 1)

class MaskModal(nn.Module):
    def __init__(self):
        super(MaskModal, self).__init__()
    
    def forward(self, x, mask):
        B, K, H, W, Z = x.size()
        y = torch.zeros_like(x)
        y[mask, ...] = x[mask, ...]
        x = y.view(B, -1, H, W, Z)
        return x

class swinunetr(nn.Module):
    def __init__(self, modality_num=4,
        num_classes=4,
        feature_size=48,
        use_checkpoint=False):
        super(swinunetr, self).__init__()
        self.final_nonlin = softmax_helper
        self.modality_num = modality_num
        self.num_classes = num_classes
        self.is_training = False

        self.modality_specific_models = SwinUNETR(
        img_size=(96,96,96),
        in_channels=modality_num,
        out_channels=num_classes,
        feature_size=feature_size,
        use_checkpoint=use_checkpoint)

        self.masker = MaskModal()

    def forward(self, x, mask):
        x = self.masker(x, mask)
        ms_outputs = self.modality_specific_models(x)
        output = self.final_nonlin(ms_outputs)
        return output
