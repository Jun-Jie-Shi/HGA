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
        feature_size=12,
        use_checkpoint=False):
        super(swinunetr, self).__init__()
        self.final_nonlin = softmax_helper
        self.modality_num = modality_num
        self.num_classes = num_classes
        self.is_training = False
        self.modality_specific_models = []

        for i in range(modality_num):
            self.modality_specific_models.append(SwinUNETR(
                img_size=(96,96,96),
                in_channels=1,
                out_channels=feature_size,
                feature_size=feature_size,
                use_checkpoint=use_checkpoint)
            )

        self.masker = MaskModal()
        self.output = nn.Conv3d(feature_size, num_classes, 1, 1, 0, 1, 1, False)
        self.modality_specific_models = nn.ModuleList(self.modality_specific_models)

    def forward(self, x, mask):
        x = [x[:, i:i + 1] for i in range(self.modality_num)]

        modality_features = []
        # final_outputs = []

        for i in range(self.modality_num):
            ms_outputs = self.modality_specific_models[i](x[i])
            modality_features.append(ms_outputs)

        out = [self.output(x_i) for ii, x_i in enumerate(modality_features)]

        preserved = list(range(self.modality_num))
        for num in range(self.modality_num):
            if mask[0,num] == False:
                preserved.remove(num)
        out_all = torch.stack(out, dim=0)
        out_mul = torch.mean(out_all[preserved], dim=0)
        if self.is_training:
            de_fe_all = torch.stack(modality_features, dim=0)
            de_fe_mul = torch.mean(de_fe_all[preserved], dim=0)

            out.append(out_mul)
            modality_features.append(de_fe_mul)

            return out, modality_features
        else:
            # return output
            return self.final_nonlin(out_mul)

