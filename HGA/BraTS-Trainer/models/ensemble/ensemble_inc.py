import torch
import torch.nn as nn
from models.ensemble.unet3d_parallel import UNet as UNetPara
# from modeling.unet3d.unet3d import UNet
from utils.criterions import DiceCoef
# import logging


class Ensemble(nn.Module):
    def __init__(self, in_channels, out_channels,
                 output='list', exchange=False, feature=False, modality_specific_norm=True, width_ratio=1., sharing=True, **kwargs):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output = output
        self.feature = feature
        self.modality_specific_norm = modality_specific_norm
        self.width_ratio = width_ratio
        self.sharing = sharing
        if self.modality_specific_norm and sharing:
            self.module = UNetPara(1, out_channels, num_modalities=in_channels, parallel=True,
                            exchange=exchange, feature=feature, width_multiplier=width_ratio)
        else:
            raise NotImplementedError()
        # else:
        #     if sharing:
        #         self.module = UNet(1, out_channels, width_multiplier=width_ratio)
        #     elif not sharing:
        #         self.module = nn.ModuleList()
        #         for ii in range(in_channels):
        #             self.module.append(UNet(1, out_channels, width_multiplier=width_ratio))
        #     else:
        #         raise NotImplementedError()
        self.is_training = False
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight) #

        # self.weights = nn.parameter.Parameter(torch.ones(in_channels) / in_channels, requires_grad=True)

    def forward(self, x, mask, target=None, weights=None):
        x = [x[:, i:i + 1] for i in range(self.in_channels)]

        if self.modality_specific_norm:
            (out, de_fe) = self.module(x)
        else:
            if self.sharing:
                out = [self.module(x_i) for x_i in x]
            else:
                out = [self.module[ii](x_i) for ii, x_i in enumerate(x)]

        if self.output == 'list' and self.is_training:
            preserved = list(range(self.in_channels))
            # loss = []
            # out_ava = []
            for num in range(self.in_channels):
                if mask[0,num] == False:
                    preserved.remove(num)
            out_all = torch.stack(out, dim=0)
            out_mul = torch.mean(out_all[preserved], dim=0)
            # logging.info(out_mul.size())
            de_fe_all = torch.stack(de_fe, dim=0)
            de_fe_mul = torch.mean(de_fe_all[preserved], dim=0)

            out.append(out_mul)
            de_fe.append(de_fe_mul)

            # logging.info(out)

            # loss.append(DiceCoef(out_mul, target, num_cls=self.out_channels))
            return out, de_fe

        out = torch.stack(out, dim=0)
        preserved = list(range(self.in_channels))
        for num in range(self.in_channels):
            if mask[0,num] == False:
                preserved.remove(num)

        if weights is None:
            return nn.Softmax(dim=1)(torch.mean(out[preserved], dim=0))
        else:
            w = weights[preserved] / weights[preserved].sum()

            return nn.Softmax(dim=1)(torch.einsum('mncwhd,m->ncwhd', out[preserved], w))

    def shared_module_zero_grad(self,):
        for module in self.shared_modules:
            module.zero_grad()
