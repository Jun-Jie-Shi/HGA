# from utils.criterions import DiceCoef
# import logging
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModuleParallel(nn.Module):
    def __init__(self, module):
        super(ModuleParallel, self).__init__()
        self.module = module

    def forward(self, x_parallel):
        return [self.module(x) for x in x_parallel]

class GroupNormPara(nn.Module):
    def __init__(self, num_groups, num_channles, num_parallel):
        super(GroupNormPara, self).__init__()
        for i in range(num_parallel):
            setattr(self, 'norm_' + str(i), nn.GroupNorm(num_groups, num_channles))

    def forward(self, x_parallel):
        return [getattr(self, 'norm_' + str(i))(x) for i, x in enumerate(x_parallel)]

class Exchange(nn.Module):
    def __init__(self):
        super(Exchange, self).__init__()

    def forward(self, x, threshold=1e-2):
        n = x[0].shape[0]
        c = x[0].shape[1]
        out = []
        for i in range(len(x)):
            xi_out = torch.zeros_like(x[i])
            var = torch.var(x[i].view(n, c, -1), dim=2)
            xi_out[var >= threshold] = x[i][var >= threshold]

            modal_to_exchange = list(range(len(x)))
            modal_to_exchange.remove(i)
            xi_out[var < threshold] = torch.mean(
                                        torch.stack(
                                            [x[j][var < threshold] for j in modal_to_exchange], dim=0), dim=0)

            out.append(xi_out)

        return out

def conv_para(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return ModuleParallel(
            nn.Conv3d(in_channels,
                      out_channels,
                      kernel_size,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      groups=groups,
                      bias=bias))

def relu_para(inplace):
    return ModuleParallel(nn.ReLU(inplace=inplace))

def upsample_para(scale_factor, mode, align_corners):
    return ModuleParallel(nn.Upsample(scale_factor=scale_factor, mode=mode, align_corners=align_corners))

def conv_trans3d_para(in_channels, out_channnels, kernel_size, stride):
    return ModuleParallel(nn.ConvTranspose3d(in_channels, out_channnels, kernel_size, stride))

def maxpool3d_para(kernel_size):
    return ModuleParallel(nn.MaxPool3d(kernel_size))

def padding_para(x1_list, x2_list):
    out = []
    for x1, x2 in zip(x1_list, x2_list):
        out.append(padding(x1, x2))

    return out

def padding(x1, x2):
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]

    x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                    diffY // 2, diffY - diffY // 2])
    # if you have padding issues, see
    # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
    # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
    x = torch.cat([x2, x1], dim=1)

    return x


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, conv_type=nn.Conv3d, mid_channels=None, num_groups=8, num_modalities=4, exchange=False):
        super().__init__()
        self.num_modalities = num_modalities
        self.exchange = exchange
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = conv_type(in_channels, mid_channels, kernel_size=3, padding=1)
        self.norm1 = GroupNormPara(num_groups, mid_channels, num_modalities)
        if self.exchange:
            self.exchange1 = Exchange()
        self.relu1 = relu_para(inplace=True)

        self.conv2 = conv_type(mid_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = GroupNormPara(num_groups, out_channels, num_modalities)
        self.relu2 = relu_para(inplace=True)

    def forward(self, x, modality=0):
        x = self.conv1(x)
        x = self.norm1(x)
        if self.exchange:
            x = self.exchange1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        if self.exchange:
            x = self.exchange1(x)
        x = self.relu2(x)

        return x

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, conv_type=nn.Conv3d, num_modalities=4, exchange=False):
        super().__init__()
        self.pool = maxpool3d_para(2)
        self.double_conv = DoubleConv(in_channels, out_channels, conv_type=conv_type, num_modalities=num_modalities, exchange=exchange)

    def forward(self, x, modality=0):
        x = self.pool(x)
        x = self.double_conv(x, modality=modality)

        return x


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, trilinear=True, conv_type=nn.Conv3d, num_modalities=4, parallel=False, exchange=False):
        super().__init__()
        self.parallel = parallel

        # if trilinear, use the normal convolutions to reduce the number of channels
        if trilinear:
            self.up = upsample_para(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv(
                            in_channels,
                            out_channels,
                            conv_type=conv_type,
                            mid_channels=in_channels // 2,
                            num_modalities=num_modalities,
                            exchange=exchange)
        else:
            self.up = conv_trans3d_para(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(
                            in_channels,
                            out_channels,
                            conv_type=conv_type,
                            num_modalities=num_modalities,
                            exchange=exchange)

        if self.parallel:
            self.padding = padding_para
        else:
            self.padding = padding

    def forward(self, x1, x2, modality=0):
        x1 = self.up(x1)

        x = self.padding(x1, x2)

        return self.conv(x, modality=modality)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, conv_type=nn.Conv3d):
        super(OutConv, self).__init__()
        self.conv = conv_type(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
class UNetPara(nn.Module):
    def __init__(
        self,
        n_channels,
        n_classes,
        width_multiplier=1,
        trilinear=True,
        conv_type=conv_para,
        num_modalities=4,
        parallel=False,
        exchange=False,
        feature=False):
        """A simple 3D Unet, adapted from a 2D Unet from https://github.com/milesial/Pytorch-UNet/tree/master/unet
        Arguments:
          n_channels = number of input channels; 3 for RGB, 1 for grayscale input
          n_classes = number of output channels/classes
          width_multiplier = how much 'wider' your UNet should be compared with a standard UNet
                  default is 1;, meaning 32 -> 64 -> 128 -> 256 -> 512 -> 256 -> 128 -> 64 -> 32
                  higher values increase the number of kernels pay layer, by that factor
          trilinear = use trilinear interpolation to upsample; if false, 3D convtranspose layers will be used instead
          use_ds_conv = if True, we use depthwise-separable convolutional layers. in my experience, this is of little help. This
                  appears to be because with 3D data, the vast vast majority of GPU RAM is the input data/labels, not the params, so little
                  VRAM is saved by using ds_conv, and yet performance suffers."""
        super(UNetPara, self).__init__()
        _channels = (32, 64, 128, 256, 512)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.channels = [int(c*width_multiplier) for c in _channels]
        self.trilinear = trilinear
        self.conv_type = conv_type
        self.parallel = parallel
        self.feature = feature

        self.inc = DoubleConv(
            n_channels, self.channels[0], conv_type=self.conv_type, num_modalities=num_modalities, exchange=exchange)
        self.down1 = Down(self.channels[0], self.channels[1], conv_type=self.conv_type,
                          num_modalities=num_modalities, exchange=exchange)
        self.down2 = Down(self.channels[1], self.channels[2], conv_type=self.conv_type,
                          num_modalities=num_modalities, exchange=exchange)
        self.down3 = Down(self.channels[2], self.channels[3], conv_type=self.conv_type,
                          num_modalities=num_modalities, exchange=exchange)
        factor = 2 if trilinear else 1
        self.down4 = Down(self.channels[3], self.channels[4] // factor,
                          conv_type=self.conv_type, num_modalities=num_modalities, exchange=exchange)
        self.up1 = Up(self.channels[4], self.channels[3] // factor, trilinear, conv_type=self.conv_type,
                      num_modalities=num_modalities, parallel=parallel, exchange=exchange)
        self.up2 = Up(self.channels[3], self.channels[2] // factor, trilinear, conv_type=self.conv_type,
                      num_modalities=num_modalities, parallel=parallel, exchange=exchange)
        self.up3 = Up(self.channels[2], self.channels[1] // factor, trilinear, conv_type=self.conv_type,
                      num_modalities=num_modalities, parallel=parallel, exchange=exchange)
        self.up4 = Up(self.channels[1], self.channels[0], trilinear, conv_type=self.conv_type,
                      num_modalities=num_modalities, parallel=parallel, exchange=exchange)
        self.outc = OutConv(
            self.channels[0], n_classes, conv_type=self.conv_type)

    def forward(self, x: list, modality=0):
        # x a list of modalities
        x1 = self.inc(x, modality=modality)
        x2 = self.down1(x1, modality=modality)
        x3 = self.down2(x2, modality=modality)
        x4 = self.down3(x3, modality=modality)
        x5 = self.down4(x4, modality=modality)

        x = self.up1(x5, x4, modality=modality)
        x = self.up2(x, x3, modality=modality)
        x = self.up3(x, x2, modality=modality)
        x = self.up4(x, x1, modality=modality)
        logits = self.outc(x)

        if self.feature and self.training:
            return (logits, x5)
        return (logits, x)


class DepthwiseSeparableConv3d(nn.Module):
    def __init__(self, nin, nout, kernel_size, padding, kernels_per_layer=1):
        super(DepthwiseSeparableConv3d, self).__init__()
        self.depthwise = nn.Conv3d(
            nin, nin * kernels_per_layer, kernel_size=kernel_size, padding=padding, groups=nin)
        self.pointwise = nn.Conv3d(
            nin * kernels_per_layer, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


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
        #         self.module = UNetPara(1, out_channels, width_multiplier=width_ratio)
        #     elif not sharing:
        #         self.module = nn.ModuleList()
        #         for ii in range(in_channels):
        #             self.module.append(UNetPara(1, out_channels, width_multiplier=width_ratio))
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
