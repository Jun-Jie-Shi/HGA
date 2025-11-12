import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import fusion_prenorm_2d, general_conv3d_prenorm_2d
from torch.nn.init import constant_, xavier_uniform_
from utils.criterions import temp_kl_loss, softmax_weighted_loss, dice_loss, prototype_passion_loss, prototype_kl_loss

# from visualizer import get_local

basic_dims = 8
### 原论文模型维度
# basic_dims = 16
### 相同基线下模型维度
transformer_basic_dims = 512
mlp_dim = 4096
num_heads = 8
depth = 1
num_modals = 3
num_cls = 6
# patch_size = 5
H = W = 256
Z = 1

### 与原论文中模型输入128*128*128输入相比，对比方法采用80*80*80输入
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.e1_c1 = nn.Conv3d(in_channels=1, out_channels=basic_dims, kernel_size=(3,3,1), stride=1, padding=(1,1,0), padding_mode='reflect', bias=True)
        self.e1_c2 = general_conv3d_prenorm_2d(basic_dims, basic_dims, pad_type='reflect')
        self.e1_c3 = general_conv3d_prenorm_2d(basic_dims, basic_dims, pad_type='reflect')

        self.e2_c1 = general_conv3d_prenorm_2d(basic_dims, basic_dims*2, stride=(2,2,1), pad_type='reflect')
        self.e2_c2 = general_conv3d_prenorm_2d(basic_dims*2, basic_dims*2, pad_type='reflect')
        self.e2_c3 = general_conv3d_prenorm_2d(basic_dims*2, basic_dims*2, pad_type='reflect')

        self.e3_c1 = general_conv3d_prenorm_2d(basic_dims*2, basic_dims*4, stride=(2,2,1), pad_type='reflect')
        self.e3_c2 = general_conv3d_prenorm_2d(basic_dims*4, basic_dims*4, pad_type='reflect')
        self.e3_c3 = general_conv3d_prenorm_2d(basic_dims*4, basic_dims*4, pad_type='reflect')

        self.e4_c1 = general_conv3d_prenorm_2d(basic_dims*4, basic_dims*8, stride=(2,2,1), pad_type='reflect')
        self.e4_c2 = general_conv3d_prenorm_2d(basic_dims*8, basic_dims*8, pad_type='reflect')
        self.e4_c3 = general_conv3d_prenorm_2d(basic_dims*8, basic_dims*8, pad_type='reflect')

        self.e5_c1 = general_conv3d_prenorm_2d(basic_dims*8, basic_dims*16, stride=(2,2,1), pad_type='reflect')
        self.e5_c2 = general_conv3d_prenorm_2d(basic_dims*16, basic_dims*16, pad_type='reflect')
        self.e5_c3 = general_conv3d_prenorm_2d(basic_dims*16, basic_dims*16, pad_type='reflect')

    def forward(self, x):
        x1 = self.e1_c1(x)
        x1 = x1 + self.e1_c3(self.e1_c2(x1))

        x2 = self.e2_c1(x1)
        x2 = x2 + self.e2_c3(self.e2_c2(x2))

        x3 = self.e3_c1(x2)
        x3 = x3 + self.e3_c3(self.e3_c2(x3))

        x4 = self.e4_c1(x3)
        x4 = x4 + self.e4_c3(self.e4_c2(x4))

        x5 = self.e5_c1(x4)
        x5 = x5 + self.e5_c3(self.e5_c2(x5))

        return x1, x2, x3, x4, x5

class Decoder_sep(nn.Module):
    def __init__(self, num_cls=4):
        super(Decoder_sep, self).__init__()

        self.d4 = nn.Upsample(scale_factor=(2,2,1), mode='trilinear', align_corners=True)
        self.d4_c1 = general_conv3d_prenorm_2d(basic_dims*16, basic_dims*8, pad_type='reflect')
        self.d4_c2 = general_conv3d_prenorm_2d(basic_dims*16, basic_dims*8, pad_type='reflect')
        self.d4_out = general_conv3d_prenorm_2d(basic_dims*8, basic_dims*8, k_size=1, padding=0, pad_type='reflect')

        self.d3 = nn.Upsample(scale_factor=(2,2,1), mode='trilinear', align_corners=True)
        self.d3_c1 = general_conv3d_prenorm_2d(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d3_c2 = general_conv3d_prenorm_2d(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d3_out = general_conv3d_prenorm_2d(basic_dims*4, basic_dims*4, k_size=1, padding=0, pad_type='reflect')

        self.d2 = nn.Upsample(scale_factor=(2,2,1), mode='trilinear', align_corners=True)
        self.d2_c1 = general_conv3d_prenorm_2d(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d2_c2 = general_conv3d_prenorm_2d(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d2_out = general_conv3d_prenorm_2d(basic_dims*2, basic_dims*2, k_size=1, padding=0, pad_type='reflect')

        self.d1 = nn.Upsample(scale_factor=(2,2,1), mode='trilinear', align_corners=True)
        self.d1_c1 = general_conv3d_prenorm_2d(basic_dims*2, basic_dims, pad_type='reflect')
        self.d1_c2 = general_conv3d_prenorm_2d(basic_dims*2, basic_dims, pad_type='reflect')
        self.d1_out = general_conv3d_prenorm_2d(basic_dims, basic_dims, k_size=1, padding=0, pad_type='reflect')

        self.seg_layer = nn.Conv3d(in_channels=basic_dims, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2, x3, x4, x5):
        de_x5 = self.d4_c1(self.d4(x5))

        cat_x4 = torch.cat((de_x5, x4), dim=1)
        de_x4 = self.d4_out(self.d4_c2(cat_x4))
        de_x4 = self.d3_c1(self.d3(de_x4))

        cat_x3 = torch.cat((de_x4, x3), dim=1)
        de_x3 = self.d3_out(self.d3_c2(cat_x3))
        de_x3 = self.d2_c1(self.d2(de_x3))

        cat_x2 = torch.cat((de_x3, x2), dim=1)
        de_x2 = self.d2_out(self.d2_c2(cat_x2))
        de_x2 = self.d1_c1(self.d1(de_x2))

        cat_x1 = torch.cat((de_x2, x1), dim=1)
        de_x1 = self.d1_out(self.d1_c2(cat_x1))

        logits = self.seg_layer(de_x1)
        pred = self.softmax(logits)

        return pred

class Decoder_fuse(nn.Module):
    def __init__(self, num_cls=6):
        super(Decoder_fuse, self).__init__()

        self.d4_c1 = general_conv3d_prenorm_2d(basic_dims*16, basic_dims*8, pad_type='reflect')
        self.d4_c2 = general_conv3d_prenorm_2d(basic_dims*16, basic_dims*8, pad_type='reflect')
        self.d4_out = general_conv3d_prenorm_2d(basic_dims*8, basic_dims*8, k_size=1, padding=0, pad_type='reflect')

        self.d3_c1 = general_conv3d_prenorm_2d(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d3_c2 = general_conv3d_prenorm_2d(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d3_out = general_conv3d_prenorm_2d(basic_dims*4, basic_dims*4, k_size=1, padding=0, pad_type='reflect')

        self.d2_c1 = general_conv3d_prenorm_2d(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d2_c2 = general_conv3d_prenorm_2d(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d2_out = general_conv3d_prenorm_2d(basic_dims*2, basic_dims*2, k_size=1, padding=0, pad_type='reflect')

        self.d1_c1 = general_conv3d_prenorm_2d(basic_dims*2, basic_dims, pad_type='reflect')
        self.d1_c2 = general_conv3d_prenorm_2d(basic_dims*2, basic_dims, pad_type='reflect')
        self.d1_out = general_conv3d_prenorm_2d(basic_dims, basic_dims, k_size=1, padding=0, pad_type='reflect')

        self.seg_d4 = nn.Conv3d(in_channels=basic_dims*16, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_d3 = nn.Conv3d(in_channels=basic_dims*8, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_d2 = nn.Conv3d(in_channels=basic_dims*4, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_d1 = nn.Conv3d(in_channels=basic_dims*2, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_layer = nn.Conv3d(in_channels=basic_dims, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        # self.softmax = nn.Softmax(dim=1)

        self.up2 = nn.Upsample(scale_factor=(2,2,1), mode='trilinear', align_corners=True)
        # self.up4 = nn.Upsample(scale_factor=(4,4,1), mode='trilinear', align_corners=True)
        # self.up8 = nn.Upsample(scale_factor=(8,8,1), mode='trilinear', align_corners=True)
        # self.up16 = nn.Upsample(scale_factor=(16,16,1), mode='trilinear', align_corners=True)

        self.RFM5 = fusion_prenorm_2d(in_channel=basic_dims*16, num_modals=num_modals)
        self.RFM4 = fusion_prenorm_2d(in_channel=basic_dims*8, num_modals=num_modals)
        self.RFM3 = fusion_prenorm_2d(in_channel=basic_dims*4, num_modals=num_modals)
        self.RFM2 = fusion_prenorm_2d(in_channel=basic_dims*2, num_modals=num_modals)
        self.RFM1 = fusion_prenorm_2d(in_channel=basic_dims*1, num_modals=num_modals)


    def forward(self, x1, x2, x3, x4, x5, mask=None):
        de_x5_f = self.RFM5(x5)
        # pred4 = self.softmax(self.seg_d4(de_x5_f))
        pred4 = self.seg_d4(de_x5_f)
        de_x5 = self.d4_c1(self.up2(de_x5_f))

        de_x4 = self.RFM4(x4)
        de_x4 = torch.cat((de_x4, de_x5), dim=1)
        de_x4_f = self.d4_out(self.d4_c2(de_x4))
        # pred3 = self.softmax(self.seg_d3(de_x4))
        pred3 = self.seg_d3(de_x4_f)
        de_x4 = self.d3_c1(self.up2(de_x4_f))

        de_x3 = self.RFM3(x3)
        de_x3 = torch.cat((de_x3, de_x4), dim=1)
        de_x3_f = self.d3_out(self.d3_c2(de_x3))
        # pred2 = self.softmax(self.seg_d2(de_x3))
        pred2 = self.seg_d2(de_x3_f)
        de_x3 = self.d2_c1(self.up2(de_x3_f))

        de_x2 = self.RFM2(x2)
        de_x2 = torch.cat((de_x2, de_x3), dim=1)
        de_x2_f = self.d2_out(self.d2_c2(de_x2))
        pred1 = self.seg_d1(de_x2_f)
        de_x2 = self.d1_c1(self.up2(de_x2_f))

        de_x1 = self.RFM1(x1)
        de_x1 = torch.cat((de_x1, de_x2), dim=1)
        de_x1_f = self.d1_out(self.d1_c2(de_x1))

        logits = self.seg_layer(de_x1_f)
        # pred = self.softmax(logits)
        pred = logits

        return pred, (pred1, pred2, pred3, pred4), (de_x1_f, de_x2_f, de_x3_f, de_x4_f, de_x5_f)


class SelfAttention(nn.Module):
    def __init__(
        self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0
    ):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    # @get_local('attn')
    def forward(self, x):
        B, N, C = x.shape
        # input = x
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        # output = x
        # div = output/input
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class PreNormDrop(nn.Module):
    def __init__(self, dim, dropout_rate, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn

    def forward(self, x):
        return self.dropout(self.fn(self.norm(x)))


class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return F.gelu(x)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        return self.net(x)


class Transformer(nn.Module):
    def __init__(self, embedding_dim, depth, heads, mlp_dim, dropout_rate=0.1, n_levels=1, n_points=4):
        super(Transformer, self).__init__()
        self.cross_attention_list = []
        self.cross_ffn_list = []
        self.depth = depth
        for j in range(self.depth):
            self.cross_attention_list.append(
                Residual(
                    PreNormDrop(
                        embedding_dim,
                        dropout_rate,
                        SelfAttention(embedding_dim, heads=heads, dropout_rate=dropout_rate),
                    )
                )
            )
            self.cross_ffn_list.append(
                Residual(
                    PreNorm(embedding_dim, FeedForward(embedding_dim, mlp_dim, dropout_rate))
                )
            )

        self.cross_attention_list = nn.ModuleList(self.cross_attention_list)
        self.cross_ffn_list = nn.ModuleList(self.cross_ffn_list)


    def forward(self, x, pos):
        for j in range(self.depth):
            x = x + pos
            x = self.cross_attention_list[j](x)
            x = self.cross_ffn_list[j](x)
        return x


class MaskModal(nn.Module):
    def __init__(self):
        super(MaskModal, self).__init__()
    
    def forward(self, x, mask):
        B, K, C, H, W, Z = x.size()
        y = torch.zeros_like(x)
        y[mask, ...] = x[mask, ...]
        x = y.view(B, -1, H, W, Z)
        return x


class Model(nn.Module):
    def __init__(self, num_cls=6):
        super(Model, self).__init__()
        self.bssfp_encoder = Encoder()
        self.lge_encoder = Encoder()
        self.t2_encoder = Encoder()

        ########### IntraFormer
        self.bssfp_encode_conv = nn.Conv3d(basic_dims*16, transformer_basic_dims, kernel_size=1, stride=1, padding=0)
        self.lge_encode_conv = nn.Conv3d(basic_dims*16, transformer_basic_dims, kernel_size=1, stride=1, padding=0)
        self.t2_encode_conv = nn.Conv3d(basic_dims*16, transformer_basic_dims, kernel_size=1, stride=1, padding=0)

        self.bssfp_pos = nn.Parameter(torch.zeros(1, (H//16)*(W//16)*Z, transformer_basic_dims))
        self.lge_pos = nn.Parameter(torch.zeros(1, (H//16)*(W//16)*Z, transformer_basic_dims))
        self.t2_pos = nn.Parameter(torch.zeros(1, (H//16)*(W//16)*Z, transformer_basic_dims))

        self.bssfp_transformer = Transformer(embedding_dim=transformer_basic_dims, depth=depth, heads=num_heads, mlp_dim=mlp_dim)
        self.lge_transformer = Transformer(embedding_dim=transformer_basic_dims, depth=depth, heads=num_heads, mlp_dim=mlp_dim)
        self.t2_transformer = Transformer(embedding_dim=transformer_basic_dims, depth=depth, heads=num_heads, mlp_dim=mlp_dim)
        ########### IntraFormer

        ########### InterFormer
        self.multimodal_transformer = Transformer(embedding_dim=transformer_basic_dims, depth=depth, heads=num_heads, mlp_dim=mlp_dim, n_levels=num_modals)
        self.multimodal_decode_conv = nn.Conv3d(transformer_basic_dims*num_modals, basic_dims*16*num_modals, kernel_size=1, padding=0)
        ########### InterFormer

        self.masker = MaskModal()
        self.mask_type = 'nosplit_longtail'
        self.zeros_x1 = torch.zeros(1,basic_dims,H,W,Z).detach()
        self.zeros_x2 = torch.zeros(1,basic_dims*2,H//2,W//2,Z).detach()
        self.zeros_x3 = torch.zeros(1,basic_dims*4,H//4,W//4,Z).detach()
        self.zeros_x4 = torch.zeros(1,basic_dims*8,H//8,W//8,Z).detach()
        self.zeros_x5 = torch.zeros(1,basic_dims*16,H//16,W//16,Z).detach()
        self.zeros_intra = torch.zeros(1,transformer_basic_dims,H//16,W//16,Z).detach()

        self.decoder_fuse = Decoder_fuse(num_cls=num_cls)
        # self.decoder_sep = Decoder_sep(num_cls=num_cls)

        self.up2 = nn.Upsample(scale_factor=(2,2,1), mode='trilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=(4,4,1), mode='trilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=(8,8,1), mode='trilinear', align_corners=True)
        self.up16 = nn.Upsample(scale_factor=(16,16,1), mode='trilinear', align_corners=True)
        self.up_ops = nn.ModuleList([self.up2, self.up4, self.up8, self.up16])

        self.is_training = False

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight) #

    def forward(self, x, mask, target=None, temp=1.0):
        if self.mask_type == 'random_all':
            #extract feature from different layers
            bssfp_x1, bssfp_x2, bssfp_x3, bssfp_x4, bssfp_x5 = self.bssfp_encoder(x[:, 0:1, :, :, :])
            lge_x1, lge_x2, lge_x3, lge_x4, lge_x5 = self.lge_encoder(x[:, 1:2, :, :, :])
            t2_x1, t2_x2, t2_x3, t2_x4, t2_x5 = self.t2_encoder(x[:, 2:3, :, :, :])

            ########### IntraFormer
            bssfp_token_x5 = self.bssfp_encode_conv(bssfp_x5).permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims)
            lge_token_x5 = self.lge_encode_conv(lge_x5).permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims)
            t2_token_x5 = self.t2_encode_conv(t2_x5).permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims)

            bssfp_intra_token_x5 = self.bssfp_transformer(bssfp_token_x5, self.bssfp_pos)
            lge_intra_token_x5 = self.lge_transformer(lge_token_x5, self.lge_pos)
            t2_intra_token_x5 = self.t2_transformer(t2_token_x5, self.t2_pos)

            bssfp_intra_x5 = bssfp_intra_token_x5.view(x.size(0), (H//16),(W//16),Z, transformer_basic_dims).permute(0, 4, 1, 2, 3).contiguous()
            lge_intra_x5 = lge_intra_token_x5.view(x.size(0), (H//16),(W//16),Z, transformer_basic_dims).permute(0, 4, 1, 2, 3).contiguous()
            t2_intra_x5 = t2_intra_token_x5.view(x.size(0), (H//16),(W//16),Z, transformer_basic_dims).permute(0, 4, 1, 2, 3).contiguous()
        else:
            x = torch.unsqueeze(x, dim=2)
            x = self.masker(x, mask)
            bssfp_x1, bssfp_x2, bssfp_x3, bssfp_x4, bssfp_x5 = self.bssfp_encoder(x[:, 0:1, :, :, :]) if mask[0,0] else (self.zeros_x1.cuda(), self.zeros_x2.cuda(), self.zeros_x3.cuda(), self.zeros_x4.cuda(), self.zeros_x5.cuda())
            lge_x1, lge_x2, lge_x3, lge_x4, lge_x5 = self.lge_encoder(x[:, 1:2, :, :, :]) if mask[0,1] else (self.zeros_x1.cuda(), self.zeros_x2.cuda(), self.zeros_x3.cuda(), self.zeros_x4.cuda(), self.zeros_x5.cuda())
            t2_x1, t2_x2, t2_x3, t2_x4, t2_x5 = self.t2_encoder(x[:, 2:3, :, :, :]) if mask[0,2] else (self.zeros_x1.cuda(), self.zeros_x2.cuda(), self.zeros_x3.cuda(), self.zeros_x4.cuda(), self.zeros_x5.cuda())

            if mask[0,0]:
                bssfp_token_x5 = self.bssfp_encode_conv(bssfp_x5).permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims)
                bssfp_intra_token_x5 = self.bssfp_transformer(bssfp_token_x5, self.bssfp_pos)
                bssfp_intra_x5 = bssfp_intra_token_x5.view(x.size(0), (H//16),(W//16),Z, transformer_basic_dims).permute(0, 4, 1, 2, 3).contiguous()
            else:
                bssfp_intra_x5 = self.zeros_intra.cuda()
            if mask[0,1]:
                lge_token_x5 = self.lge_encode_conv(lge_x5).permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims)
                lge_intra_token_x5 = self.lge_transformer(lge_token_x5, self.lge_pos)
                lge_intra_x5 = lge_intra_token_x5.view(x.size(0), (H//16),(W//16),Z, transformer_basic_dims).permute(0, 4, 1, 2, 3).contiguous()
            else:
                lge_intra_x5 = self.zeros_intra.cuda()
            if mask[0,2]:
                t2_token_x5 = self.t2_encode_conv(t2_x5).permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims)
                t2_intra_token_x5 = self.t2_transformer(t2_token_x5, self.t2_pos)
                t2_intra_x5 = t2_intra_token_x5.view(x.size(0), (H//16),(W//16),Z, transformer_basic_dims).permute(0, 4, 1, 2, 3).contiguous()
            else:
                t2_intra_x5 = self.zeros_intra.cuda()

        # if self.is_training:
        #     if self.mask_type == 'random_all':
        #         bssfp_pred = self.decoder_sep(bssfp_x1, bssfp_x2, bssfp_x3, bssfp_x4, bssfp_x5)
        #         lge_pred = self.decoder_sep(lge_x1, lge_x2, lge_x3, lge_x4, lge_x5)
        #         t2_pred = self.decoder_sep(t2_x1, t2_x2, t2_x3, t2_x4, t2_x5)
        #     else:
        #         bssfp_pred = self.decoder_sep(bssfp_x1, bssfp_x2, bssfp_x3, bssfp_x4, bssfp_x5) if mask[0,0] else 0
        #         lge_pred = self.decoder_sep(lge_x1, lge_x2, lge_x3, lge_x4, lge_x5) if mask[0,1] else 0
        #         t2_pred = self.decoder_sep(t2_x1, t2_x2, t2_x3, t2_x4, t2_x5) if mask[0,2] else 0
        ########### IntraFormer

        x1 = self.masker(torch.stack((bssfp_x1, lge_x1, t2_x1), dim=1), mask) #Bx4xCxHWZ
        x2 = self.masker(torch.stack((bssfp_x2, lge_x2, t2_x2), dim=1), mask)
        x3 = self.masker(torch.stack((bssfp_x3, lge_x3, t2_x3), dim=1), mask)
        x4 = self.masker(torch.stack((bssfp_x4, lge_x4, t2_x4), dim=1), mask)
        x5_intra = self.masker(torch.stack((bssfp_intra_x5, lge_intra_x5, t2_intra_x5), dim=1), mask)

        ########### InterFormer
        bssfp_intra_x5, lge_intra_x5, t2_intra_x5 = torch.chunk(x5_intra, num_modals, dim=1)
        multimodal_token_x5 = torch.cat((bssfp_intra_x5.permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims),
                                         lge_intra_x5.permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims),
                                         t2_intra_x5.permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims),
                                        ), dim=1)
        multimodal_pos = torch.cat((self.bssfp_pos, self.lge_pos, self.t2_pos), dim=1)
        multimodal_inter_token_x5 = self.multimodal_transformer(multimodal_token_x5, multimodal_pos)
        multimodal_inter_x5 = self.multimodal_decode_conv(multimodal_inter_token_x5.view(multimodal_inter_token_x5.size(0), (H//16),(W//16),Z, transformer_basic_dims*num_modals).permute(0, 4, 1, 2, 3).contiguous())
        x5_inter = multimodal_inter_x5

        fuse_pred, preds, de_f_avg = self.decoder_fuse(x1, x2, x3, x4, x5_inter)
        ########### InterFormer
        
        if self.is_training:
            if mask[0,0]:
                x1_bssfp = torch.cat((bssfp_x1, self.zeros_x1.cuda(), self.zeros_x1.cuda()), dim=1) #Bx4xCxHWZ
                x2_bssfp = torch.cat((bssfp_x2, self.zeros_x2.cuda(), self.zeros_x2.cuda()), dim=1)
                x3_bssfp = torch.cat((bssfp_x3, self.zeros_x3.cuda(), self.zeros_x3.cuda()), dim=1)
                x4_bssfp = torch.cat((bssfp_x4, self.zeros_x4.cuda(), self.zeros_x4.cuda()), dim=1)
                bssfp_token_x5 = torch.cat((bssfp_intra_x5.permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims),
                                        self.zeros_intra.cuda().permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims),
                                        self.zeros_intra.cuda().permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims),
                                        ), dim=1)
                bssfp_inter_token_x5 = self.multimodal_transformer(bssfp_token_x5, multimodal_pos)
                bssfp_inter_x5 = self.multimodal_decode_conv(bssfp_inter_token_x5.view(bssfp_inter_token_x5.size(0), (H//16),(W//16),Z, transformer_basic_dims*num_modals).permute(0, 4, 1, 2, 3).contiguous())
                x5_bssfp = bssfp_inter_x5
                fuse_pred_bssfp, preds_bssfp, de_f_bssfp = self.decoder_fuse(x1_bssfp, x2_bssfp, x3_bssfp, x4_bssfp, x5_bssfp, mask)
            if mask[0,1]:
                x1_lge = torch.cat((self.zeros_x1.cuda(), lge_x1, self.zeros_x1.cuda()), dim=1) #Bx4xCxHWZ
                x2_lge = torch.cat((self.zeros_x2.cuda(), lge_x2, self.zeros_x2.cuda()), dim=1)
                x3_lge = torch.cat((self.zeros_x3.cuda(), lge_x3, self.zeros_x3.cuda()), dim=1)
                x4_lge = torch.cat((self.zeros_x4.cuda(), lge_x4, self.zeros_x4.cuda()), dim=1)
                lge_token_x5 = torch.cat((self.zeros_intra.cuda().permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims),
                                        lge_intra_x5.permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims),
                                        self.zeros_intra.cuda().permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims),
                                        ), dim=1)
                lge_inter_token_x5 = self.multimodal_transformer(lge_token_x5, multimodal_pos)
                lge_inter_x5 = self.multimodal_decode_conv(lge_inter_token_x5.view(lge_inter_token_x5.size(0), (H//16),(W//16),Z, transformer_basic_dims*num_modals).permute(0, 4, 1, 2, 3).contiguous())
                x5_lge = lge_inter_x5
                fuse_pred_lge, preds_lge, de_f_lge = self.decoder_fuse(x1_lge, x2_lge, x3_lge, x4_lge, x5_lge, mask)
            if mask[0,2]:
                x1_t2 = torch.cat((self.zeros_x1.cuda(), self.zeros_x1.cuda(), t2_x1), dim=1) #Bx4xCxHWZ
                x2_t2 = torch.cat((self.zeros_x2.cuda(), self.zeros_x2.cuda(), t2_x2), dim=1)
                x3_t2 = torch.cat((self.zeros_x3.cuda(), self.zeros_x3.cuda(), t2_x3), dim=1)
                x4_t2 = torch.cat((self.zeros_x4.cuda(), self.zeros_x4.cuda(), t2_x4), dim=1)
                t2_token_x5 = torch.cat((self.zeros_intra.cuda().permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims),
                                        self.zeros_intra.cuda().permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims),
                                        t2_intra_x5.permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims),
                                        ), dim=1)
                t2_inter_token_x5 = self.multimodal_transformer(t2_token_x5, multimodal_pos)
                t2_inter_x5 = self.multimodal_decode_conv(t2_inter_token_x5.view(t2_inter_token_x5.size(0), (H//16),(W//16),Z, transformer_basic_dims*num_modals).permute(0, 4, 1, 2, 3).contiguous())
                x5_t2 = t2_inter_x5
                fuse_pred_t2, preds_t2, de_f_t2 = self.decoder_fuse(x1_t2, x2_t2, x3_t2, x4_t2, x5_t2, mask)
            # kl_loss = torch.zeros(3).cuda().float()
            sep_cross_loss = torch.zeros(3).cuda().float()
            sep_dice_loss = torch.zeros(3).cuda().float()
            prm_cross_loss = torch.zeros(1).cuda().float()
            prm_dice_loss = torch.zeros(1).cuda().float()
            # proto_loss = torch.zeros(3).cuda().float()
            # fkl_loss = torch.zeros(3).cuda().float()

            if mask[0,0]:
                bssfp_pred = F.softmax(fuse_pred_bssfp, dim=1)
                sep_cross_loss[0] = softmax_weighted_loss(bssfp_pred, target, num_cls=num_cls)
                sep_dice_loss[0] = dice_loss(bssfp_pred, target, num_cls=num_cls)

            if mask[0,1]:
                lge_pred = F.softmax(fuse_pred_lge, dim=1)
                sep_cross_loss[1] = softmax_weighted_loss(lge_pred, target, num_cls=num_cls)
                sep_dice_loss[1] = dice_loss(lge_pred, target, num_cls=num_cls)

            if mask[0,2]:
                t2_pred = F.softmax(fuse_pred_t2, dim=1)
                sep_cross_loss[2] = softmax_weighted_loss(t2_pred, target, num_cls=num_cls)
                sep_dice_loss[2] = dice_loss(t2_pred, target, num_cls=num_cls)

            weight_prm = 1.0
            for prm_pred, up_op in zip(preds, self.up_ops):
                weight_prm /= 2.0
                prm_cross_loss += weight_prm * softmax_weighted_loss(F.softmax(prm_pred, dim=1), target, num_cls=num_cls, up_op=up_op)
                prm_dice_loss += weight_prm * dice_loss(F.softmax(prm_pred, dim=1), target, num_cls=num_cls, up_op=up_op)
            return F.softmax(fuse_pred, dim=1), (prm_cross_loss, prm_dice_loss), (sep_cross_loss, sep_dice_loss)
        return F.softmax(fuse_pred, dim=1)
        #     return fuse_pred, (bssfp_pred, lge_pred, t2_pred), preds
        # return fuse_pred
        # return self.decoder_sep(bssfp_x1, bssfp_x2, bssfp_x3, bssfp_x4, bssfp_x5)