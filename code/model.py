import math

import torch.nn as nn
import torch
import torch.nn.functional as F

class StripAttention(nn.Module):
    def __init__(self, dim, kernel=5, group=2, H=True) -> None:
        super().__init__()
        self.k = kernel
        pad = kernel // 2
        self.kernel = (1, kernel) if H else (kernel, 1)
        self.padding = (kernel // 2, 1) if H else (1, kernel // 2)

        self.group = group
        self.pad = nn.ReflectionPad2d((pad, pad, 0, 0)) if H else nn.ReflectionPad2d((0, 0, pad, pad))
        self.conv = nn.Conv2d(dim, group * kernel, kernel_size=1, stride=1, bias=False)
        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        self.filter_act = nn.Sigmoid()

    def forward(self, x):
        filter = self.ap(x)
        filter = self.conv(filter)
        n, c, h, w = x.shape
        x = F.unfold(self.pad(x), kernel_size=self.kernel).reshape(n, self.group, c // self.group, self.k, h * w)

        n, c1, p, q = filter.shape
        filter = filter.reshape(n, c1 // self.k, self.k, p * q).unsqueeze(2)
        filter = self.filter_act(filter)

        out = torch.sum(x * filter, dim=3).reshape(n, c, h, w)
        return out


class SAM(nn.Module):


    def __init__(self, dim, group, kernel) -> None:
        super().__init__()

        self.H_spatial_att = StripAttention(dim, group=group, kernel=kernel)
        self.W_spatial_att = StripAttention(dim, group=group, kernel=kernel, H=False)
        self.gamma = nn.Parameter(torch.zeros(dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        out = self.H_spatial_att(x)
        out = self.W_spatial_att(out)
        return self.gamma * out + x * self.beta


class SNetBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SNetBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.sam_11 = SAM(out_channel, group=1, kernel=11)
        self.sam_7 = SAM(out_channel, group=1, kernel=7)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel * 2, out_channel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.down = nn.MaxPool2d(2)

    def forward(self, x):
        out = self.conv1(x)

        out = torch.chunk(out, 2, dim=1)
        out_11 = self.sam_11(out[0])
        out_7 = self.sam_7(out[1])
        out = torch.cat((out_11, out_7), dim=1)

        out = self.conv2(out) + x
        out = self.down(out)
        return out


class SNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=1):

        super(SNet, self).__init__()

        self._block1 = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

        self._block2 = SNetBlock(48, 48)

        self._block3 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(48, 48, 3, stride=2, padding=1, output_padding=1))

        self._block4 = nn.Sequential(
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))

        self._block5 = nn.Sequential(
            nn.Conv2d(144, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))

        self._block6 = nn.Sequential(
            nn.Conv2d(96 + in_channels, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1))

    def _init_weights(self):
       

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):

        pool1 = self._block1(x)
        pool2 = self._block2(pool1)
        pool3 = self._block2(pool2)
        pool4 = self._block2(pool3)
        pool5 = self._block2(pool4)

        upsample5 = self._block3(pool5)

        concat5 = torch.cat((upsample5, pool4), dim=1)
        upsample4 = self._block4(concat5)
        concat4 = torch.cat((upsample4, pool3), dim=1)
        upsample3 = self._block5(concat4)
        concat3 = torch.cat((upsample3, pool2), dim=1)
        upsample2 = self._block5(concat3)
        concat2 = torch.cat((upsample2, pool1), dim=1)
        upsample1 = self._block5(concat2)
        concat1 = torch.cat((upsample1, x), dim=1)

        return self._block6(concat1)

class Upsampler(nn.Sequential):
    def __init__(self, scale, n_feats, bn=False, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(n_feats, 4 * n_feats, kernel_size=3, padding=(3 // 2), bias=bias))
                m.append(nn.PixelShuffle(2))

                if bn:
                    m.append(nn.BatchNorm2d(n_feats))

                    if act == 'relu':
                        m.append(nn.ReLU(True))
                    elif act == 'prelu':
                        m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(nn.Conv2d(n_feats, 9 * n_feats, kernel_size=3, padding=(3 // 2), bias=bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)
class DepthwiseConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(DepthwiseConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class GatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(GatedConv, self).__init__()
        self.feature_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.gate_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        feature = self.feature_conv(x)
        gate = torch.sigmoid(self.gate_conv(x))
        return feature * gate

class SRNet(nn.Module):

    def __init__(self, in_channels, scale_factor):
        super(SRNet, self).__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * scale_factor ** 2, kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor))

    def forward(self, x):
        return self.up(x)

class FNet(nn.Module):

    def __init__(self, in_channels, mid_channels, out_c):
        super(FNet, self).__init__()
        self.branchA0 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.GELU()
        )

        self.branchA1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0),
            DepthwiseConv(mid_channels, mid_channels, kernel_size=3, padding=1))

        self.branchA2 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0),
            DepthwiseConv(mid_channels, mid_channels, kernel_size=5, padding=2))

        self.Gate = GatedConv(mid_channels*2, mid_channels, 3, 1, 1)
        self.fuse_conv = nn.Conv2d(mid_channels, mid_channels, kernel_size=1, padding=0)
        self.channel_attn = ChannelAttention(mid_channels)
        self.pre = nn.Conv2d(mid_channels, out_c, kernel_size=3, padding=1)

    def forward(self, fea):
        A0 = self.branchA0(fea)
        A1 = self.branchA1(fea)
        A2 = self.branchA2(fea)
        A1A2 = self.Gate(torch.cat((A1, A2), dim=1))
        fea_fusion = self.fuse_conv(A0*A1A2)
        fea_fusion = self.channel_attn(fea_fusion)
        out = self.pre(fea_fusion)
        return out

class CRTFR(nn.Module):

    def __init__(self, in_c, out_c, feature_dim, scale_factor):
        super().__init__()
        self.snet = SNet(in_channels=in_c, out_channels=feature_dim)
        self.srnet = SRNet(feature_dim, scale_factor)
        self.fnet = FNet(feature_dim, feature_dim, out_c)

    def forward(self, x):
        x_encoded = self.snet(x)
        x_sr = self.srnet(x_encoded)
        out = self.fnet(x_sr)
        return out
