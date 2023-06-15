"""
VSN model
VSNLite4M_n1=12_//2_ns（VSNLite1MC）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    (convolution => [BN] => ReLU) * 2
    经过ConvBlock的特征，其height和width不会改变
    """

    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.DSC1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=True, groups=in_channels),
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            nn.InstanceNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.DSC2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=True, groups=mid_channels),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=1),
            nn.InstanceNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.DSC3 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=True, groups=mid_channels),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # self.short_DSC = nn.Sequential(
        #     nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=True, groups=in_channels),
        #     nn.Conv2d(in_channels, out_channels, kernel_size=1),
        #     nn.InstanceNorm2d(out_channels),
        #     nn.ReLU(inplace=True)
        # )

    def forward(self, x):
        out = self.DSC1(x)
        out = self.DSC2(out)
        out = self.DSC3(out)
        # short = self.short_DSC(x)
        return out


class Down(nn.Module):
    """
    Downscaling with maxpool then double conv
    经过Down的特征，其输出heigh和width是输入特征的一半
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, in_channels // 4, out_channels)
        )

    def forward(self, x):
        return self.down_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBlock(in_channels, in_channels // 4, out_channels)
        )

    def forward(self, x):
        return self.up_conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


def SafeCat(x, y):
    # input is NCHW
    diffH = y.size()[2] - x.size()[2]
    diffW = y.size()[3] - x.size()[3]
    x = F.pad(x, [diffW // 2, diffW - diffW // 2,
                  diffH // 2, diffH - diffH // 2])
    out = torch.cat([y, x], dim=1)
    return out


# End of components define #


class VSNLite1MC(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(VSNLite1MC, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        n1 = 12
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]  # 保持两倍关系

        self.Init_conv = ConvBlock(n_channels, n_channels, filters[0])

        self.Down1 = Down(filters[0], filters[0])
        self.Down_conv1 = ConvBlock(filters[0], filters[0] // 2, filters[1])
        self.Down2 = Down(filters[1], filters[1])
        self.Down_conv2 = ConvBlock(filters[1], filters[1] // 2, filters[2])
        self.Down3 = Down(filters[2], filters[2])
        self.Down_conv3 = ConvBlock(filters[2], filters[2] // 2, filters[3])
        self.Down4 = Down(filters[3], filters[3])
        self.Down_conv4 = ConvBlock(filters[3], filters[3] // 2, filters[4])

        self.Up4 = Up(filters[4], filters[3])  # 注意，由于Up后要cat之前的特征，故输出特征通道数减半
        self.Up_conv4 = ConvBlock(filters[4], filters[4] // 2, filters[3])
        self.Up3 = Up(filters[3], filters[2])
        self.Up_conv3 = ConvBlock(filters[3], filters[3] // 2, filters[2])
        self.Up2 = Up(filters[2], filters[1])
        self.Up_conv2 = ConvBlock(filters[2], filters[2] // 2, filters[1])
        self.Up1 = Up(filters[1], filters[0])
        self.Up_conv1 = ConvBlock(filters[1], filters[1] // 2, filters[0])

        self.Out_conv = OutConv(filters[0], n_classes)

    def forward(self, x):
        e1 = self.Init_conv(x)

        e2 = self.Down1(e1)
        e2 = self.Down_conv1(e2)

        e3 = self.Down2(e2)
        e3 = self.Down_conv2(e3)

        e4 = self.Down3(e3)
        e4 = self.Down_conv3(e4)

        e5 = self.Down4(e4)
        e5 = self.Down_conv4(e5)

        d4 = self.Up4(e5)
        d4 = SafeCat(d4, e4)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = SafeCat(d3, e3)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = SafeCat(d2, e2)
        d2 = self.Up_conv2(d2)

        d1 = self.Up1(d2)
        d1 = SafeCat(d1, e1)
        d1 = self.Up_conv1(d1)

        return self.Out_conv(d1)
