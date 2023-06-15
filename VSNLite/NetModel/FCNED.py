""" FCNED model """

import torch
import torch.nn as nn
import torch.nn.functional as F


# 卷积块
class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


# 卷积下采样
class DownConvBlock(nn.Module):
    # 在卷积块上添加MaxPool

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


# 卷积上采样
class UpConvBlock(nn.Module):
    # 在卷积块上添加上采样

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # 选择是卷积上采样还是插值
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = ConvBlock(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = ConvBlock(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # diff - diff //2 ，是为了当输入图片尺寸为奇数时也能正常使用
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# 用于转换通道的输出
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# End of FCNED components define #


class FCNED(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(FCNED, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = ConvBlock(n_channels, 64)
        self.down1 = DownConvBlock(64, 128)
        self.down2 = DownConvBlock(128, 256)
        self.down3 = DownConvBlock(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = DownConvBlock(512, 1024 // factor)
        self.up1 = UpConvBlock(1024, 512 // factor, bilinear)
        self.up2 = UpConvBlock(512, 256 // factor, bilinear)
        self.up3 = UpConvBlock(256, 128 // factor, bilinear)
        self.up4 = UpConvBlock(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
