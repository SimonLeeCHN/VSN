""" Full assembly of the parts to form the complete network """

from .model_parts import *


class VSN(nn.Module):
    # def __init__(self, n_channels, n_classes, bilinear=True):
    #     super(VSN, self).__init__()
    #     self.n_channels = n_channels
    #     self.n_classes = n_classes
    #     self.bilinear = bilinear
    #
    #     self.inc = DoubleConv(n_channels, 64)
    #     self.down1 = Down(64, 128)
    #     self.down2 = Down(128, 256)
    #     self.down3 = Down(256, 512)
    #     factor = 2 if bilinear else 1
    #     self.down4 = Down(512, 1024 // factor)
    #     self.up1 = Up(1024, 512 // factor, bilinear)
    #     self.up2 = Up(512, 256 // factor, bilinear)
    #     self.up3 = Up(256, 128 // factor, bilinear)
    #     self.up4 = Up(128, 64, bilinear)
    #     self.outc = OutConv(64, n_classes)
    #
    # def forward(self, x):
    #     x1 = self.inc(x)
    #     x2 = self.down1(x1)
    #     x3 = self.down2(x2)
    #     x4 = self.down3(x3)
    #     x5 = self.down4(x4)
    #     x = self.up1(x5, x4)
    #     x = self.up2(x, x3)
    #     x = self.up3(x, x2)
    #     x = self.up4(x, x1)
    #     logits = self.outc(x)
    #     return logits


    def __init__(self, n_channels, n_classes):
        super(VSN, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        n1 = 64
        filters = [n1, n1*2, n1*4, n1*8, n1*16]                                 # 保持两倍关系

        self.Init_conv = ConvBlock(n_channels, n_channels, filters[0])

        self.Down1 = Down(filters[0], filters[0])
        self.Down_conv1 = ConvBlock(filters[0], filters[0] // 4, filters[1])
        self.Down2 = Down(filters[1], filters[1])
        self.Down_conv2 = ConvBlock(filters[1], filters[1] // 4, filters[2])
        self.Down3 = Down(filters[2], filters[2])
        self.Down_conv3 = ConvBlock(filters[2], filters[2] // 4, filters[3])
        self.Down4 = Down(filters[3], filters[3])
        self.Down_conv4 = ConvBlock(filters[3], filters[3] // 4, filters[4])

        self.Up4 = Up(filters[4], filters[3])                                   # 注意，由于Up后要cat之前的特征，故输出特征通道数减半
        self.Up_conv4 = ConvBlock(filters[4], filters[4] // 4, filters[3])
        self.Up3 = Up(filters[3], filters[2])
        self.Up_conv3 = ConvBlock(filters[3], filters[3] // 4, filters[2])
        self.Up2 = Up(filters[2], filters[1])
        self.Up_conv2 = ConvBlock(filters[2], filters[2] // 4, filters[1])
        self.Up1 = Up(filters[1], filters[0])
        self.Up_conv1 = ConvBlock(filters[1], filters[1] // 4, filters[0])

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
