from networks.models.resnet import *
from networks.models.unet_parts import *

class Res_UNet_50(nn.Module):
    def __init__(self, input_bands, n_classes, thread_pro=0.4, bilinear=True):
        super(Res_UNet_50, self).__init__()
        self.n_channels = 3
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.thread_pro = thread_pro
        self.backbone = Res50(pretrained=False, band_num=input_bands)

        self.inc = DoubleConv(input_bands, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(2048+1024, 256, bilinear)
        self.up2 = Up(512+256, 128, bilinear)
        self.up3 = Up(256+128, 64, bilinear)
        self.up4 = Up(64+64, 64, bilinear)
        self.conv =  nn.Conv2d(64, 6, kernel_size=3, padding=1)

        if self.thread_pro:
            self.outc = OutConv(64, 1)
        else:
            self.outc = OutConv(64, n_classes)
        self.last = nn.Sigmoid()


    def forward(self, x):
        size = x.size()
        layers = self.backbone(x)
        x1, x2, x3, x4, x5 = layers[0], layers[1], layers[2], layers[3], layers[4]

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.conv(x)
        logits = F.interpolate(x, size=(size[2], size[3]), mode='bilinear')
        # logits = self.last(self.outc(logits))
        return logits


class Res_UNet_34(nn.Module):
    def __init__(self, input_bands, n_classes, thread_pro=0.4, bilinear=True):
        super(Res_UNet_34, self).__init__()
        self.n_channels = 3
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.thread_pro = thread_pro
        self.backbone = Res50(pretrained=False, band_num=input_bands)

        self.inc = DoubleConv(input_bands, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(2048 + 1024, 256, bilinear)
        self.up2 = Up(512 + 256, 128, bilinear)
        self.up3 = Up(256 + 128, 64, bilinear)
        self.up4 = Up(64 + 64, 64, bilinear)

        if self.thread_pro:
            self.outc = OutConv(64, 1)
        else:
            self.outc = OutConv(64, n_classes)
        self.last = nn.Sigmoid()

    def forward(self, x):
        size = x.size()
        layers = self.backbone(x)
        x1, x2, x3, x4, x5 = layers[0], layers[1], layers[2], layers[3], layers[4]

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        # logits = F.interpolate(x, size=(size[2], size[3]), mode='bilinear')
        # logits = self.last(self.outc(logits))
        return x




import torch
from thop import profile
if __name__ == "__main__":
    import time

    model = Res_UNet_50(3, 5, 0.3)
    # stat(net, (3, 512, 512))

    for idx, m in enumerate(model.modules()):
        print(idx, "-", m)
    s = time.time()

    rgb = torch.ones(1, 3, 448, 448, dtype=torch.float, requires_grad=False)
    out = model(rgb)
    flops, params = profile(model, inputs=(rgb,))
    print('parameters:', params)
    print('flops', flops)
    print('time: {:.4f}ms'.format((time.time()-s)*10))
