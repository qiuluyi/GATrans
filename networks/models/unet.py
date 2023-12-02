""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
from networks.models.unet_parts import *

class unet(nn.Module):
    def __init__(self, input_bands, n_classes, thread_pro, bilinear=True):
        super(unet, self).__init__()
        self.n_channels = 3
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.thread_pro= thread_pro

        self.inc = DoubleConv(input_bands, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)


        if self.thread_pro:
            self.outc = OutConv(64, 6)
        else:
            self.outc = OutConv(64, 6)
        self.last = nn.Sigmoid()

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

import torch
from thop import profile
if __name__ == "__main__":
    import time

    model = unet(3,6,0.4)
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

