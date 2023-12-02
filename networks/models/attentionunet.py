""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F

class MecaAttention(nn.Module):
    def __init__(self, input_c, length, k_size=3):
        super(MecaAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.conv_channel = nn.Conv1d(input_c, input_c, kernel_size=k_size, stride=1, padding=1)
        self.conv_xy = nn.Conv1d(length, length, kernel_size=k_size, stride=1, padding=1)
        self.input_c = input_c
        self.length = length
        self.activation = nn.Sigmoid()
        self.conv_out = nn.Sequential(nn.Conv2d(input_c, input_c, 3, 1, 1, 1),
                                      nn.BatchNorm2d(input_c),
                                      nn.ReLU())

    def get_weights(self, inputs, length, conv):
        x_avgp = self.avg_pool(inputs)
        x_avgp = x_avgp.view(-1, length, 1, 1)
        x_avgp = x_avgp.squeeze(-1)
        x_avgp = conv(x_avgp)
        x_avgp = x_avgp.unsqueeze(-1)

        x_maxp = self.max_pool(inputs)
        x_maxp = x_maxp.view(-1, length, 1, 1)
        x_maxp = x_maxp.squeeze(-1)
        x_maxp = conv(x_maxp)
        x_maxp = x_maxp.unsqueeze(-1)

        eca_feature = x_maxp + x_avgp
        eca_feature = self.activation(eca_feature)

        return eca_feature

    def forward(self, input_tensor):
        channel_weight = self.get_weights(input_tensor, self.input_c, self.conv_channel)
        input_1 = channel_weight * input_tensor

        input_tensor_x = input_tensor.permute(0, 2, 1, 3)
        x_weights = self.get_weights(input_tensor_x, self.length, self.conv_xy)
        input_2 = x_weights * input_tensor_x
        input_2 = input_2.permute(0, 2, 1, 3)

        input_tensor_y = input_tensor.permute(0, 3, 2, 1)
        y_weights = self.get_weights(input_tensor_y, self.length, self.conv_xy)
        input_3 = y_weights * input_tensor_y
        input_3 = input_3.permute(0, 3, 2, 1)

        output = input_1+input_2+input_3
        output = self.conv_out(output)
        return output

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, meca_c, out_channels, length=256, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.meca = MecaAttention(input_c=meca_c, length=length)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.meca(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
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
        return x * y.expand_as(x)

class SFAM(nn.Module):
    def __init__(self, planes, planes2, num_levels=4, num_scales=4, compress_ratio=16):
        super(SFAM, self).__init__()
        self.planes = planes
        self.planes2 = planes2
        self.num_levels = num_levels
        self.num_scales = num_scales
        self.compress_ratio = compress_ratio
        self.pre_conv = nn.Sequential(nn.Conv2d(self.planes2, self.planes, 3, 1, 1),
                                      nn.BatchNorm2d(self.planes),
                                      nn.ReLU())

        self.fc1 = nn.ModuleList([nn.Conv2d(self.planes,
                                            self.planes * self.num_levels // self.compress_ratio,
                                            1, 1, 0)] * self.num_scales)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.ModuleList([nn.Conv2d(self.planes * self.num_levels // self.compress_ratio,
                                            self.planes,
                                            1, 1, 0)] * self.num_scales)
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        attention_feat = []
        x[1] = self.pre_conv(x[1])
        for i, _mf in enumerate(x):
            _tmp_f = self.avgpool(_mf)
            _tmp_f = self.fc1[i](_tmp_f)
            _tmp_f = self.relu(_tmp_f)
            _tmp_f = self.fc2[i](_tmp_f)
            _tmp_f = self.sigmoid(_tmp_f)
            attention_feat.append(_mf * _tmp_f)
        return attention_feat

class unet_att(nn.Module):
    def __init__(self, input_bands, n_classes, thread_pro, bilinear=True):
        super(unet_att, self).__init__()
        self.n_channels = 3
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.thread_pro = thread_pro

        self.inc = DoubleConv(input_bands, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.down5 = Down(512, 512)
        self.down6 = Down(512, 1024)
        self.sfam00 = SFAM(1024, 512)
        self.up00 = Up(1024+1024, 1024, 512, 16)
        self.sfam0 = SFAM(512, 512)
        self.up0 = Up(512+512, 512, 512, 32)
        self.sfam1 = SFAM(512, 512)
        self.up1 = Up(512+512, 512, 256, 64)
        self.sfam2 = SFAM(256, 256)
        self.up2 = Up(256+256, 256, 128, 128)
        self.sfam3 = SFAM(128, 128)
        self.up3 = Up(128+128, 128, 64, 256)
        self.sfam4 = SFAM(64, 64)
        self.up4 = Up(64+64, 64, 64, 512)
        # self.outc = OutConv(64, n_classes)
        if self.thread_pro:
            self.outc = OutConv(64, 6)
        else:
            self.outc = OutConv(64, n_classes)
        self.last = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        x_cat = self.sfam00([x7, x6])
        x = self.up00(x_cat[0], x_cat[1])
        x_cat = self.sfam0([x, x5])
        x = self.up0(x_cat[0], x_cat[1])
        x_cat = self.sfam1([x, x4])
        x = self.up1(x_cat[0], x_cat[1])
        x_cat = self.sfam2([x, x3])
        x = self.up2(x_cat[0], x_cat[1])
        x_cat = self.sfam3([x, x2])
        x = self.up3(x_cat[0], x_cat[1])
        x_cat = self.sfam4([x, x1])
        x = self.up4(x_cat[0], x_cat[1])
        logits = self.outc(x)

        return logits



import torch
from thop import profile
if __name__ == "__main__":
    import time

    device = torch.device("cuda:1")
    model = unet_att(input_bands=4, n_classes=6, thread_pro=0.3)
    model.to(device)
    # stat(net, (3, 512, 512))
    # flops, params = profile(model, inputs=(rgb,))
    # print('parameters:', params)
    # print('flops', flops)
    for idx, m in enumerate(model.modules()):
        print(idx, "-", m)
    s = time.time()

    for i in range(1000):
        rgb = torch.ones(1, 4, 512, 512, dtype=torch.float, requires_grad=False)
        rgb = rgb.to(device)
        out = model(rgb)
        print('time: {:.4f}ms'.format((time.time()-s)*10))
    print(((time.time()-s)*10)/1000)



