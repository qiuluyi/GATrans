import torch
import torch.nn as nn
from networks.sync_batchnorm import SynchronizedBatchNorm2d

from collections import OrderedDict


def conv3x3(in_planes, out_planes, stride=1):
    # "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv3x3_long(in_planes, out_planes, stride=1):
    # "3x3 convolution with padding"
    return ConvLong(in_planes, out_planes, stride)

class ConvLong(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        mid_channel = int(in_channels / 4)
        self.conv39 = nn.Conv2d(in_channels, mid_channel, kernel_size=(3, 9), stride=1, padding=(1, 4),
                                bias=False)
        self.conv93 = nn.Conv2d(in_channels, mid_channel, kernel_size=(9, 3), stride=1, padding=(4, 1),
                                bias=False)
        self.conv37 = nn.Conv2d(in_channels, mid_channel, kernel_size=(3, 7), stride=1, padding=(1, 3),
                                bias=False)
        self.conv73 = nn.Conv2d(in_channels, mid_channel, kernel_size=(7, 3), stride=1, padding=(3, 1),
                                bias=False)
        self.conv35 = nn.Conv2d(in_channels, mid_channel, kernel_size=(3, 5), stride=1, padding=(1, 2),
                                bias=False)
        self.conv53 = nn.Conv2d(in_channels, mid_channel, kernel_size=(5, 3), stride=1, padding=(2, 1),
                                bias=False)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(mid_channel)
        self.conv_out = nn.Conv2d(mid_channel, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv_last = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        x39 = self.relu(self.bn1(self.conv39(x)))
        x93 = self.relu(self.bn1(self.conv93(x)))
        x1 = self.relu(self.bn2(self.conv_out(x93 + x39)))

        x37 = self.relu(self.bn1(self.conv37(x)))
        x73 = self.relu(self.bn1(self.conv73(x)))
        x2 = self.relu(self.bn2(self.conv_out(x73 + x37)))

        x35 = self.relu(self.bn1(self.conv35(x)))
        x53 = self.relu(self.bn1(self.conv53(x)))
        x3 = self.relu(self.bn2(self.conv_out(x53 + x35)))

        out = self.conv_last(x1 + x2 + x3)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        m = OrderedDict()
        m['conv1'] = conv3x3_long(inplanes, planes, stride)
        m['bn1'] = SynchronizedBatchNorm2d(planes)
        m['relu1'] = nn.ReLU(inplace=True)
        m['conv2'] = conv3x3_long(planes, planes)
        m['bn2'] = SynchronizedBatchNorm2d(planes)
        self.group1 = nn.Sequential(m)

        self.relu = nn.Sequential(nn.ReLU(inplace=True))
        self.downsample = downsample

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.group1(x) + residual

        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        m = OrderedDict()
        m['conv1'] = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        m['bn1'] = SynchronizedBatchNorm2d(planes)
        m['relu1'] = nn.ReLU(inplace=True)
        m['conv2'] = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        m['bn2'] = SynchronizedBatchNorm2d(planes)
        m['relu2'] = nn.ReLU(inplace=True)
        m['conv3'] = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        m['bn3'] = SynchronizedBatchNorm2d(planes * 4)
        self.group1 = nn.Sequential(m)

        self.relu = nn.Sequential(nn.ReLU(inplace=True))
        self.downsample = downsample

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.group1(x) + residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, band_num=4, os=16):
        super(ResNet, self).__init__()

        self.inplanes = 64
        self.conv1 = nn.Conv2d(band_num, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = SynchronizedBatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.Sequential(nn.AvgPool2d(7))

        self.group2 = nn.Sequential(
            OrderedDict([
                ('fc', nn.Linear(512 * block.expansion, num_classes))
            ])
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                SynchronizedBatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def get_layers(self):
        return self.layers

    def forward(self, x):
        self.layers = []
        x = self.conv1(x)  # downsamples X2
        x = self.bn1(x)
        x = self.relu(x)
        self.layers.append(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        self.layers.append(x)
        x = self.layer2(x)
        self.layers.append(x)
        x = self.layer3(x)
        self.layers.append(x)
        x = self.layer4(x)
        self.layers.append(x)
        # import ipdb;ipdb.set_trace()

        return self.layers


def Res18(pretrained=False, model_root=None, band_num=4, **kwargs):
    """Constructs a atrous ResNet-18 model."""
    model = ResNet(BasicBlock, [2, 2, 2, 2], band_num=band_num, **kwargs)
    # model = torchvision.models.resnet18()
    # model.conv1 = nn.Conv2d(band_num, 64, kernel_size=7, stride=2, padding=3,
    #                            bias=False)
    if pretrained:
        old_dict = torch.load('./networks/pretrained/resnet18-5c106cde.pth')

        conv1_weight = old_dict['conv1.weight']
        for i in range(3, band_num):
            conv1_weight = torch.cat((conv1_weight, conv1_weight[:, (i % 3):(i % 3 + 1), :, :]), 1)

        # import ipdb;ipdb.set_trace()
        model_dict = model.state_dict()
        old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
        old_dict['conv1.weight'] = conv1_weight
        model_dict.update(old_dict)
        model.load_state_dict(model_dict)
    return model


def Res34(pretrained=False, model_root=None, band_num=4, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], band_num=band_num, **kwargs)
    # model = torchvision.models.resnet34()
    # model.conv1 = nn.Conv2d(band_num, 64, kernel_size=7, stride=2, padding=3,
    #                            bias=False)
    if pretrained:
        old_dict = torch.load('./networks/pretrained/resnet34-333f7ec4.pth')

        conv1_weight = old_dict['conv1.weight']
        for i in range(3, band_num):
            conv1_weight = torch.cat((conv1_weight, conv1_weight[:, (i % 3):(i % 3 + 1), :, :]), 1)

        model_dict = model.state_dict()
        old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
        old_dict['conv1.weight'] = conv1_weight
        model_dict.update(old_dict)
        model.load_state_dict(model_dict)
    return model


def Res50(pretrained=False, model_root=None, band_num=4, **kwargs):
    """Constructs a atrous ResNet-50 model."""
    model = ResNet(Bottleneck, [3, 4, 6, 3], band_num=band_num, **kwargs)
    # import ipdb;ipdb.set_trace()
    # model = torchvision.models.resnet50()
    # model.conv1 = nn.Conv2d(band_num, 64, kernel_size=7, stride=2, padding=3,
    #                            bias=False)

    if pretrained:
        old_dict = torch.load('./networks/pretrained/resnet50-19c8e357.pth')  # resnet50-19c8e357 RS_resnet_50

        conv1_weight = old_dict['conv1.weight']
        for i in range(3, band_num):
            conv1_weight = torch.cat((conv1_weight, conv1_weight[:, (i % 3):(i % 3 + 1), :, :]), 1)

        # import ipdb;ipdb.set_trace()
        model_dict = model.state_dict()
        old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
        old_dict['conv1.weight'] = conv1_weight
        model_dict.update(old_dict)
        model.load_state_dict(model_dict)
    return model


def Res101(pretrained=False, model_root=None, band_num=4, **kwargs):
    """Constructs a atrous ResNet-50 model."""
    model = ResNet(Bottleneck, [3, 4, 23, 3], band_num=band_num, **kwargs)
    # model = torchvision.models.resnet101()
    # model.conv1 = nn.Conv2d(band_num, 64, kernel_size=7, stride=2, padding=3,
    #                            bias=False)

    if pretrained:
        old_dict = torch.load('./networks/pretrained/resnet101-5d3b4d8f.pth')

        conv1_weight = old_dict['conv1.weight']
        for i in range(3, band_num):
            conv1_weight = torch.cat((conv1_weight, conv1_weight[:, (i % 3):(i % 3 + 1), :, :]), 1)

        # import ipdb;ipdb.set_trace()
        model_dict = model.state_dict()
        old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
        old_dict['conv1.weight'] = conv1_weight
        model_dict.update(old_dict)
        model.load_state_dict(model_dict)
    return model


def Res152(pretrained=False, model_root=None, band_num=4, **kwargs):
    """Constructs a atrous ResNet-50 model."""
    model = ResNet(Bottleneck, [3, 8, 36, 3], band_num=band_num, **kwargs)
    # model = torchvision.models.resnet152()
    # model.conv1 = nn.Conv2d(band_num, 64, kernel_size=7, stride=2, padding=3,
    #                            bias=False)

    if pretrained:
        old_dict = torch.load('./networks/pretrained/resnet152-b121ed2d.pth')

        conv1_weight = old_dict['conv1.weight']
        for i in range(3, band_num):
            conv1_weight = torch.cat((conv1_weight, conv1_weight[:, (i % 3):(i % 3 + 1), :, :]), 1)

        # import ipdb;ipdb.set_trace()
        model_dict = model.state_dict()
        old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
        old_dict['conv1.weight'] = conv1_weight
        model_dict.update(old_dict)
        model.load_state_dict(model_dict)
    return model
