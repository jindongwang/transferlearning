import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import mmd
import torch
import torch.nn.functional as F
import random


__all__ = ['ResNet', 'resnet50']


model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.baselayer = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

class MRANNet(nn.Module):

    def __init__(self, num_classes=31):
        super(MRANNet, self).__init__()
        self.sharedNet = resnet50(True)
        self.Inception = InceptionA(2048, 64, num_classes)

    def forward(self, source, target, s_label):
        source = self.sharedNet(source)
        target = self.sharedNet(target)
        source, loss = self.Inception(source, target, s_label)

        return source, loss

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features, num_classes):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

        self.avg_pool = nn.AvgPool2d(7, stride=1)

        self.source_fc = nn.Linear(288, num_classes)

    def forward(self, source, target, s_label):
        s_branch1x1 = self.branch1x1(source)

        s_branch5x5 = self.branch5x5_1(source)
        s_branch5x5 = self.branch5x5_2(s_branch5x5)

        s_branch3x3dbl = self.branch3x3dbl_1(source)
        s_branch3x3dbl = self.branch3x3dbl_2(s_branch3x3dbl)
        s_branch3x3dbl = self.branch3x3dbl_3(s_branch3x3dbl)

        s_branch_pool = F.avg_pool2d(source, kernel_size=3, stride=1, padding=1)
        s_branch_pool = self.branch_pool(s_branch_pool)

        s_branch1x1 = self.avg_pool(s_branch1x1)
        s_branch5x5 = self.avg_pool(s_branch5x5)
        s_branch3x3dbl = self.avg_pool(s_branch3x3dbl)
        s_branch_pool = self.avg_pool(s_branch_pool)

        s_branch1x1 = s_branch1x1.view(s_branch1x1.size(0), -1)
        s_branch5x5 = s_branch5x5.view(s_branch5x5.size(0), -1)
        s_branch3x3dbl = s_branch3x3dbl.view(s_branch3x3dbl.size(0), -1)
        s_branch_pool = s_branch_pool.view(s_branch_pool.size(0), -1)

        t_branch1x1 = self.branch1x1(target)

        t_branch5x5 = self.branch5x5_1(target)
        t_branch5x5 = self.branch5x5_2(t_branch5x5)

        t_branch3x3dbl = self.branch3x3dbl_1(target)
        t_branch3x3dbl = self.branch3x3dbl_2(t_branch3x3dbl)
        t_branch3x3dbl = self.branch3x3dbl_3(t_branch3x3dbl)

        t_branch_pool = F.avg_pool2d(target, kernel_size=3, stride=1, padding=1)
        t_branch_pool = self.branch_pool(t_branch_pool)

        t_branch1x1 = self.avg_pool(t_branch1x1)
        t_branch5x5 = self.avg_pool(t_branch5x5)
        t_branch3x3dbl = self.avg_pool(t_branch3x3dbl)
        t_branch_pool = self.avg_pool(t_branch_pool)

        t_branch1x1 = t_branch1x1.view(t_branch1x1.size(0), -1)
        t_branch5x5 = t_branch5x5.view(t_branch5x5.size(0), -1)
        t_branch3x3dbl = t_branch3x3dbl.view(t_branch3x3dbl.size(0), -1)
        t_branch_pool = t_branch_pool.view(t_branch_pool.size(0), -1)

        source = torch.cat([s_branch1x1, s_branch5x5, s_branch3x3dbl, s_branch_pool], 1)
        target = torch.cat([t_branch1x1, t_branch5x5, t_branch3x3dbl, t_branch_pool], 1)

        source = self.source_fc(source)
        t_label = self.source_fc(target)
        t_label = t_label.data.max(1)[1]

        loss = torch.Tensor([0])
        loss = loss.cuda()
        if self.training == True:
            loss += mmd.cmmd(s_branch1x1, t_branch1x1, s_label, t_label)
            loss += mmd.cmmd(s_branch5x5, t_branch5x5, s_label, t_label)
            loss += mmd.cmmd(s_branch3x3dbl, t_branch3x3dbl, s_label, t_label)
            loss += mmd.cmmd(s_branch_pool, t_branch_pool, s_label, t_label)
        return source, loss

def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model