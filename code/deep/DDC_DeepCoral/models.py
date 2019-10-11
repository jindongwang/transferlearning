import torch.nn as nn
import torchvision
from Coral import CORAL
import mmd

fc_layer = {
    'alexnet': 256 * 6 * 6,
    'resnet50': 2048
}

class DeepCoral(nn.Module):
    def __init__(self, num_classes, backbone='resnet50', adapt_loss='mmd'):
        super(DeepCoral, self).__init__()
        self.isTrain = True
        self.backbone = backbone
        self.adapt_loss = adapt_loss
        if self.backbone == 'resnet50':
            model_resnet = torchvision.models.resnet50(pretrained=True)
            self.conv1 = model_resnet.conv1
            self.bn1 = model_resnet.bn1
            self.relu = model_resnet.relu
            self.maxpool = model_resnet.maxpool
            self.layer1 = model_resnet.layer1
            self.layer2 = model_resnet.layer2
            self.layer3 = model_resnet.layer3
            self.layer4 = model_resnet.layer4
            self.avgpool = model_resnet.avgpool
            self.sharedNet = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                                self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)
        
            n_features = model_resnet.fc.in_features
            self.cls_fc = nn.Linear(n_features, num_classes)
            self.fc = nn.Linear(1,1)
        elif self.backbone == 'alexnet':
            model_alexnet = torchvision.models.alexnet(pretrained=True)
            self.sharedNet = model_alexnet.features
            self.fc = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
            )
            self.cls_fc = nn.Linear(4096, num_classes)
        self.cls_fc.weight.data.normal_(0, 0.005)
        
    def compute_adapt_loss(self, X, Y, adapt_loss):
        """Compute adaptation loss, currently we support mmd and coral
        
        Arguments:
            X {tensor} -- source matrix
            Y {tensor} -- target matrix
            adapt_loss {string} -- loss type, 'mmd' or 'coral'. You can add your own loss
        
        Returns:
            [tensor] -- adaptation loss tensor
        """
        if adapt_loss == 'mmd':
            mmd_loss = mmd.MMD_loss()
            loss = mmd_loss(X, Y)
        elif adapt_loss == 'coral':
            loss = CORAL(X, Y)
        else:
            loss = 0
        return loss

    def forward(self, source, target):
        adapt_loss = 0
        source = self.sharedNet(source)
        source = source.view(source.size(0), fc_layer[self.backbone])
        if self.backbone == 'alexnet':
            source = self.fc(source)
        if self.isTrain:
            target = self.sharedNet(target)
            target = target.view(target.size(0), fc_layer[self.backbone])
            if self.backbone == 'alexnet':
                target = self.fc(target)
            adapt_loss = self.compute_adapt_loss(source, target, adapt_loss = self.adapt_loss)
        clf = self.cls_fc(source)
        return clf, adapt_loss

    