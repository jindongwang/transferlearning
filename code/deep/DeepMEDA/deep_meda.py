import torch
import torch.nn as nn
import ResNet
import mmd
from Config import bottle_neck
import proxy_a_distance

class DeepMEDA(nn.Module):

    def __init__(self, num_classes=31):
        super(DeepMEDA, self).__init__()
        self.feature_layers = ResNet.resnet50(True)
        self.mmd_marginal = mmd.MMD_loss()
        if bottle_neck:
            self.bottle = nn.Linear(2048, 256)
            self.cls_fc = nn.Linear(256, num_classes)
        else:
            self.cls_fc = nn.Linear(2048, num_classes)


    def forward(self, source, target, s_label):
        source = self.feature_layers(source)
        if bottle_neck:
            source = self.bottle(source)
        s_pred = self.cls_fc(source)
        if self.training ==True:
            target = self.feature_layers(target)
            if bottle_neck:
                target = self.bottle(target)
            t_label = self.cls_fc(target)
            loss_c = mmd.lmmd(source, target, s_label, torch.nn.functional.softmax(t_label, dim=1))
            loss_m = self.mmd_marginal(source, target)
            mu = proxy_a_distance.estimate_mu(source.detach().cpu().numpy(), s_label.detach().cpu().numpy(), target.detach().cpu().numpy(), torch.max(t_label, 1)[1].detach().cpu().numpy())
        else:
            loss_c, loss_m, mu = 0
        return s_pred, loss_c, loss_m, mu

    def predict(self, x):
        x = self.feature_layers(x)
        if bottle_neck:
            x = self.bottle(x)
        return self.cls_fc(x)