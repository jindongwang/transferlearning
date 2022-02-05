import torch
import torch.nn as nn
import ResNet
import mmd
import dynamic_factor

class DeepMEDA(nn.Module):

    def __init__(self, num_classes=31, bottle_neck=True):
        super(DeepMEDA, self).__init__()
        self.feature_layers = ResNet.resnet50(True)
        self.mmd_loss = mmd.MMD_loss()
        self.bottle_neck = bottle_neck
        if bottle_neck:
            self.bottle = nn.Linear(2048, 256)
            self.cls_fc = nn.Linear(256, num_classes)
        else:
            self.cls_fc = nn.Linear(2048, num_classes)


    def forward(self, source, target, s_label):
        source = self.feature_layers(source)
        if self.bottle_neck:
            source = self.bottle(source)
        s_pred = self.cls_fc(source)
        target = self.feature_layers(target)
        if self.bottle_neck:
            target = self.bottle(target)
        t_label = self.cls_fc(target)
        loss_c = self.mmd_loss.conditional(source, target, s_label, torch.nn.functional.softmax(t_label, dim=1))
        loss_m = self.mmd_loss.marginal(source, target)
        mu = dynamic_factor.estimate_mu(source.detach().cpu().numpy(), s_label.detach().cpu().numpy(), target.detach().cpu().numpy(), torch.max(t_label, 1)[1].detach().cpu().numpy())
        return s_pred, loss_c, loss_m, mu

    def predict(self, x):
        x = self.feature_layers(x)
        if self.bottle_neck:
            x = self.bottle(x)
        return self.cls_fc(x)