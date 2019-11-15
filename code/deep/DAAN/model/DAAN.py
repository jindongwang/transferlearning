import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from functions import ReverseLayerF
from IPython import embed
import torch
import model.backbone as backbone

class DAANNet(nn.Module):

    def __init__(self, num_classes=65, base_net='ResNet50'):
        super(DAANNet, self).__init__()
        self.sharedNet = backbone.network_dict[base_net]()
        self.bottleneck = nn.Linear(2048, 256)
        self.source_fc = nn.Linear(256, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.classes = num_classes
        # global domain discriminator
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('fc1', nn.Linear(256, 1024))
        self.domain_classifier.add_module('relu1', nn.ReLU(True))
        self.domain_classifier.add_module('dpt1', nn.Dropout())
        self.domain_classifier.add_module('fc2', nn.Linear(1024, 1024))
        self.domain_classifier.add_module('relu2', nn.ReLU(True))
        self.domain_classifier.add_module('dpt2', nn.Dropout())
        self.domain_classifier.add_module('fc3', nn.Linear(1024, 2))

        # local domain discriminator
        self.dcis = nn.Sequential()
        self.dci = {}
        for i in range(num_classes):
            self.dci[i] = nn.Sequential()
            self.dci[i].add_module('fc1', nn.Linear(256, 1024))
            self.dci[i].add_module('relu1', nn.ReLU(True))
            self.dci[i].add_module('dpt1', nn.Dropout())
            self.dci[i].add_module('fc2', nn.Linear(1024, 1024))
            self.dci[i].add_module('relu2', nn.ReLU(True))
            self.dci[i].add_module('dpt2', nn.Dropout())
            self.dci[i].add_module('fc3', nn.Linear(1024, 2))
            self.dcis.add_module('dci_'+str(i), self.dci[i])

    def forward(self, source, target, s_label, DEV, alpha=0.0):
        source_share = self.sharedNet(source)
        source_share = self.bottleneck(source_share)
        source = self.source_fc(source_share)
        p_source = self.softmax(source)

        target = self.sharedNet(target)
        target = self.bottleneck(target)
        t_label = self.source_fc(target)
        p_target = self.softmax(t_label)
        t_label = t_label.data.max(1)[1]
        s_out = []
        t_out = []
        if self.training == True:
            # RevGrad
            s_reverse_feature = ReverseLayerF.apply(source_share, alpha)
            t_reverse_feature = ReverseLayerF.apply(target, alpha)
            s_domain_output = self.domain_classifier(s_reverse_feature)
            t_domain_output = self.domain_classifier(t_reverse_feature)

            # p*feature-> classifier_i ->loss_i
            for i in range(self.classes):
                ps = p_source[:, i].reshape((target.shape[0],1))
                fs = ps * s_reverse_feature
                pt = p_target[:, i].reshape((target.shape[0],1))
                ft = pt * t_reverse_feature
                outsi = self.dcis[i](fs)
                s_out.append(outsi)
                outti = self.dcis[i](ft)
                t_out.append(outti)
        else:
            s_domain_output = 0
            t_domain_output = 0
            s_out = [0]*self.classes
            t_out = [0]*self.classes
        return source, s_domain_output, t_domain_output, s_out, t_out
