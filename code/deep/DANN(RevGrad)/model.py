import torch
import torch.nn as nn
import adv_layer


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        feature = nn.Sequential()
        feature.add_module('f_conv1', nn.Conv2d(3, 64, kernel_size=5))
        feature.add_module('f_bn1', nn.BatchNorm2d(64))
        feature.add_module('f_pool1', nn.MaxPool2d(2))
        feature.add_module('f_relu1', nn.ReLU(True))
        feature.add_module('f_conv2', nn.Conv2d(64, 50, kernel_size=5))
        feature.add_module('f_bn2', nn.BatchNorm2d(50))
        feature.add_module('f_drop1', nn.Dropout2d())
        feature.add_module('f_pool2', nn.MaxPool2d(2))
        feature.add_module('f_relu2', nn.ReLU(True))
        self.feature = feature

    def forward(self, x):
        return self.feature(x)


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(50 * 4 * 4, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 10))

    def forward(self, x):
        return self.class_classifier(x)


class DANN(nn.Module):

    def __init__(self, device):
        super(DANN, self).__init__()
        self.device = device
        self.feature = FeatureExtractor()
        self.classifier = Classifier()
        self.domain_classifier = adv_layer.Discriminator(
            input_dim=50 * 4 * 4, hidden_dim=100)

    def forward(self, input_data, alpha=1, source=True):
        input_data = input_data.expand(len(input_data), 3, 28, 28)
        feature = self.feature(input_data)
        feature = feature.view(-1, 50 * 4 * 4)
        class_output = self.classifier(feature)
        domain_output = self.get_adversarial_result(
            feature, source, alpha)
        return class_output, domain_output

    def get_adversarial_result(self, x, source=True, alpha=1):
        loss_fn = nn.BCELoss()
        if source:
            domain_label = torch.ones(len(x)).long().to(self.device)
        else:
            domain_label = torch.zeros(len(x)).long().to(self.device)
        x = adv_layer.ReverseLayerF.apply(x, alpha)
        domain_pred = self.domain_classifier(x)
        loss_adv = loss_fn(domain_pred, domain_label.float())
        return loss_adv
