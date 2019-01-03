# encoding=utf-8
"""
    Created on 10:29 2018/12/29 
    @author: Jindong Wang
"""

import torch.nn as nn


class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(3, 64, kernel_size=5))
        self.feature.add_module('f_bn1', nn.BatchNorm2d(64))
        self.feature.add_module('f_pool1', nn.MaxPool2d(2))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_conv2', nn.Conv2d(64, 50, kernel_size=5))
        self.feature.add_module('f_bn2', nn.BatchNorm2d(50))
        self.feature.add_module('f_drop1', nn.Dropout2d())
        self.feature.add_module('f_pool2', nn.MaxPool2d(2))
        self.feature.add_module('f_relu2', nn.ReLU(True))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(50 * 5 * 5, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 500))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(500))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(500, 10))

    def forward(self, input_data):
        # input_data = input_data.expand(len(input_data), 3, 28, 28)
        feature = self.feature(input_data)
        feature = feature.view(-1, 50 * 5 * 5)
        class_output = self.class_classifier(feature)

        return class_output


    # Exactly like forward function, but return features
    def get_feature(self, input_data):
        # input_data = input_data.expand(len(input_data), 3, 28, 28)
        feature = self.feature(input_data)
        feature = feature.view(-1, 50 * 5 * 5)
        fea = self.class_classifier.c_fc1(feature)
        fea = self.class_classifier.c_bn1(fea)
        fea = self.class_classifier.c_relu1(fea)
        fea = self.class_classifier.c_drop1(fea)
        fea = self.class_classifier.c_fc2(fea)
        fea = self.class_classifier.c_bn2(fea)
        fea = self.class_classifier.c_relu2(fea)

        return fea
