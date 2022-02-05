import torch
import torch.nn as nn
import backbone


class Network(nn.Module):
    def __init__(self, base_net='alexnet', n_class=31):
        super(Network, self).__init__()
        self.n_class = n_class
        self.base_network = backbone.network_dict[base_net]()
        self.classifier_layer = nn.Linear(
            self.base_network.output_num(), n_class)
        self.classifier_layer.weight.data.normal_(0, 0.005)
        self.classifier_layer.bias.data.fill_(0.1)

    def forward(self, x):
        features = self.base_network(x)
        clf = self.classifier_layer(features)
        return clf

    def get_features(self, x):
        features = self.base_network(x)
        return features
