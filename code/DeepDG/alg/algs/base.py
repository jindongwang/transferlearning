# coding=utf-8
import torch


class Algorithm(torch.nn.Module):

    def __init__(self, args):
        super(Algorithm, self).__init__()

    def update(self, minibatches, opt, sch):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError
