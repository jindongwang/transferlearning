# coding=utf-8
from torchvision import transforms
import numpy as np


def act_train():
    return transforms.Compose([
        transforms.ToTensor()
    ])


def act_test():
    return transforms.Compose([
        transforms.ToTensor()
    ])


def loaddata_from_numpy(dataset='dsads', root_dir='./data/act/'):
    x = np.load(root_dir+dataset+'/'+dataset+'_x.npy')
    ty = np.load(root_dir+dataset+'/'+dataset+'_y.npy')
    cy, py, sy = ty[:, 0], ty[:, 1], ty[:, 2]
    return x, cy, py, sy
