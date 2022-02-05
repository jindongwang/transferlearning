import torch
import numpy as np


def pairwise_dist(X, Y):
    n, d = X.shape
    m, _ = Y.shape
    assert d == Y.shape[1]
    a = X.unsqueeze(1).expand(n, m, d)
    b = Y.unsqueeze(0).expand(n, m, d)
    return torch.pow(a - b, 2).sum(2)


def pairwise_dist_np(X, Y):
    n, d = X.shape
    m, _ = Y.shape
    assert d == Y.shape[1]
    a = np.expand_dims(X, 1)
    b = np.expand_dims(Y, 0)
    a = np.tile(a, (1, m, 1))
    b = np.tile(b, (n, 1, 1))
    return np.power(a - b, 2).sum(2)

def pa(X, Y):
    XY = np.dot(X, Y.T)
    XX = np.sum(np.square(X), axis=1)
    XX = np.transpose([XX])
    YY = np.sum(np.square(Y), axis=1)
    dist = XX + YY - 2 * XY

    return dist


if __name__ == '__main__':
    import sys
    args = sys.argv
    data = args[0]
    print(data)

    # a = torch.arange(1, 7).view(2, 3)
    # b = torch.arange(12, 21).view(3, 3)
    # print(pairwise_dist(a, b))

    # a = np.arange(1, 7).reshape((2, 3))
    # b = np.arange(12, 21).reshape((3, 3))
    # print(pa(a, b))