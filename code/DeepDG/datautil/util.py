# coding=utf-8
import numpy as np
import torch


def Nmax(test_envs, d):
    for i in range(len(test_envs)):
        if d < test_envs[i]:
            return i
    return len(test_envs)


def random_pairs_of_minibatches_by_domainperm(minibatches):
    perm = torch.randperm(len(minibatches)).tolist()
    pairs = []

    for i in range(len(minibatches)):
        j = i + 1 if i < (len(minibatches) - 1) else 0

        xi, yi = minibatches[perm[i]][0], minibatches[perm[i]][1]
        xj, yj = minibatches[perm[j]][0], minibatches[perm[j]][1]

        min_n = min(len(xi), len(xj))

        pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs


def random_pairs_of_minibatches(args, minibatches):
    ld = len(minibatches)
    pairs = []
    tdlist = np.arange(ld)
    txlist = np.arange(args.batch_size)
    for i in range(ld):
        for j in range(args.batch_size):
            (tdi, tdj), (txi, txj) = np.random.choice(tdlist, 2,
                                                      replace=False), np.random.choice(txlist, 2, replace=True)
            if j == 0:
                xi, yi, di = torch.unsqueeze(
                    minibatches[tdi][0][txi], dim=0), minibatches[tdi][1][txi], minibatches[tdi][2][txi]
                xj, yj, dj = torch.unsqueeze(
                    minibatches[tdj][0][txj], dim=0), minibatches[tdj][1][txj], minibatches[tdj][2][txj]
            else:
                xi, yi, di = torch.vstack((xi, torch.unsqueeze(minibatches[tdi][0][txi], dim=0))), torch.hstack(
                    (yi, minibatches[tdi][1][txi])), torch.hstack((di, minibatches[tdi][2][txi]))
                xj, yj, dj = torch.vstack((xj, torch.unsqueeze(minibatches[tdj][0][txj], dim=0))), torch.hstack(
                    (yj, minibatches[tdj][1][txj])), torch.hstack((dj, minibatches[tdj][2][txj]))
        pairs.append(((xi, yi, di), (xj, yj, dj)))
    return pairs
