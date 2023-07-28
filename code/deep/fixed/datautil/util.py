# coding=utf-8
import numpy as np
import torch
from collections import Counter


def Nmax(args, d):
    for i in range(len(args.test_envs)):
        if d < args.test_envs[i]:
            return i
    return len(args.test_envs)


class mydataset(object):
    def __init__(self, args):
        self.x = None
        self.labels = None
        self.dlabels = None
        self.task = None
        self.dataset = None
        self.transform = None
        self.target_transform = None
        self.args = args

    def set_labels(self, tlabels=None, label_type='domain_label'):
        assert len(tlabels) == len(self.x)
        if label_type == 'domain_label':
            self.dlabels = tlabels
        elif label_type == 'class_label':
            self.labels = tlabels

    def set_labels_by_index(self, tlabels=None, tindex=None, label_type='domain_label'):
        if label_type == 'domain_label':
            self.dlabels[tindex] = tlabels
        elif label_type == 'class_label':
            self.labels[tindex] = tlabels

    def target_trans(self, y):
        if self.target_transform is not None:
            return self.target_transform(y)
        else:
            return y

    def input_trans(self, x):
        if self.transform is not None:
            return self.transform(x)
        else:
            return x

    def __getitem__(self, index):
        x = self.input_trans(self.x[index])
        ctarget = self.target_trans(self.labels[index])
        dtarget = self.target_trans(self.dlabels[index])
        return x, ctarget, dtarget, index

    def __len__(self):
        return len(self.x)


class subdataset(mydataset):
    def __init__(self, args, dataset, indices):
        super(subdataset, self).__init__(args)
        self.x = dataset.x[indices]
        self.labels = dataset.labels[indices]
        self.dlabels = dataset.dlabels[indices] if dataset.dlabels is not None else None
        self.task = dataset.task
        self.dataset = dataset.dataset
        self.transform = dataset.transform
        self.target_transform = dataset.target_transform


def split_trian_val_test(args, da, rate=0.8):
    dsize = len(da)
    tr = int(rate*dsize)
    indexall = np.arange(dsize)
    np.random.seed(args.seed)
    np.random.shuffle(indexall)
    indextr, indexte = indexall[:tr], indexall[tr:]
    tr_da = subdataset(args, da, indextr)
    te_da = subdataset(args, da, indexte)
    return tr_da, te_da


def make_weights_for_balanced_classes(dataset):
    counts = Counter()
    classes = []
    for _, y, _, _, _, _ in dataset:
        y = int(y)
        counts[y] += 1
        classes.append(y)

    n_classes = len(counts)

    weight_per_class = {}
    for y in counts:
        weight_per_class[y] = 1 / (counts[y] * n_classes)

    weights = torch.zeros(len(dataset))
    for i, y in enumerate(classes):
        weights[i] = weight_per_class[int(y)]

    return weights


def random_pairs_of_minibatches_by_domainperm(minibatches):
    perm = torch.randperm(len(minibatches)).tolist()
    pairs = []

    for i in range(len(minibatches)):
        j = i + 1 if i < (len(minibatches) - 1) else 0

        xi, yi, di, = minibatches[perm[i]
                                  ][0], minibatches[perm[i]][1], minibatches[perm[i]][2]
        xj, yj, dj = minibatches[perm[j]
                                 ][0], minibatches[perm[j]][1], minibatches[perm[j]][2]

        min_n = min(len(xi), len(xj))

        pairs.append(((xi[:min_n], yi[:min_n], di[:min_n]),
                     (xj[:min_n], yj[:min_n], dj[:min_n])))

    return pairs
