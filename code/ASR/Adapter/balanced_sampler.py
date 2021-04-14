import torch
import torch.utils.data
import random
import collections
import logging

import numpy as np
from torch.utils.data.sampler import BatchSampler, WeightedRandomSampler

# https://github.com/khornlund/pytorch-balanced-sampler
class BalancedBatchSampler(torch.utils.data.sampler.Sampler):
    '''
    https://github.com/galatolofederico/pytorch-balanced-batch/blob/master/sampler.py
    '''
    def __init__(self, dataset, labels=None):
        self.labels = labels
        self.dataset = collections.defaultdict(list)
        self.balanced_max = 0
        # Save all the indices for all the classes
        for idx in range(0, len(dataset)):
            label = self._get_label(dataset, idx)
            #break
            self.dataset[label].append(idx)
            self.balanced_max = max(self.balanced_max, len(self.dataset[label]))
            #len(self.dataset[label]) if len(self.dataset[label]) > self.balanced_max else self.balanced_max
        
        # Oversample the classes with fewer elements than the max
        for label in self.dataset:
            while len(self.dataset[label]) < self.balanced_max:
                self.dataset[label].append(random.choice(self.dataset[label]))
        self.keys = list(self.dataset.keys())
        logging.warning(self.keys)
        self.currentkey = 0
        self.indices = [-1] * len(self.keys)

    def __iter__(self):
        while self.indices[self.currentkey] < self.balanced_max - 1:
            self.indices[self.currentkey] += 1
            yield self.dataset[self.keys[self.currentkey]][self.indices[self.currentkey]]
            self.currentkey = (self.currentkey + 1) % len(self.keys)
        self.indices = [-1] * len(self.keys)
    
    def _get_label(self, dataset, idx):
        #logging.warning(len(dataset))
        # logging.warning(dataset[idx])
        return dataset[idx][0][1]['category']#[1]['output'][0]['token'].split(' ')[1]
    # def _get_label(self, dataset, idx, labels = None):
    #     if self.labels is not None:
    #         return self.labels[idx].item()
    #     else:
    #         raise Exception("You should pass the tensor of labels to the constructor as second argument")

    def __len__(self):
        return self.balanced_max * len(self.keys)

