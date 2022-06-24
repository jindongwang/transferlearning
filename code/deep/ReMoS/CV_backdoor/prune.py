import os
import os.path as osp
import sys
import time
import argparse
from pdb import set_trace as st
import json
import random

import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchcontrib


def weight_prune(
        model,
        prune_ratio,
):
    total = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            total += module.weight.data.numel()

    conv_weights = torch.zeros(total).cuda()
    index = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            size = module.weight.data.numel()
            conv_weights[index:(index + size)] = module.weight.data.view(-1).abs().clone()
            index += size

    y, i = torch.sort(conv_weights)
    thre_index = int(total * prune_ratio)
    thre = y[thre_index]
    log = f"Pruning threshold: {thre:.4f}"
    print(log)

    pruned = 0
    zero_flag = False
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            weight_copy = module.weight.data.abs().clone()
            mask = weight_copy.gt(thre).float()

            pruned = pruned + mask.numel() - torch.sum(mask)
            # Save the update mask to the module
            module.mask = mask
            if int(torch.sum(mask)) == 0:
                zero_flag = True
            remain_ratio = int(torch.sum(mask)) / mask.numel()
            log = (f"layer {name} \t total params: {mask.numel()} \t "
                   f"not update params: {int(torch.sum(mask))}({remain_ratio:.2f})")
            print(log)

    if zero_flag:
        raise RuntimeError("There exists a layer with 0 parameters left.")
    log = (f"Total conv params: {total}, not update conv params: {pruned}, "
           f"not update ratio: {pruned / total:.2f}")
    print(log)
    return model
