import os
import os.path as osp
import sys
import time
import argparse
from pdb import set_trace as st
import json

import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchcontrib

from torchvision import transforms




class MovingAverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', momentum=0.9):
        self.name = name
        self.fmt = fmt
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0

    def update(self, val, n=1):
        self.val = val
        self.avg = self.momentum*self.avg + (1-self.momentum)*val

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", output_dir=None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        if output_dir is not None:
            self.filepath = osp.join(output_dir, "progress")

            

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        log_str = '\t'.join(entries)
        print(log_str)
        if self.filepath is not None:
            with open(self.filepath, "a") as f:
                f.write(log_str+"\n")

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
    
class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon = 0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).sum(1)
        return loss.mean()


def linear_l2(model, beta_lmda):
    beta_loss = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            beta_loss += (m.weight).pow(2).sum()
            beta_loss += (m.bias).pow(2).sum()
    return 0.5*beta_loss*beta_lmda, beta_loss

def l2sp(model, reg):
    reg_loss = 0
    dist = 0
    for m in model.modules():
        if hasattr(m, 'weight') and hasattr(m, 'old_weight'):
            diff = (m.weight - m.old_weight).pow(2).sum()
            dist += diff
            reg_loss += diff 

        if hasattr(m, 'bias') and hasattr(m, 'old_bias'):
            diff = (m.bias - m.old_bias).pow(2).sum()
            dist += diff
            reg_loss += diff 

    if dist > 0:
        dist = dist.sqrt()
    
    loss = (reg * reg_loss)
    return loss, dist

def advtest_fast(model, loader, adversary, args):
    advDataset = torch.load(args.adv_data_dir)
    test_loader = torch.utils.data.DataLoader(
        advDataset,
        batch_size=4, shuffle=False,
        num_workers=0, pin_memory=False)
    model.eval()

    total_ce = 0
    total = 0
    top1 = 0

    total = 0
    top1_clean = 0
    top1_adv = 0
    adv_success = 0
    adv_trial = 0
    for i, (batch, label, adv_batch, adv_label) in enumerate(test_loader):

        total += batch.size(0)
        out_clean = model(batch)

        out_adv = model(adv_batch)

        _, pred_clean = out_clean.max(dim=1)
        _, pred_adv = out_adv.max(dim=1)

        clean_correct = pred_clean.eq(label)
        adv_trial += int(clean_correct.sum().item())
        adv_success += int(pred_adv[clean_correct].eq(label[clean_correct]).sum().detach().item())
        top1_clean += int(pred_clean.eq(label).sum().detach().item())
        top1_adv += int(pred_adv.eq(label).sum().detach().item())

        # print('{}/{}...'.format(i+1, len(test_loader)))
    print(f"Finish adv test fast")
    del test_loader
    del advDataset
    return float(top1_clean)/total*100, float(top1_adv)/total*100, float(adv_trial-adv_success) / adv_trial *100
