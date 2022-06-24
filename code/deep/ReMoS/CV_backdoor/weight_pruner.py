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


from model.fe_resnet import resnet18_dropout, resnet34_dropout, resnet50_dropout, resnet101_dropout
from model.fe_resnet import feresnet18, feresnet34, feresnet50, feresnet101

from utils import *
from finetuner import Finetuner

class WeightPruner(Finetuner):
    def __init__(
        self,
        args,
        model,
        teacher,
        train_loader,
        test_loader,
    ):
        super(WeightPruner, self).__init__(
            args, model, teacher, train_loader, test_loader, "ckpt"
        )
        assert (
            self.args.weight_total_ratio >= 0 and
            self.args.weight_ratio_per_prune >= 0 and 
            self.args.prune_interval >= 0 and 
            self.args.weight_init_prune_ratio >= 0 and
            self.args.weight_total_ratio >= self.args.weight_init_prune_ratio
        )
        self.log_path = osp.join(self.args.output_dir, "prune.log")
        self.logger = open(self.log_path, "w")
        self.init_prune()
        self.logger.close()

    def prune_record(self, log):
        print(log)
        self.logger.write(log+"\n")

    def init_prune(self):
        ratio = self.args.weight_init_prune_ratio
        log = f"Init prune ratio {ratio:.2f}"
        self.prune_record(log)
        self.weight_prune(ratio)
        self.check_param_num()

    def iterative_prune(self, iteration):
        if iteration == 0:
            return
        init = self.args.weight_init_prune_ratio
        interval = self.args.prune_interval
        per_ratio = self.args.weight_ratio_per_prune
        ratio = init + per_ratio * (iteration / interval)
        if ratio - self.args.weight_total_ratio > per_ratio:
            return
        self.logger = open(self.log_path, "a")
        ratio = min(
            self.args.weight_total_ratio,
            ratio
        )
        log = f"Iteration {iteration}, prune ratio {ratio}"
        self.prune_record(log)
        self.weight_prune(ratio)
        self.check_param_num()
        self.logger.close()

    def check_param_num(self):
        model = self.model        
        total = sum([module.weight.nelement() for module in model.modules() if isinstance(module, nn.Conv2d) ])
        num = total
        for m in model.modules():
            if ( isinstance(m, nn.Conv2d) ):
                num -= int((m.weight.data == 0).sum())
        ratio = (total - num) / total
        log = f"===>Check: Total {total}, current {num}, prune ratio {ratio:2f}"
        self.prune_record(log)


    def weight_prune(
        self,
        prune_ratio,
        random_prune=False,
    ):
        model = self.model.cpu()
        total = 0
        for name, module in model.named_modules():
            if ( isinstance(module, nn.Conv2d) ):
                    total += module.weight.data.numel()
        
        conv_weights = torch.zeros(total)
        index = 0
        for name, module in model.named_modules():
            if ( isinstance(module, nn.Conv2d) ):
                size = module.weight.data.numel()
                conv_weights[index:(index+size)] = module.weight.data.view(-1).abs().clone()
                index += size
        
        y, i = torch.sort(conv_weights, descending=self.args.prune_descending)
        # thre_index = int(total * prune_ratio)
        # thre = y[thre_index]
        thre_index = int(total * prune_ratio)
        thre = y[thre_index]
        log = f"Pruning threshold: {thre:.4f}"
        self.prune_record(log)

        pruned = 0
        
        zero_flag = False
        
        for name, module in model.named_modules():
            if ( isinstance(module, nn.Conv2d) ):
                weight_copy = module.weight.data.abs().clone()
                # if self.args.prune_descending:
                #     mask = weight_copy.lt(thre).float()
                # else:
                #     mask = weight_copy.gt(thre).float()

                if random_prune:
                    print(f"Random prune {name}")
                    mask = np.zeros(weight_copy.numel()) + 1
                    prune_number = round(prune_ratio * weight_copy.numel())
                    mask[:prune_number] = 0
                    np.random.shuffle(mask)
                    mask = mask.reshape(weight_copy.shape)
                    mask = torch.Tensor(mask)

                pruned = pruned + mask.numel() - torch.sum(mask)
                # np.random.shuffle(mask)
                module.weight.data.mul_(mask)
                if int(torch.sum(mask)) == 0:
                    zero_flag = True
                remain_ratio = int(torch.sum(mask)) / mask.numel()
                log = (f"layer {name} \t total params: {mask.numel()} \t "
                f"remaining params: {int(torch.sum(mask))}({remain_ratio:.2f})")
                self.prune_record(log)
                
        if zero_flag:
            raise RuntimeError("There exists a layer with 0 parameters left.")
        log = (f"Total conv params: {total}, Pruned conv params: {pruned}, "
        f"Pruned ratio: {pruned/total:.2f}")
        self.prune_record(log)
        self.model = model.cuda()
        

    def final_check_param_num(self):
        self.logger = open(self.log_path, "a")
        self.check_param_num()
        self.logger.close()
