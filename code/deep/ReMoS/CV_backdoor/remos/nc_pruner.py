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

from torchvision import transforms
from advertorch.attacks import LinfPGDAttack

from model.fe_resnet import resnet18_dropout, resnet50_dropout, resnet101_dropout
from model.fe_resnet import feresnet18, feresnet50, feresnet101

from utils import *
from weight_pruner import WeightPruner

class NCPruner(WeightPruner):
    def __init__(
        self,
        args,
        model,
        teacher,
        train_loader,
        test_loader,
    ):
        super(NCPruner, self).__init__(
            args, model, teacher, train_loader, test_loader
        )

    def prune_record(self, log):
        print(log)
        self.logger.write(log+"\n")

    def init_prune(self):
        ratio = self.args.weight_init_prune_ratio
        log = f"Init prune ratio {ratio:.2f}"
        self.prune_record(log)
        self.weight_prune(ratio)
        self.check_param_num()

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
        
    def load_nc_info(self,):
        path = osp.join(self.args.nc_info_dir, "accumulate_coverage.npy")
        with open(path, "rb") as f:
            accumulate_coverage = np.load(f, allow_pickle=True)
        
        path = osp.join(self.args.nc_info_dir, "log_module_names.npy")
        with open(path, "rb") as f:
            log_names = np.load(f, allow_pickle=True)
        return accumulate_coverage, log_names
        st()


    def weight_prune(
        self,
        prune_ratio,
        random_prune=False,
    ):
        
        model = self.model.cpu()
        
        total_weight = 0
        for name, module in model.named_modules():
            if ( isinstance(module, nn.Conv2d) ):
                total_weight += module.weight.numel()
        
        accumulate_coverage, log_names = self.load_nc_info()
        all_weight_coverage = []
        for layer_idx, (input_coverage, output_coverage) in enumerate(accumulate_coverage):
            input_dim, output_dim = len(input_coverage), len(output_coverage)
            for input_idx in range(input_dim):
                for output_idx in range(output_dim):
                    all_weight_coverage.append(input_coverage[input_idx] + output_coverage[output_idx])
        
        # prune_ratio = 0.05
        total = len(all_weight_coverage)
        sorted_coverage = np.sort(all_weight_coverage, )
        thre_index = int(total * prune_ratio)
        
        if thre_index == total:
            thre_index -= 1
        thre = sorted_coverage[thre_index]
        log = f"Pruning threshold: {thre:.4f}"
        self.prune_record(log)
        
        prune_index = {}
        for layer_index, module_name in enumerate(log_names):
            prune_index[module_name] = []
        
        for layer_idx, (input_coverage, output_coverage) in enumerate(accumulate_coverage):
            input_dim, output_dim = len(input_coverage), len(output_coverage)
            module_name = log_names[layer_idx]
            for input_idx in range(input_dim):
                for output_idx in range(output_dim):
                    score = input_coverage[input_idx] + output_coverage[output_idx]
                    if score < thre:
                        prune_index[module_name].append((input_idx, output_idx))

        pruned = 0
        for name, module in model.named_modules():
            if ( isinstance(module, nn.Conv2d) ):
                weight_copy = module.weight.data.abs().clone()
                assert name in prune_index, f"{name} not in log names"
                if len(prune_index[name]) == 0:
                    continue
                for (input_idx, output_idx) in prune_index[name]:
                    weight_copy[output_idx, input_idx] -= weight_copy[output_idx, input_idx]
                weight_copy[weight_copy!=0] = 1.
                mask = weight_copy

                pruned = pruned + mask.numel() - torch.sum(mask)
                # np.random.shuffle(mask)
                module.weight.data.mul_(mask)

                remain_ratio = int(torch.sum(mask)) / mask.numel()
                log = (f"layer {name} \t total params: {mask.numel()} \t "
                f"remaining params: {int(torch.sum(mask))}({remain_ratio:.2f})")
                self.prune_record(log)
                

        log = (f"Total conv params: {total_weight}, Pruned conv params: {pruned}, "
        f"Pruned ratio: {pruned/total_weight:.2f}")
        self.prune_record(log)
        self.model = model.cuda()

        self.check_param_num()




    def final_check_param_num(self):
        self.logger = open(self.log_path, "a")
        self.check_param_num()
        self.logger.close()
