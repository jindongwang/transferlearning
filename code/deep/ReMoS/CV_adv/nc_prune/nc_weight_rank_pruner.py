import os
import os.path as osp
import sys
import time
import argparse
from pdb import set_trace as st
import json
import random
import time
import pickle 

import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchcontrib

from torchvision import transforms
from advertorch.attacks import LinfPGDAttack

from dataset.cub200 import CUB200Data
from dataset.mit67 import MIT67Data
from dataset.stanford_dog import SDog120Data
from dataset.stanford_40 import Stanford40Data
from dataset.flower102 import Flower102Data

from model.fe_resnet import feresnet18, feresnet50, feresnet101

from eval_robustness import advtest, myloss
from utils import *
from .nc_pruner import NCPruner

class NCWeightRankPruner(NCPruner):
    def __init__(
        self,
        args,
        model,
        teacher,
        train_loader,
        test_loader,
    ):
        super(NCWeightRankPruner, self).__init__(
            args, model, teacher, train_loader, test_loader
        )

    def load_nc_info(self,):
        path = osp.join(self.args.nc_info_dir, "accumulate_coverage.pkl")
        with open(path, "rb") as f:
            accumulate_coverage = pickle.load(f, )
        
        path = osp.join(self.args.nc_info_dir, "log_module_names.pkl")
        with open(path, "rb") as f:
            log_names = pickle.load(f, )
        return accumulate_coverage, log_names



    def weight_prune(
        self,
        prune_ratio,
        random_prune=False,
    ):
        
        model = self.model.cpu()
        
        total_weight = 0
        layer_to_rank = {}
        for name, module in model.named_modules():
            if ( isinstance(module, nn.Conv2d) ):
                total_weight += module.weight.numel()
                layer_to_rank[name] = module.weight.data.clone().numpy()
                layer_to_rank[name].fill(0)
        
        
        accumulate_coverage, log_names = self.load_nc_info()
        all_weight_coverage, adv_weight_coverage = [], []
        for layer_name, (input_coverage, output_coverage) in accumulate_coverage.items():
            input_dim, output_dim = len(input_coverage), len(output_coverage)
            for input_idx in range(input_dim):
                for output_idx in range(output_dim):
                    coverage_score = input_coverage[input_idx] + output_coverage[output_idx]
                    all_weight_coverage.append((coverage_score,  (layer_name, input_idx, output_idx)))


        # prune_ratio = 0.05
        
        sorted_coverage = sorted(all_weight_coverage, key=lambda item: item[0])
        
        accumulate_index = 0
        for (coverage_score, pos) in sorted_coverage:
            layer_name, input_idx, output_idx = pos
            layer_to_rank[layer_name][output_idx, input_idx] = accumulate_index
            h, w = layer_to_rank[layer_name].shape[2:]
            accumulate_index += h*w
        
        start = time.time()
        layer_idx = 0
        weight_list = []
        for name, module in model.named_modules():
            if ( isinstance(module, nn.Conv2d) ):
                weight_copy = module.weight.data.abs().clone().numpy()
                output_dim, input_dim, h, w = weight_copy.shape
                for output_idx in range(output_dim):
                    for input_idx in range(input_dim):
                        for h_idx in range(h):
                            for w_idx in range(w):
                                
                                weight_score = weight_copy[output_idx, input_idx, h_idx, w_idx]
                                weight_list.append( (weight_score, (layer_idx, input_idx, output_idx, h_idx, w_idx)) )
                layer_idx += 1
        sorted_weight = sorted(weight_list, key=lambda item: item[0])
        end = time.time()
        weight_sort_time = end - start
        log = f"Sort weight time {weight_sort_time}"
        self.prune_record(log)
        
        for weight_rank, (weight_score, pos) in enumerate(sorted_weight):
            layer_idx, input_idx, output_idx, h_idx, w_idx = pos
            layer_name = log_names[layer_idx]
            layer_to_rank[layer_name][output_idx, input_idx, h_idx, w_idx] -= weight_rank
        
        start = time.time()
        nc_weight_ranks = []
        for layer_name in log_names:
            nc_weight_ranks.append( layer_to_rank[layer_name].flatten() )
        nc_weight_ranks = np.concatenate(nc_weight_ranks)
        nc_weight_ranks = np.sort(nc_weight_ranks)
        end = time.time()
        weight_sort_time = end - start
        log = f"Sort nc weight rank time {weight_sort_time}"
        self.prune_record(log)

        
        total = len(nc_weight_ranks)
        thre_index = int(total * prune_ratio)
        
        if thre_index == total:
            thre_index -= 1
        thre = nc_weight_ranks[thre_index]
        log = f"Pruning threshold: {thre:.4f}"
        self.prune_record(log)
        

        pruned = 0
        for name, module in model.named_modules():
            if ( isinstance(module, nn.Conv2d) ):
                mask = layer_to_rank[name]
                mask = torch.Tensor(mask > thre)
                

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
