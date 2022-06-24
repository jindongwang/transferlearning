import argparse
import torch
import time
import sys
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchcontrib
import os
import os.path as osp
import random
import copy
from matplotlib import pyplot as plt

import logging

from PIL import Image
from pdb import set_trace as st

from torchvision import transforms

from dataset.cub200 import CUB200Data
from dataset.mit67 import MIT67Data
from dataset.stanford_dog import SDog120Data
from dataset.caltech256 import Caltech257Data
from dataset.stanford_40 import Stanford40Data
from dataset.flower102 import Flower102Data


from model.fe_resnet import resnet18_dropout, resnet34_dropout, resnet50_dropout, resnet101_dropout
from model.fe_mobilenet import mbnetv2_dropout
from model.fe_resnet import feresnet18, feresnet34, feresnet50, feresnet101
from model.fe_mobilenet import fembnetv2
from model.fe_vgg16 import *


from coverage.neuron_coverage import MyNeuronCoverage
# from DNNtest.coverage.my_neuron_coverage import MyNeuronCoverage
from coverage.pytorch_wrapper import PyTorchModel



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, default='/data', help='path to the dataset')
    parser.add_argument("--dataset", type=str, default='CUB200Data', help='Target dataset. Currently support: \{SDog120Data, CUB200Data, Stanford40Data, MIT67Data, Flower102Data\}')
    parser.add_argument("--name", type=str, default='test')
    parser.add_argument("--B", type=float, default=0.1, help='Attack budget')
    parser.add_argument("--m", type=float, default=1000, help='Hyper-parameter for task-agnostic attack')
    parser.add_argument("--pgd_iter", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--checkpoint", type=str, default='')
    parser.add_argument("--network", type=str, default='resnet18', help='Network architecture. Currently support: \{resnet18, resnet50, resnet101, mbnetv2\}')
    parser.add_argument("--teacher", default=None)
    parser.add_argument("--output_dir")

    parser.add_argument("--test_num", type=int, default=500)
    parser.add_argument("--num_try_per_sample", type=int, default=10)
    
    parser.add_argument("--sample_queue_length", type=int, default=10)
    parser.add_argument("--nc_threshold", type=float, default=0.5)
    parser.add_argument("--strategy", default="random", choices=["random", "deepxplore", "dlfuzz", "dlfuzzfirst"])
    parser.add_argument("--coverage", default="neuron_coverage")
    parser.add_argument("--k_select_neuron", type=int, default=20)
    parser.add_argument("--intermedia_mode", default="")
    
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--random_start", type=bool, default=True)
    parser.add_argument("--targeted", type=bool, default=False)
    
    args = parser.parse_args()
    if args.teacher is None:
        args.teacher = args.network
    return args

if __name__ == '__main__':
    seed = 98
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    np.set_printoptions(precision=4)
    
    args = get_args()
    # print(args)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    args.pid = os.getpid()
    args.log_path = osp.join(args.output_dir, "log.txt")
    if os.path.exists(args.log_path):
        log_lens = len(open(args.log_path, 'r').readlines())
        if log_lens > 5:
            print(f"{args.log_path} exists")
            exit()
    args.info = f"{args.strategy}_{args.coverage}_{args.dataset}_{args.network}"
    
    logging.basicConfig(filename=args.log_path, filemode="w", level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logger = logging.getLogger()
    logger.info(args)

    path = osp.join(args.output_dir, "accumulate_coverage.npy")
    with open(path, "rb") as f:
        accumulate_coverage = np.load(f, allow_pickle=True)
    
    path = osp.join(args.output_dir, "log_module_names.npy")
    with open(path, "rb") as f:
        log_names = np.load(f, allow_pickle=True)
        
    
    all_weight_coverage = []
    for layer_idx, (input_coverage, output_coverage) in enumerate(accumulate_coverage):
        input_dim, output_dim = len(input_coverage), len(output_coverage)
        for input_idx in range(input_dim):
            for output_idx in range(output_dim):
                all_weight_coverage.append(input_coverage[input_idx] + output_coverage[output_idx])
    
    prune_ratio = 0.05
    total = len(all_weight_coverage)
    sorted_coverage = np.sort(all_weight_coverage, )
    thre_index = int(total * prune_ratio)
    
    thre = sorted_coverage[thre_index]
    log = f"Pruning threshold: {thre:.4f}"
    print(log)
    
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

    prune_count = []
    for layer_index, module_name in enumerate(log_names):
        prune_count.append(len(prune_index[module_name]))
    print(prune_count)
    # st()


    prune_index = {}
    for layer_index, module_name in enumerate(log_names):
        prune_index[module_name] = []
        fig, axs = plt.subplots(1, 20, figsize=(100, 6), )
    
    for layer_idx, (input_coverage, output_coverage) in enumerate(accumulate_coverage):
        scores = []
        input_dim, output_dim = len(input_coverage), len(output_coverage)
        for input_idx in range(input_dim):
            for output_idx in range(output_dim):
                score = input_coverage[input_idx] + output_coverage[output_idx]
                # prune_index[module_name].append((input_idx, output_idx))
                scores.append(score)
        ax = axs[layer_idx]
        ax.hist(scores, bins=20)
        ax.set_title(f"{log_names[layer_idx]} min {np.min(scores)} max {np.max(scores)}")
                
    

    path = osp.join(args.output_dir, "hist.png")
    plt.tight_layout()
    plt.savefig(path)
    plt.clf()