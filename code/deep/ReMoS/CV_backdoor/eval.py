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
import torch.utils.data
import torchcontrib

from torchvision import transforms
import copy


sys.path.append('../CV_adv')

from model.fe_resnet import resnet18_dropout, resnet50_dropout, resnet101_dropout
from model.fe_resnet import feresnet18, feresnet50, feresnet101

from eval_robustness import advtest, myloss
from utils import *

from weight_pruner import WeightPruner
from finetuner import Finetuner as RawFinetuner

from attack_finetuner import AttackFinetuner
from prune import weight_prune
from finetuner import Finetuner
# from nc_prune.nc_weight_rank_pruner import NCWeightRankPruner


from backdoor_dataset.cub200 import CUB200Data
from backdoor_dataset.mit67 import MIT67Data
from backdoor_dataset.stanford_dog import SDog120Data
from backdoor_dataset.caltech256 import Caltech257Data
from backdoor_dataset.stanford_40 import Stanford40Data
from backdoor_dataset.flower102 import Flower102Data
from trigger import teacher_train

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, default='../data/LISA', help='path to the dataset')
    parser.add_argument("--dataset", type=str, default='LISAData',
                        help='Target dataset. Currently support: \{SDog120Data, CUB200Data, Stanford40Data, MIT67Data, Flower102Data\}')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--network", type=str, default='resnet18',
                        help='Network architecture. Currently support: \{resnet18, resnet50, resnet101, mbnetv2\}')
    parser.add_argument("--dropout", type=float, default=0, help='Dropout rate for spatial dropout')
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--shot", type=int, default=-1)
    parser.add_argument("--fixed_pic", default=False, action="store_true")
    parser.add_argument("--four_corner", default=False, action="store_true")
    parser.add_argument("--is_poison", default=False, action="store_true")

    args = parser.parse_args()

    args.pid = os.getpid()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    return args


def target_test(model, loader):
    with torch.no_grad():
        model.eval()

        total = 0
        top1 = 0
        for i, (raw_batch, batch, raw_label, target_label) in enumerate(loader):
            raw_batch = raw_batch.to('cuda')
            batch = batch.to('cuda')
            raw_label = raw_label.to('cuda')
            target_label = target_label.to('cuda')

            out = model(raw_batch)
            _, raw_pred = out.max(dim=1)
            out = model(batch)
            _, pred = out.max(dim=1)

            raw_correct = raw_pred.eq(raw_label)
            total += int(raw_correct.sum().item())
            valid_target_correct = pred.eq(target_label) * raw_correct
            top1 += int(valid_target_correct.sum().item())
    return float(top1) / total * 100


def clean_test(model, loader):
    with torch.no_grad():
        model.eval()

        total = 0
        top1 = 0
        for i, (raw_batch, batch, raw_label, target_label) in enumerate(loader):
            raw_batch = raw_batch.to('cuda')
            batch = batch.to('cuda')
            raw_label = raw_label.to('cuda')
            target_label = target_label.to('cuda')

            total += batch.size(0)
            out = model(raw_batch)
            _, raw_pred = out.max(dim=1)

            raw_correct = raw_pred.eq(raw_label)
            top1 += int(raw_correct.sum().item())
    return float(top1) / total * 100


def untarget_test(model, loader):
    with torch.no_grad():
        model.eval()

        total = 0
        top1 = 0
        for i, (raw_batch, batch, raw_label, target_label) in enumerate(loader):
            raw_batch = raw_batch.to('cuda')
            batch = batch.to('cuda')
            raw_label = raw_label.to('cuda')
            target_label = target_label.to('cuda')

            out = model(raw_batch)
            _, raw_pred = out.max(dim=1)
            out = model(batch)
            _, pred = out.max(dim=1)

            raw_correct = raw_pred.eq(raw_label)
            total += int(raw_correct.sum().item())
            valid_untarget_correct = (pred != raw_label) * raw_correct
            top1 += int(valid_untarget_correct.sum().item())
    return float(top1) / total * 100


def testing(model, test_loader):
    test_path = osp.join(args.output_dir, "test2.tsv")

    test_top = untarget_test(model, test_loader)
    with open(test_path, 'a') as af:
        af.write('Test untarget\n')
        columns = ['time', 'Acc']
        af.write('\t'.join(columns) + '\n')
        localtime = time.asctime(time.localtime(time.time()))[4:-6]
        test_cols = [
            localtime,
            round(test_top, 2),
        ]
        af.write('\t'.join([str(c) for c in test_cols]) + '\n')

    test_top = target_test(model, test_loader)
    with open(test_path, 'a') as af:
        af.write('Test target\n')
        columns = ['time', 'Acc']
        af.write('\t'.join(columns) + '\n')
        localtime = time.asctime(time.localtime(time.time()))[4:-6]
        test_cols = [
            localtime,
            round(test_top, 2),
        ]
        af.write('\t'.join([str(c) for c in test_cols]) + '\n')

    test_top = clean_test(model, test_loader)
    with open(test_path, 'a') as af:
        af.write('Test clean\n')
        columns = ['time', 'Acc']
        af.write('\t'.join(columns) + '\n')
        localtime = time.asctime(time.localtime(time.time()))[4:-6]
        test_cols = [
            localtime,
            round(test_top, 2),
        ]
        af.write('\t'.join([str(c) for c in test_cols]) + '\n')


def generate_dataloader(args, normalize, seed):

    train_set = eval(args.dataset)(
        args.datapath, True, [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ],
        args.shot, seed, preload=False, portion=0, fixed_pic=args.fixed_pic, is_poison=args.is_poison  # !use raw data to finetune 
    )

    test_set = eval(args.dataset)(
        args.datapath, False, [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ],
        args.shot, seed, preload=False, portion=1, fixed_pic=args.fixed_pic, four_corner=args.four_corner, is_poison=args.is_poison
    )
    test_set_1 = eval(args.dataset)(
        args.datapath, False, [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ],
        args.shot, seed, preload=False, portion=1, return_raw=True, fixed_pic=args.fixed_pic,
        four_corner=args.four_corner, is_poison=args.is_poison
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size, shuffle=True,
        num_workers=8, pin_memory=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size, shuffle=False,
        num_workers=8, pin_memory=False
    )
    test_loader_1 = torch.utils.data.DataLoader(
        test_set_1,
        batch_size=args.batch_size, shuffle=False,
        num_workers=8, pin_memory=False
    )
    return train_loader, test_loader, test_loader_1


if __name__ == '__main__':
    seed = 259
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    args = get_args()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader, test_loader, test_loader_1 = generate_dataloader(args, normalize, seed)

    dataset = eval(args.dataset)(args.datapath)
    model = eval('{}_dropout'.format(args.network))(
        pretrained=True,
        dropout=args.dropout,
        num_classes=dataset.num_classes
    ).cuda()
    
    ckpt_path = os.path.join(args.output_dir, "ckpt.pth")
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    
    # testing(model, test_loader_1)
    
    with torch.no_grad():
        # untargeted_dir = untarget_test(model, test_loader_1)
        acc = clean_test(model, test_loader_1)
        print(acc)
    
