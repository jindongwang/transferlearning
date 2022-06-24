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


sys.path.append('..')
from model.fe_resnet import resnet18_dropout, resnet50_dropout, resnet101_dropout
from model.fe_resnet import feresnet18, feresnet50, feresnet101

from utils import *

from weight_pruner import WeightPruner

from attack_finetuner import AttackFinetuner
from prune import weight_prune
from finetuner import Finetuner

from backdoor_dataset.cub200 import CUB200Data
from backdoor_dataset.mit67 import MIT67Data
from backdoor_dataset.stanford_40 import Stanford40Data

def teacher_train(teacher, args):
    seed = 98
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    # Used to make sure we sample the same image for few-shot scenarios
    seed = 98

    train_set = eval(args.teacher_dataset)(
        args.teacher_datapath, True, [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ],
        args.shot, seed, preload=False, portion=args.argportion, fixed_pic=args.fixed_pic, is_poison=args.is_poison
    )

    test_set = eval(args.teacher_dataset)(
        args.teacher_datapath, False, [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ],  # target attack
        args.shot, seed, preload=False, portion=1, fixed_pic=args.fixed_pic, four_corner=args.four_corner,
        is_poison=args.is_poison
    )
    clean_set = eval(args.teacher_dataset)(
        args.teacher_datapath, False, [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ],
        args.shot, seed, preload=False, portion=0, fixed_pic=args.fixed_pic, is_poison=args.is_poison
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
    clean_loader = torch.utils.data.DataLoader(
        clean_set,
        batch_size=args.batch_size, shuffle=False,
        num_workers=8, pin_memory=False
    )

    # input()
    student = copy.deepcopy(teacher).cuda()
    if True:
        if args.teacher_method == "weight":
            finetune_machine = WeightPruner(
                args,
                student, teacher,
                train_loader, test_loader,
            )
        elif args.teacher_method == "backdoor_finetune":
            student = weight_prune(
                student, args.backdoor_update_ratio,
            )
            finetune_machine = AttackFinetuner(
                args,
                student, teacher,
                train_loader, test_loader,
            )
        else:
            finetune_machine = Finetuner(
                args,
                student, teacher,
                train_loader, test_loader,
                "ONE"
            )

    finetune_machine.train()

    # start testing (more testing, more cases)
    finetune_machine.test_loader = test_loader

    test_top1, test_ce_loss = finetune_machine.test()
    test_path = osp.join(args.output_dir, "test.tsv")

    with open(test_path, 'a') as af:
        af.write('Teacher! Start testing:    trigger dataset(target attack):\n')
        columns = ['time', 'Acc', 'celoss', 'featloss', 'l2sp']
        af.write('\t'.join(columns) + '\n')
        localtime = time.asctime(time.localtime(time.time()))[4:-6]
        test_cols = [
            localtime,
            round(test_top1, 2),
            round(test_ce_loss, 2),
        ]
        af.write('\t'.join([str(c) for c in test_cols]) + '\n')

    finetune_machine.test_loader = clean_loader
    test_top2, clean_test_ce_loss = finetune_machine.test()
    test_path = osp.join(args.output_dir, "test.tsv")

    with open(test_path, 'a') as af:
        af.write('Teacher! Start testing:    clean dataset:\n')
        columns = ['time', 'Acc', 'celoss', 'featloss', 'l2sp']
        af.write('\t'.join(columns) + '\n')
        localtime = time.asctime(time.localtime(time.time()))[4:-6]
        test_cols = [
            localtime,
            round(test_top2, 2),
            round(clean_test_ce_loss, 2),
        ]
        af.write('\t'.join([str(c) for c in test_cols]) + '\n')
    return student
