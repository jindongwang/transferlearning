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


# sys.path.append('../CV_adv')

from model.fe_resnet import resnet18_dropout, resnet50_dropout, resnet101_dropout
from model.fe_resnet import feresnet18, feresnet50, feresnet101

from utils import *

from weight_pruner import WeightPruner
from finetuner import Finetuner as RawFinetuner

from attack_finetuner import AttackFinetuner
from prune import weight_prune
from finetuner import Finetuner
from remos.remos_pruner import ReMoSPruner


from backdoor_dataset.cub200 import CUB200Data
from backdoor_dataset.mit67 import MIT67Data
from backdoor_dataset.stanford_40 import Stanford40Data
from trigger import teacher_train

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_datapath", type=str, default='../data/LISA', help='path to the dataset')
    parser.add_argument("--teacher_dataset", type=str, default='LISAData',
                        help='Target dataset. Currently support: \{SDog120Data, CUB200Data, Stanford40Data, MIT67Data, Flower102Data\}')
    parser.add_argument("--student_datapath", type=str, default='../data/pubfig83', help='path to the dataset')
    parser.add_argument("--student_dataset", type=str, default='PUBFIGData',
                        help='Target dataset. Currently support: \{SDog120Data, CUB200Data, Stanford40Data, MIT67Data, Flower102Data\}')

    parser.add_argument("--iterations", type=int, default=30000, help='Iterations to train')
    parser.add_argument("--print_freq", type=int, default=100, help='Frequency of printing training logs')
    parser.add_argument("--test_interval", type=int, default=1000, help='Frequency of testing')
    parser.add_argument("--adv_test_interval", type=int, default=1000)
    parser.add_argument("--name", type=str, default='test', help='Name for the checkpoint')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=1e-2,
                        help='The strength of the L2 regularization on the last linear layer')
    parser.add_argument("--dropout", type=float, default=0, help='Dropout rate for spatial dropout')
    parser.add_argument("--no_save", action='store_true', default=False, help='Do not save checkpoints')
    parser.add_argument("--checkpoint", type=str, default='', help='Load a previously trained checkpoint')
    parser.add_argument("--network", type=str, default='resnet18',
                        help='Network architecture. Currently support: \{resnet18, resnet50, resnet101, mbnetv2\}')
    parser.add_argument("--shot", type=int, default=-1,
                        help='Number of training samples per class for the training dataset. -1 indicates using the '
                             'full dataset.')
    parser.add_argument("--log", action='store_true', default=False, help='Redirect the output to log/args.name.log')
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--B", type=float, default=0.1, help='Attack budget')
    parser.add_argument("--m", type=float, default=1000, help='Hyper-parameter for task-agnostic attack')
    parser.add_argument("--pgd_iter", type=int, default=40)
    parser.add_argument("--teacher_method", default=None,)
    parser.add_argument("--student_method", default=None,)
    parser.add_argument("--reinit", action="store_true", default=False)

    parser.add_argument("--prune_interval", default=10000, type=int)
    # Weight prune
    parser.add_argument("--weight_total_ratio", default=0.5, type=float)
    parser.add_argument("--weight_ratio_per_prune", default=0, type=float)
    parser.add_argument("--weight_init_prune_ratio", default=0.5, type=float)
    parser.add_argument("--prune_descending", default=False, action="store_true")

    parser.add_argument("--argportion", default=0.2, type=float)
    parser.add_argument("--student_ckpt", type=str, default='')

    # Finetune for backdoor attack
    parser.add_argument("--backdoor_update_ratio", default=0, type=float,
                        help="From how much ratio does the weight update")
    parser.add_argument("--fixed_pic", default=False, action="store_true")
    parser.add_argument("--four_corner", default=False, action="store_true")
    parser.add_argument("--is_poison", default=False, action="store_true")

    parser.add_argument("--nc_info_dir")

    args = parser.parse_args()

    args.family_output_dir = args.output_dir
    args.output_dir = osp.join(
        args.output_dir,
        args.name
    )
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
    test_path = osp.join(args.output_dir, "test.tsv")

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

    train_set = eval(args.student_dataset)(
        args.student_datapath, True, [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ],
        args.shot, seed, preload=False, portion=0, fixed_pic=args.fixed_pic, is_poison=args.is_poison 
    )

    test_set = eval(args.student_dataset)(
        args.student_datapath, False, [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ],
        args.shot, seed, preload=False, portion=1, fixed_pic=args.fixed_pic, four_corner=args.four_corner, is_poison=args.is_poison
    )
    test_set_1 = eval(args.student_dataset)(
        args.student_datapath, False, [
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
    # Used to make sure we sample the same image for few-shot scenarios

    train_loader, test_loader, test_loader_1 = generate_dataloader(args, normalize, seed)

    teacher_set = eval(args.teacher_dataset)(args.teacher_datapath)
    teacher = eval('{}_dropout'.format(args.network))(
        pretrained=True,
        dropout=args.dropout,
        num_classes=teacher_set.num_classes
    )

    if args.checkpoint == '':
        teacher_train(teacher, args)
        load_path = args.output_dir + '/teacher_ckpt.pth'
        exit()
    else:
        load_path = args.checkpoint

    checkpoint = torch.load(load_path)
    teacher.load_state_dict(checkpoint['state_dict'])
    print(f"Loaded teacher checkpoint from {args.checkpoint}")

    if "resnet18" in args.network:
        teacher.fc = nn.Linear(512, train_loader.dataset.num_classes)
    elif "resnet50" in args.network:
        teacher.fc = nn.Linear(2048, train_loader.dataset.num_classes)
    teacher = teacher

    student = copy.deepcopy(teacher)

    if args.student_ckpt != '':
        checkpoint = torch.load(args.student_ckpt)
        student.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded student checkpoint from {args.student_ckpt}")

    if args.reinit:
        for m in student.modules():
            if type(m) in [nn.Linear, nn.BatchNorm2d, nn.Conv2d]:
                m.reset_parameters()

    if args.student_method == "weight":
        finetune_machine = WeightPruner(
            args,
            student, teacher,
            train_loader, test_loader,
        )
    elif args.student_method == "backdoor_finetune":
        student = weight_prune(
            student, args.backdoor_update_ratio,
        )
        finetune_machine = AttackFinetuner(
            args,
            student, teacher,
            train_loader, test_loader,
        )
    elif args.student_method == "finetune":
        finetune_machine = RawFinetuner(
            args,
            student, teacher,
            train_loader, test_loader,
        )
    elif args.student_method == "nc_weight_rank_prune":
        finetune_machine = ReMoSPruner(
            args, student, teacher, train_loader, test_loader,
        )
    else:
        finetune_machine = Finetuner(
            args,
            student, teacher,
            train_loader, test_loader,
            "TWO"
        )

    if args.student_ckpt == '':
        finetune_machine.train()

    if hasattr(finetune_machine, "final_check_param_num"):
        finetune_machine.final_check_param_num()

    testing(finetune_machine.model, test_loader_1)
