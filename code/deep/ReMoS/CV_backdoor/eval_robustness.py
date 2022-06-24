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

from PIL import Image
from pdb import set_trace as st

from torchvision import transforms

from advertorch.attacks import LinfPGDAttack
from model.fe_resnet import resnet18_dropout, resnet34_dropout, resnet50_dropout, resnet101_dropout
from model.fe_resnet import feresnet18, feresnet34, feresnet50, feresnet101

def advtest(model, loader, adversary, args):
    model.eval()

    total_ce = 0
    total = 0
    top1 = 0

    total = 0
    top1_clean = 0
    top1_adv = 0
    adv_success = 0
    adv_trial = 0
    for i, (batch, label) in enumerate(loader):
        batch, label = batch.to('cuda'), label.to('cuda')

        total += batch.size(0)
        out_clean = model(batch)

        if 'mbnetv2' in args.network:
            y = torch.zeros(batch.shape[0], model.classifier[1].in_features).cuda()
        elif 'resnet' in args.network:
            y = torch.zeros(batch.shape[0], model.fc.in_features).cuda()

        y[:,0] = args.m
        advbatch = adversary.perturb(batch, y)

        out_adv = model(advbatch)

        _, pred_clean = out_clean.max(dim=1)
        _, pred_adv = out_adv.max(dim=1)

        clean_correct = pred_clean.eq(label)
        adv_trial += int(clean_correct.sum().item())
        adv_success += int(pred_adv[clean_correct].eq(label[clean_correct]).sum().detach().item())
        top1_clean += int(pred_clean.eq(label).sum().detach().item())
        top1_adv += int(pred_adv.eq(label).sum().detach().item())

        print('{}/{}...'.format(i+1, len(loader)))
        if i > 5:
            break

    if adv_trial==0:
        adv_trial = 1
    return float(top1_clean)/total*100, float(top1_adv)/total*100, float(adv_trial-adv_success) / adv_trial *100

def record_act(self, input, output):
    pass

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, default='/data', help='path to the dataset')
    parser.add_argument("--dataset", type=str, default='CUB200Data', help='Target dataset. Currently support: \{SDog120Data, CUB200Data, Stanford40Data, MIT67Data, Flower102Data\}')
    parser.add_argument("--name", type=str, default='test')
    parser.add_argument("--B", type=float, default=0.1, help='Attack budget')
    parser.add_argument("--m", type=float, default=1000, help='Hyper-parameter for task-agnostic attack')
    parser.add_argument("--pgd_iter", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--checkpoint", type=str, default='')
    parser.add_argument("--network", type=str, default='resnet18', help='Network architecture. Currently support: \{resnet18, resnet50, resnet101, mbnetv2\}')
    parser.add_argument("--teacher", default=None)
    args = parser.parse_args()
    if args.teacher is None:
        args.teacher = args.network
    return args

def myloss(yhat, y):
    return -((yhat[:,0]-y[:,0])**2 + 0.1*((yhat[:,1:]-y[:,1:])**2).mean(1)).mean()

if __name__ == '__main__':
    args = get_args()
    print(args)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    seed = int(time.time())

    test_set = eval(args.dataset)(
        args.datapath, False, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]), 
        -1, seed, preload=False
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size, shuffle=False,
        num_workers=8, pin_memory=False)
    
    transferred_model = eval('{}_dropout'.format(args.network))(pretrained=False, dropout=args.dropout, num_classes=test_loader.dataset.num_classes).cuda()
    checkpoint = torch.load(args.checkpoint)
    transferred_model.load_state_dict(checkpoint['state_dict'])

    pretrained_model = eval('fe{}'.format(args.teacher))(pretrained=True).cuda().eval()

    adversary = LinfPGDAttack(
            pretrained_model, loss_fn=myloss, eps=args.B,
            nb_iter=args.pgd_iter, eps_iter=0.01, 
            rand_init=True, clip_min=-2.2, clip_max=2.2,
            targeted=False)

    clean_top1, adv_top1, adv_sr = advtest(transferred_model, test_loader, adversary, args)

    print('Clean Top-1: {:.2f} | Adv Top-1: {:.2f} | Attack Success Rate: {:.2f}'.format(clean_top1, adv_top1, adv_sr))
