import os
import os.path as osp
import sys
import time
import argparse
from pdb import set_trace as st
import json
import random
from functools import partial

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

from model.fe_resnet import resnet18_dropout, resnet50_dropout, resnet101_dropout, resnet34_dropout
from model.fe_resnet import feresnet18, feresnet50, feresnet101, feresnet34


from eval_robustness import advtest, myloss
from utils import *

def linear_l2(model):
    beta_loss = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            beta_loss += (m.weight).pow(2).sum()
            beta_loss += (m.bias).pow(2).sum()
    return 0.5*beta_loss*args.beta, beta_loss

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


def test(model, teacher, loader, loss=False):
    with torch.no_grad():
        model.eval()

        if loss:
            teacher.eval()

            ce = CrossEntropyLabelSmooth(loader.dataset.num_classes, args.label_smoothing)
            featloss = torch.nn.MSELoss(reduction='none')

        total_ce = 0
        total_feat_reg = np.zeros(len(reg_layers))
        total_l2sp_reg = 0
        total = 0
        top1 = 0

        total = 0
        top1 = 0
        for i, (batch, label) in enumerate(loader):

            total += batch.size(0)
            out = model(batch)
            _, pred = out.max(dim=1)
            top1 += int(pred.eq(label).sum().item())

            if loss:
                total_ce += ce(out, label).item()
                if teacher is not None:
                    with torch.no_grad():
                        tout = teacher(batch)

                    # for key in reg_layers:
                    for i, key in enumerate(reg_layers):
                        src_x = reg_layers[key][0].out
                        tgt_x = reg_layers[key][1].out

                        regloss = featloss(src_x, tgt_x.detach()).mean()

                        total_feat_reg[i] += regloss.item()

                _, unweighted = l2sp(model, 0)
                total_l2sp_reg += unweighted.item()

    return float(top1)/total*100, total_ce/(i+1), np.sum(total_feat_reg)/(i+1), total_l2sp_reg/(i+1), total_feat_reg/(i+1)

def train(
    model, 
    train_loader, 
    val_loader, 
    adv_eval_fn,
    iterations=9000, 
    lr=1e-2, 
    output_dir='results', 
    l2sp_lmda=1e-2, 
    teacher=None, 
    reg_layers={}
):

    if l2sp_lmda == 0:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=0)

    end_iter = iterations
    if args.swa:
        optimizer = torchcontrib.optim.SWA(optimizer, swa_start=args.swa_start, swa_freq=args.swa_freq)
        end_iter = args.swa_start

    teacher.eval()
    ce = CrossEntropyLabelSmooth(train_loader.dataset.num_classes, args.label_smoothing)
    featloss = torch.nn.MSELoss()


    batch_time = MovingAverageMeter('Time', ':6.3f')
    data_time = MovingAverageMeter('Data', ':6.3f')
    ce_loss_meter = MovingAverageMeter('CE Loss', ':6.3f')
    feat_loss_meter  = MovingAverageMeter('Feat. Loss', ':6.3f')
    l2sp_loss_meter  = MovingAverageMeter('L2SP Loss', ':6.3f')
    linear_loss_meter  = MovingAverageMeter('LinearL2 Loss', ':6.3f')
    total_loss_meter  = MovingAverageMeter('Total Loss', ':6.3f')
    top1_meter  = MovingAverageMeter('Acc@1', ':6.2f')

    train_path = osp.join(output_dir, "train.tsv")
    with open(train_path, 'w') as wf:
        columns = ['time', 'iter', 'Acc', 'celoss', 'featloss', 'l2sp']
        wf.write('\t'.join(columns) + '\n')
    test_path = osp.join(output_dir, "test.tsv")
    with open(test_path, 'w') as wf:
        columns = ['time', 'iter', 'Acc', 'celoss', 'featloss', 'l2sp']
        wf.write('\t'.join(columns) + '\n')
    adv_path = osp.join(output_dir, "adv.tsv")
    with open(adv_path, 'w') as wf:
        columns = ['time', 'iter', 'Acc', 'AdvAcc', 'ASR']
        wf.write('\t'.join(columns) + '\n')
    
    dataloader_iterator = iter(train_loader)
    for i in range(iterations):
        model.train()
        optimizer.zero_grad()

        end = time.time()
        try:
            batch, label = next(dataloader_iterator)
        except:
            dataloader_iterator = iter(train_loader)
            batch, label = next(dataloader_iterator)
        data_time.update(time.time() - end)

        out = model(batch)
        _, pred = out.max(dim=1)

        top1_meter.update(float(pred.eq(label).sum().item()) / label.shape[0] * 100.)

        loss = 0.
        loss += ce(out, label)

        ce_loss_meter.update(loss.item())

        with torch.no_grad():
            tout = teacher(batch)

        # Compute the feature distillation loss only when needed
        if args.feat_lmda != 0:
            regloss = 0
            # for layer in args.feat_layers:
            for key in reg_layers:
                # key = int(layer)-1

                src_x = reg_layers[key][0].out
                tgt_x = reg_layers[key][1].out
                regloss += featloss(src_x, tgt_x.detach())

            regloss = args.feat_lmda * regloss
            loss += regloss
            feat_loss_meter.update(regloss.item())

        beta_loss, linear_norm = linear_l2(model)
        loss = loss + beta_loss 
        linear_loss_meter.update(beta_loss.item())

        if l2sp_lmda != 0:
            reg, _ = l2sp(model, l2sp_lmda)
            l2sp_loss_meter.update(reg.item())
            loss = loss + reg

        total_loss_meter.update(loss.item())

        loss.backward()
        optimizer.step()
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']

        batch_time.update(time.time() - end)

        if (i % args.print_freq == 0) or (i == iterations-1):
            progress = ProgressMeter(
                iterations,
                [batch_time, data_time, top1_meter, total_loss_meter, ce_loss_meter, feat_loss_meter, l2sp_loss_meter, linear_loss_meter],
                prefix="LR: {:6.3f}".format(current_lr),
                output_dir=output_dir,
            )
            progress.display(i)

        if ((i+1) % args.test_interval == 0) or (i == iterations-1):
            test_top1, test_ce_loss, test_feat_loss, test_weight_loss, test_feat_layer_loss = test(
                model, teacher, val_loader, loss=True
            )
            train_top1, train_ce_loss, train_feat_loss, train_weight_loss, train_feat_layer_loss = test(
                model, teacher, train_loader, loss=True
            )
            
            print('Eval Train | Iteration {}/{} | Top-1: {:.2f} | CE Loss: {:.3f} | Feat Reg Loss: {:.6f} | L2SP Reg Loss: {:.3f}'.format(i+1, iterations, train_top1, train_ce_loss, train_feat_loss, train_weight_loss))
            print('Eval Test | Iteration {}/{} | Top-1: {:.2f} | CE Loss: {:.3f} | Feat Reg Loss: {:.6f} | L2SP Reg Loss: {:.3f}'.format(i+1, iterations, test_top1, test_ce_loss, test_feat_loss, test_weight_loss))
            localtime = time.asctime( time.localtime(time.time()) )[4:-6]
            with open(train_path, 'a') as af:
                train_cols = [
                    localtime,
                    i, 
                    round(train_top1,2), 
                    round(train_ce_loss,2), 
                    round(train_feat_loss,2),
                    round(train_weight_loss,2),
                ]
                af.write('\t'.join([str(c) for c in train_cols]) + '\n')
            with open(test_path, 'a') as af:
                test_cols = [
                    localtime,
                    i, 
                    round(test_top1,2), 
                    round(test_ce_loss,2), 
                    round(test_feat_loss,2),
                    round(test_weight_loss,2),
                ]
                af.write('\t'.join([str(c) for c in test_cols]) + '\n')
            
            if not args.no_save:
                # if not os.path.exists('ckpt'):
                #     os.makedirs('ckpt')
                # torch.save({'state_dict': model.state_dict()}, 'ckpt/{}.pth'.format(name))
                ckpt_path = osp.join(
                    args.output_dir,
                    "ckpt.pth"
                )
                torch.save({'state_dict': model.state_dict()}, ckpt_path)

        if args.adv_test_interval > 0 and ( ((i+1) % args.adv_test_interval == 0) ):
            clean_top1, adv_top1, adv_sr = adv_eval_fn(model)
            localtime = time.asctime( time.localtime(time.time()) )[4:-6]
            with open(adv_path, 'a') as af:
                test_cols = [
                    localtime,
                    i, 
                    round(clean_top1,2),
                    round(adv_top1,2),
                    round(adv_sr,2),
                ]
                af.write('\t'.join([str(c) for c in test_cols]) + '\n')

    if args.swa:
        optimizer.swap_swa_sgd()

        for m in model.modules():
            if hasattr(m, 'running_mean'):
                m.reset_running_stats()
                m.momentum = None
        with torch.no_grad():
            model.train()
            for x, y in train_loader:
                out = model(x)

        test_top1, test_ce_loss, test_feat_loss, test_weight_loss, test_feat_layer_loss = test(
            model, teacher, val_loader, loss=True
        )
        train_top1, train_ce_loss, train_feat_loss, train_weight_loss, train_feat_layer_loss = test(
            model, teacher, train_loader, loss=True
        )
        # clean_top1, adv_top1, adv_sr = adv_eval_fn(model)
        print('Eval Train | Iteration {}/{} | Top-1: {:.2f} | CE Loss: {:.3f} | Feat Reg Loss: {:.6f} | L2SP Reg Loss: {:.3f}'.format(i+1, iterations, train_top1, train_ce_loss, train_feat_loss, train_weight_loss))
        print('Eval Test | Iteration {}/{} | Top-1: {:.2f} | CE Loss: {:.3f} | Feat Reg Loss: {:.6f} | L2SP Reg Loss: {:.3f}'.format(i+1, iterations, test_top1, test_ce_loss, test_feat_loss, test_weight_loss))
        localtime = time.asctime( time.localtime(time.time()) )[4:-6]
        with open(train_path, 'a') as af:
            train_cols = [
                localtime,
                i, 
                round(train_top1,2), 
                round(train_ce_loss,2), 
                round(train_feat_loss,2),
                round(train_weight_loss,2),
            ]
            af.write('\t'.join([str(c) for c in train_cols]) + '\n')
        with open(test_path, 'a') as af:
            test_cols = [
                localtime,
                i, 
                round(test_top1,2), 
                # round(adv_top1,2),
                # round(adv_sr,2),
                round(test_ce_loss,2), 
                round(test_feat_loss,2),
                round(test_weight_loss,2),
            ]
            af.write('\t'.join([str(c) for c in test_cols]) + '\n')
        # clean_top1, adv_top1, adv_sr = adv_eval_fn(model)
        # localtime = time.asctime( time.localtime(time.time()) )[4:-6]
        # with open(adv_path, 'a') as af:
        #     test_cols = [
        #         localtime,
        #         i, 
        #         round(clean_top1,2),
        #         round(adv_top1,2),
        #         round(adv_sr,2),
        #     ]
        #     af.write('\t'.join([str(c) for c in test_cols]) + '\n')
        if not args.no_save:
            # if not os.path.exists('ckpt'):
            #     os.makedirs('ckpt')
            ckpt_path = osp.join(
                args.output_dir,
                "ckpt.pth"
            )
            torch.save({'state_dict': model.state_dict()}, ckpt_path)

    return model

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, default='/data', help='path to the dataset')
    parser.add_argument("--dataset", type=str, default='CUB200Data', help='Target dataset. Currently support: \{SDog120Data, CUB200Data, Stanford40Data, MIT67Data, Flower102Data\}')
    parser.add_argument("--iterations", type=int, default=30000, help='Iterations to train')
    parser.add_argument("--print_freq", type=int, default=100, help='Frequency of printing training logs')
    parser.add_argument("--test_interval", type=int, default=1000, help='Frequency of testing')
    parser.add_argument("--adv_test_interval", type=int, default=1000)
    parser.add_argument("--name", type=str, default='test', help='Name for the checkpoint')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=1e-2, help='The strength of the L2 regularization on the last linear layer')
    parser.add_argument("--dropout", type=float, default=0, help='Dropout rate for spatial dropout')
    parser.add_argument("--l2sp_lmda", type=float, default=0)
    parser.add_argument("--feat_lmda", type=float, default=0)
    parser.add_argument("--feat_layers", type=str, default='1234', help='Used for DELTA (which layers or stages to match), ResNets should be 1234 and MobileNetV2 should be 12345')
    parser.add_argument("--reinit", action='store_true', default=False, help='Reinitialize before training')
    parser.add_argument("--no_save", action='store_true', default=False, help='Do not save checkpoints')
    parser.add_argument("--swa", action='store_true', default=False, help='Use SWA')
    parser.add_argument("--swa_freq", type=int, default=500, help='Frequency of averaging models in SWA')
    parser.add_argument("--swa_start", type=int, default=0, help='Start SWA since which iterations')
    parser.add_argument("--label_smoothing", type=float, default=0)
    parser.add_argument("--checkpoint", type=str, default='', help='Load a previously trained checkpoint')
    parser.add_argument("--network", type=str, default='resnet18', help='Network architecture. Currently support: \{resnet18, resnet50, resnet101, mbnetv2\}')
    parser.add_argument("--shot", type=int, default=-1, help='Number of training samples per class for the training dataset. -1 indicates using the full dataset.')
    parser.add_argument("--log", action='store_true', default=False, help='Redirect the output to log/args.name.log')
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--B", type=float, default=0.1, help='Attack budget')
    parser.add_argument("--m", type=float, default=1000, help='Hyper-parameter for task-agnostic attack')
    parser.add_argument("--pgd_iter", type=int, default=40)
    parser.add_argument("--adv_data_dir", default="results/advdata")
    parser.add_argument("--seed", type=int, default=98)
    args = parser.parse_args()
    args.adv_data_dir = osp.join(
        args.adv_data_dir, f"{args.dataset}_{args.network}.pt"
    )
    return args

# Used to matching features
def record_act(self, input, output):
    self.out = output

if __name__ == '__main__':
    

    args = get_args()

    seed = args.seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if args.log:
        if not os.path.exists('log'):
            os.makedirs('log')
        sys.stdout = open('log/{}.log'.format(args.name), 'w')
    args.output_dir = osp.join(
        args.output_dir,
        args.name
    )
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    params_out_path = osp.join(args.output_dir, 'params.json')
    with open(params_out_path, 'w') as jf:
        json.dump(vars(args), jf, indent=True)
    print(args)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    # Used to make sure we sample the same image for few-shot scenarios
    seed = 98

    train_set = eval(args.dataset)(
        args.datapath, True, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]), 
        args.shot, seed, preload=False
    )

    test_set = eval(args.dataset)(
        args.datapath, False, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]), 
        args.shot, seed, preload=False
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size, shuffle=True,
        num_workers=8, pin_memory=False
    )
    
    val_loader = train_loader

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size, shuffle=False,
        num_workers=8, pin_memory=False
    )
    
    model = eval('{}_dropout'.format(args.network))(
        pretrained=True, 
        dropout=args.dropout, 
        num_classes=train_loader.dataset.num_classes
    )
    
    if args.checkpoint != '':
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['state_dict'])

    # Pre-trained model
    teacher = eval('{}_dropout'.format(args.network))(
        pretrained=True, 
        dropout=0, 
        num_classes=train_loader.dataset.num_classes
    )
    
    if 'mbnetv2' in args.network:
        reg_layers = {0: [model.layer1], 1: [model.layer2], 2: [model.layer3], 3: [model.layer4], 4: [model.layer5]}
        model.layer1.register_forward_hook(record_act)
        model.layer2.register_forward_hook(record_act)
        model.layer3.register_forward_hook(record_act)
        model.layer4.register_forward_hook(record_act)
        model.layer5.register_forward_hook(record_act)
    elif 'resnet' in args.network:
        reg_layers = {0: [model.layer1], 1: [model.layer2], 2: [model.layer3], 3: [model.layer4]}
        model.layer1.register_forward_hook(record_act)
        model.layer2.register_forward_hook(record_act)
        model.layer3.register_forward_hook(record_act)
        model.layer4.register_forward_hook(record_act)
    elif 'vgg' in args.network:
        cnt = 0
        reg_layers = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.MaxPool2d) and '28' not in name:
                reg_layers[name] = [module]
                module.register_forward_hook(record_act)
                print(name, module)
        


    # Stored pre-trained weights for computing L2SP
    for m in model.modules():
        if hasattr(m, 'weight') and not hasattr(m, 'old_weight'):
            m.old_weight = m.weight.data.clone().detach()
            # all_weights = torch.cat([all_weights.reshape(-1), m.weight.data.abs().reshape(-1)], dim=0)
        if hasattr(m, 'bias') and not hasattr(m, 'old_bias') and m.bias is not None:
            m.old_bias = m.bias.data.clone().detach()

    if args.reinit:
        for m in model.modules():
            if type(m) in [nn.Linear, nn.BatchNorm2d, nn.Conv2d]:
                m.reset_parameters()

    if 'vgg' not in args.network:
        reg_layers[0].append(teacher.layer1)
        teacher.layer1.register_forward_hook(record_act)
        reg_layers[1].append(teacher.layer2)
        teacher.layer2.register_forward_hook(record_act)
        reg_layers[2].append(teacher.layer3)
        teacher.layer3.register_forward_hook(record_act)
        reg_layers[3].append(teacher.layer4)
        teacher.layer4.register_forward_hook(record_act)

        if '5' in args.feat_layers:
            reg_layers[4].append(teacher.layer5)
            teacher.layer5.register_forward_hook(record_act)
    else:
        cnt = 0
        for name, module in teacher.named_modules():
            if isinstance(module, nn.MaxPool2d) and '28' not in name:
                reg_layers[name].append(module)
                module.register_forward_hook(record_act)
                # print(name, module)

    eval_pretrained_model = eval('fe{}'.format(args.network))(pretrained=True).eval()
    adversary = LinfPGDAttack(
            eval_pretrained_model, loss_fn=myloss, eps=args.B,
            nb_iter=args.pgd_iter, eps_iter=0.01, 
            rand_init=True, clip_min=-2.2, clip_max=2.2,
            targeted=False)
    adveval_test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=8, shuffle=False,
        num_workers=8, pin_memory=False
    )
    adv_eval_fn = partial(
        advtest,
        loader=adveval_test_loader,
        adversary=adversary,
        args=args,
    )

    train(
        model, 
        train_loader, 
        test_loader, 
        adv_eval_fn,
        l2sp_lmda=args.l2sp_lmda, 
        iterations=args.iterations, 
        lr=args.lr, 
        output_dir=args.output_dir, 
        teacher=teacher, 
        reg_layers=reg_layers,
    )

    # Evaluate
    pretrained_model = eval('fe{}'.format(args.network))(pretrained=True).eval()
    adversary = LinfPGDAttack(
            pretrained_model, loss_fn=myloss, eps=args.B,
            nb_iter=args.pgd_iter, eps_iter=0.01, 
            rand_init=True, clip_min=-2.2, clip_max=2.2,
            targeted=False)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=8, shuffle=False,
        num_workers=8, pin_memory=False
    )
    clean_top1, adv_top1, adv_sr = advtest(model, test_loader, adversary, args)
    result_sum = 'Clean Top-1: {:.2f} | Adv Top-1: {:.2f} | Attack Success Rate: {:.2f}'.format(clean_top1, adv_top1, adv_sr)
    with open(osp.join(args.output_dir, "posttrain_eval.txt"), "w") as f:
        f.write(result_sum)
