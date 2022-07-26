# coding=utf-8
import torch


def get_params(alg, args, inner=False, alias=True, isteacher=False):
    if args.schuse:
        if args.schusech == 'cos':
            initlr = args.lr
        else:
            initlr = 1.0
    else:
        if inner:
            initlr = args.inner_lr
        else:
            initlr = args.lr
    if isteacher:
        params = [
            {'params': alg[0].parameters(), 'lr': args.lr_decay1 * initlr},
            {'params': alg[1].parameters(), 'lr': args.lr_decay2 * initlr},
            {'params': alg[2].parameters(), 'lr': args.lr_decay2 * initlr}
        ]
        return params
    if inner:
        params = [
            {'params': alg[0].parameters(), 'lr': args.lr_decay1 *
             initlr},
            {'params': alg[1].parameters(), 'lr': args.lr_decay2 *
             initlr}
        ]
    elif alias:
        params = [
            {'params': alg.featurizer.parameters(), 'lr': args.lr_decay1 * initlr},
            {'params': alg.classifier.parameters(), 'lr': args.lr_decay2 * initlr}
        ]
    else:
        params = [
            {'params': alg[0].parameters(), 'lr': args.lr_decay1 * initlr},
            {'params': alg[1].parameters(), 'lr': args.lr_decay2 * initlr}
        ]
    if ('DANN' in args.algorithm) or ('CDANN' in args.algorithm):
        params.append({'params': alg.discriminator.parameters(),
                       'lr': args.lr_decay2 * initlr})
    if ('CDANN' in args.algorithm):
        params.append({'params': alg.class_embeddings.parameters(),
                       'lr': args.lr_decay2 * initlr})
    return params


def get_optimizer(alg, args, inner=False, alias=True, isteacher=False):
    params = get_params(alg, args, inner, alias, isteacher)
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    return optimizer


def get_scheduler(optimizer, args):
    if not args.schuse:
        return None
    if args.schusech == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.max_epoch * args.steps_per_epoch)
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda x:  args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    return scheduler
