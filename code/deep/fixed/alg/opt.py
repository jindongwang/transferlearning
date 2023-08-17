# coding=utf-8
import torch


def get_params(alg, args):
    if args.schuse:
        if args.schusech == 'cos':
            init_lr = args.lr
        else:
            init_lr = 1.0
    else:
        init_lr = args.lr
    params = [
        {'params': alg.featurizer.parameters(), 'lr': args.lr_decay1 * init_lr},
        {'params': alg.bottleneck.parameters(), 'lr': args.lr_decay2 * init_lr},
        {'params': alg.classifier.parameters(), 'lr': args.lr_decay2 * init_lr}
    ]
    params.append({'params': alg.discriminator.parameters(),
                    'lr': args.lr_decay2 * init_lr})
    return params


def get_optimizer(alg, args):
    params = get_params(alg, args)
    optimizer = torch.optim.Adam(
        params, lr=args.lr, weight_decay=args.weight_decay, betas=(args.beta1, 0.9))
    return optimizer


def get_scheduler(optimizer, args):
    return None
