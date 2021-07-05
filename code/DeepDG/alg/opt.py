# coding=utf-8
import torch


def get_params(alg, args, inner=False):
    if inner:
        params = [
            {'params': alg[0].parameters(), 'lr': args.lr_decay1 *
             args.inner_lr},
            {'params': alg[1].parameters(), 'lr': args.lr_decay2 *
             args.inner_lr},
            {'params': alg[2].parameters(), 'lr': args.lr_decay2 *
             args.inner_lr}
        ]
    else:
        params = [
            {'params': alg.featurizer.parameters(), 'lr': args.lr_decay1 * args.lr},
            {'params': alg.bottleneck.parameters(), 'lr': args.lr_decay2 * args.lr},
            {'params': alg.classifier.parameters(), 'lr': args.lr_decay2 * args.lr}
        ]
    if ('DANN' in args.algorithm) or ('CDANN' in args.algorithm):
        params.append({'params': alg.discriminator.parameters(),
                      'lr': args.lr_decay2 * args.lr})
    if ('CDANN' in args.algorithm):
        params.append({'params': alg.class_embeddings.parameters(),
                      'lr': args.lr_decay2 * args.lr})
    return params


def get_optimizer(alg, args, inner=False):
    params = get_params(alg, args, inner)
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=False)
    return optimizer


def get_scheduler(optimizer, args):
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda x:  args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    return scheduler
