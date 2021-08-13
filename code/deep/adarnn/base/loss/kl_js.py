import torch.nn as nn

def kl_div(source, target):
    if len(source) < len(target):
        target = target[:len(source)]
    elif len(source) > len(target):
        source = source[:len(target)]
    criterion = nn.KLDivLoss(reduction='batchmean')
    loss = criterion(source.log(), target)
    return loss


def js(source, target):
    if len(source) < len(target):
        target = target[:len(source)]
    elif len(source) > len(target):
        source = source[:len(target)]
    M = .5 * (source + target)
    loss_1, loss_2 = kl_div(source, M), kl_div(target, M)
    return .5 * (loss_1 + loss_2)