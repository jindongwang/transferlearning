# coding=utf-8
import torch
import copy
import torch.nn.functional as F

from alg.opt import *
import torch.autograd as autograd
from datautil.util import random_pairs_of_minibatches_by_domainperm
from alg.algs.ERM import ERM


class MLDG(ERM):
    def __init__(self, args):
        super(MLDG, self).__init__(args)
        self.args = args

    def update(self, minibatches, opt, sch):
        """
        For computational efficiency, we do not compute second derivatives.
        """
        num_mb = len(minibatches)
        objective = 0

        opt.zero_grad()
        for p in self.network.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)

        for (xi, yi), (xj, yj) in random_pairs_of_minibatches_by_domainperm(minibatches):

            xi, yi, xj, yj = xi.cuda().float(), yi.cuda(
            ).long(), xj.cuda().float(), yj.cuda().long()
            inner_net = copy.deepcopy(self.network)

            inner_opt = get_optimizer(inner_net, self.args, True)
            inner_sch = get_scheduler(inner_opt, self.args)

            inner_obj = F.cross_entropy(inner_net(xi), yi)

            inner_opt.zero_grad()
            inner_obj.backward()
            inner_opt.step()
            if inner_sch:
                inner_sch.step()

            for p_tgt, p_src in zip(self.network.parameters(),
                                    inner_net.parameters()):
                if p_src.grad is not None:
                    p_tgt.grad.data.add_(p_src.grad.data / num_mb)

            objective += inner_obj.item()

            loss_inner_j = F.cross_entropy(inner_net(xj), yj)
            grad_inner_j = autograd.grad(loss_inner_j, inner_net.parameters(),
                                         allow_unused=True)

            objective += (self.args.mldg_beta * loss_inner_j).item()

            for p, g_j in zip(self.network.parameters(), grad_inner_j):
                if g_j is not None:
                    p.grad.data.add_(
                        self.args.mldg_beta * g_j.data / num_mb)

        objective /= len(minibatches)

        opt.step()
        if sch:
            sch.step()
        return {'total': objective}
