# coding=utf-8
import torch
import torch.nn.functional as F
from alg.algs.ERM import ERM


class VREx(ERM):
    """V-REx algorithm from http://arxiv.org/abs/2003.00688"""

    def __init__(self, args):
        super(VREx, self).__init__(args)
        self.register_buffer('update_count', torch.tensor([0]))
        self.args = args

    def update(self, minibatches, opt, sch):
        if self.update_count >= self.args.anneal_iters:
            penalty_weight = self.args.lam
        else:
            penalty_weight = 1.0

        nll = 0.

        all_x = torch.cat([data[0].cuda().float() for data in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        losses = torch.zeros(len(minibatches))

        for i, data in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx +
                                data[0].shape[0]]
            all_logits_idx += data[0].shape[0]
            nll = F.cross_entropy(logits, data[1].cuda().long())
            losses[i] = nll

        mean = losses.mean()
        penalty = ((losses - mean) ** 2).mean()
        loss = mean + penalty_weight * penalty

        opt.zero_grad()
        loss.backward()
        opt.step()
        if sch:
            sch.step()

        self.update_count += 1
        return {'loss': loss.item(), 'nll': nll.item(),
                'penalty': penalty.item()}
