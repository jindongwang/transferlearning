# coding=utf-8

import numpy as np
import torch
import torch.nn.functional as F


def _max_with_relu(a, b):
    return a + F.relu(b - a)


def _get_grad(out_, in_):
    grad, *_ = torch.autograd.grad(out_, in_,
                                   grad_outputs=torch.ones_like(
                                       out_, dtype=torch.float32),
                                   retain_graph=True)
    return grad.view(in_.shape[0], -1)


class LargeMarginLoss:
    def __init__(self,
                 gamma=10000.0,
                 alpha_factor=4.0,
                 top_k=1,
                 dist_norm=2,
                 epsilon=1e-8,
                 use_approximation=True,
                 loss_type="all_top_k",
                 reduce='mean'):

        self.dist_upper = gamma
        self.dist_lower = gamma * (1.0 - alpha_factor)

        self.alpha = alpha_factor
        self.top_k = top_k
        self.dual_norm = {1: np.inf, 2: 2, np.inf: 1}[dist_norm]
        self.eps = epsilon

        self.use_approximation = use_approximation
        self.loss_type = loss_type
        self.reduce = reduce

    def __call__(self, logits, onehot_labels, feature_maps):
        onehot_labels = torch.zeros(logits.size()).scatter_(
            1, onehot_labels.unsqueeze(1).cpu(), 1).to(logits.device)
        prob = F.softmax(logits, dim=1)
        correct_prob = prob * onehot_labels

        correct_prob = torch.sum(correct_prob, dim=1, keepdim=True)
        other_prob = prob * (1.0 - onehot_labels)

        if self.top_k > 1:
            topk_prob, _ = other_prob.topk(self.top_k, dim=1)
        else:
            topk_prob, _ = other_prob.max(dim=1, keepdim=True)

        diff_prob = correct_prob - topk_prob

        loss = torch.empty(0, device=logits.device)
        for feature_map in feature_maps:
            diff_grad = torch.stack([_get_grad(diff_prob[:, i], feature_map) for i in range(self.top_k)],
                                    dim=1)
            diff_gradnorm = torch.norm(diff_grad, p=self.dual_norm, dim=2)

            if self.use_approximation:
                diff_gradnorm.detach_()

            dist_to_boundary = diff_prob / (diff_gradnorm + self.eps)

            if self.loss_type == "worst_top_k":
                dist_to_boundary, _ = dist_to_boundary.min(dim=1)
            elif self.loss_type == "avg_top_k":
                dist_to_boundary = dist_to_boundary.mean(dim=1)

            loss_layer = _max_with_relu(dist_to_boundary, self.dist_lower)
            loss_layer = _max_with_relu(
                0, self.dist_upper - loss_layer) - self.dist_upper
            loss = torch.cat([loss, loss_layer])
        if self.reduce == 'mean':
            return loss.mean()
        else:
            if self.loss_type in ['worst_top_k', 'avg_top_k']:
                return loss
            return loss.mean(dim=1)
