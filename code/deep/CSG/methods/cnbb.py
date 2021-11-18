#!/usr/bin/env python3.6
""" Implementation of the CNBB method "ConvNet with Batch Balancing".

Based on the original description in <https://arxiv.org/abs/1906.02899>. No official code found.
"""
import torch as tc
from torch.nn.functional import normalize

__author__ = "Chang Liu"
__email__ = "changliu@microsoft.com"
# tc.autograd.set_detect_anomaly(True)

class CNBBLoss:
    def __init__(self, f_feat, actv, f_logit, dim_y, reg_w, reg_s, lr, n_iter):
        if actv not in {"Sigmoid", "Tanh"}: raise ValueError(f"unknown activation type '{actv}'")
        if dim_y == 1:
            celossobj = tc.nn.BCEWithLogitsLoss(reduction='none')
            self.celoss = lambda logits, y: celossobj(logits, y.float())
        else: self.celoss = tc.nn.CrossEntropyLoss(reduction='none')
        self.f_feat, self.actv, self.f_logit, self.reg_w, self.reg_s, self.lr, self.n_iter \
                = f_feat, actv, f_logit, reg_w, reg_s, lr, n_iter

    def __call__(self, x, y):
        n_bat = x.shape[0]
        # Inner iteration for weight
        with tc.no_grad(): feat = self.f_feat(x)
        feat = feat.reshape(n_bat, -1)
        if self.actv == "Sigmoid": is_treat = feat > .5
        elif self.actv == "Tanh": is_treat = feat > 0.
        weight = tc.full([n_bat], 1/n_bat, device=x.device, requires_grad=True)
        proj = (tc.eye(n_bat) - tc.ones(n_bat, n_bat) / n_bat).to(x.device)
        for it in range(self.n_iter):
            loss = ((feat.T @ (
                    normalize(weight[:,None] * is_treat, p=1, dim=0)
                    - normalize(weight[:,None] * ~is_treat, p=1, dim=0)
                ) * ~tc.eye(feat.shape[-1], dtype=bool, device=x.device))**2).sum() \
                    + self.reg_w * (weight**2).sum()
            loss.backward()
            with tc.no_grad():
                weight -= self.lr * (proj @ weight.grad)
                weight.abs_()
                weight /= weight.sum()
                weight.grad.zero_()
        # Optimize the model
        if self.actv == "Sigmoid": sqnorm_s = ((self.f_feat(x) - .5)**2).sum()
        elif self.actv == "Tanh": sqnorm_s = (self.f_feat(x)**2).sum()
        loss = weight.detach() @ self.celoss(self.f_logit(x), y) - self.reg_s * sqnorm_s
        return loss

