#!/usr/bin/env python3.6
'''Supervised VAE (no s-v split, the CSGz ablation baseline)
'''
import sys
import math
import torch as tc
sys.path.append('..')
import distr as ds
from . import xdistr as xds

__author__ = "Chang Liu"
__email__ = "changliu@microsoft.com"

class SupVAE:
    @staticmethod
    def _get_priors(mean_s, std_s, shape_s, mvn_prior: bool = False, device = None):
        if not mvn_prior:
            p_s = ds.Normal('s', mean=mean_s, std=std_s, shape=shape_s)
            prior_params_list = []
        else:
            if len(shape_s) != 1:
                raise RuntimeError("only 1-dim vector is supported for `s` in `mvn_prior` mode")
            dim_s = shape_s[0]
            mean_s = tc.zeros(shape_s, device=device) if callable(mean_s) else ds.tensorify(device, mean_s)[0].expand(shape_s).clone().detach()
            std_s_offdiag = tc.zeros((dim_s, dim_s), device=device) # lower triangular of L_ss (excl. diag)
            if callable(std_s): # for diag of L_ss
                std_s_diag_param = tc.zeros(shape_s, device=device)
            else:
                std_s = ds.tensorify(device, std_s)[0].expand(shape_s)
                std_s_diag_param = std_s.log().clone().detach()
            prior_params_list = [mean_s, std_s_diag_param, std_s_offdiag]

            def std_s_tril(): # L_ss
                return std_s_offdiag.tril(-1) + std_s_diag_param.exp().diagflat()
            p_s = ds.MVNormal('s', mean=mean_s, std_tril=std_s_tril, shape=shape_s)
        return p_s, prior_params_list

    def __init__(self, shape_s, shape_x, dim_y,
            mean_x1s, std_x1s, logit_y1s,
            mean_s1x = None, std_s1x = None,
            tmean_s1x = None, tstd_s1x = None,
            mean_s = 0., std_s = 1.,
            learn_tprior = False, src_mvn_prior = False, tgt_mvn_prior = False, device = None):
        if device is not None: ds.Distr.default_device = device
        self._parameter_dict = {}
        self.shape_x, self.dim_y, self.shape_s = shape_x, dim_y, shape_s
        self.learn_tprior = learn_tprior

        self.p_x1s = ds.Normal('x', mean=mean_x1s, std=std_x1s, shape=shape_x)
        self.p_y1s = getattr(ds, 'Bern' if dim_y == 1 else 'Catg')('y', logits=logit_y1s)

        self.p_s, prior_params_list = self._get_priors(mean_s, std_s, shape_s, src_mvn_prior, device)
        if src_mvn_prior: self._parameter_dict.update(zip(['mean_s', 'std_s_diag_param', 'std_s_offdiag'], prior_params_list))
        self.p_sx = self.p_s * self.p_x1s

        if mean_s1x is not None:
            self.q_s1x = ds.Normal('s', mean=mean_s1x, std=std_s1x, shape=shape_s)
        else: self.q_s1x = None

        if tmean_s1x is not None:
            self.qt_s1x = ds.Normal('s', mean=tmean_s1x, std=tstd_s1x, shape=shape_s)
        else: self.qt_s1x = None

        if learn_tprior:
            if not tgt_mvn_prior:
                tmean_s = tc.zeros(shape_s, device=device) if callable(mean_s) else ds.tensorify(device, mean_s)[0].expand(shape_s).clone().detach()
                tstd_s_param = tc.zeros(shape_s, device=device) if callable(std_s) else ds.tensorify(device, std_s)[0].log().expand(shape_s).clone().detach()
                self._parameter_dict.update({'tmean_s': tmean_s, 'tstd_s_param': tstd_s_param})
                def tstd_s(): return tc.exp(tstd_s_param)
                self.pt_s, tprior_params_list = self._get_priors(tmean_s, tstd_s, shape_s, False, device)
            else:
                self.pt_s, tprior_params_list = self._get_priors(mean_s, std_s, shape_s, True, device)
                self._parameter_dict.update(zip(['tmean_s', 'tstd_s_diag_param', 'tstd_s_offdiag'], tprior_params_list))
        else: self.pt_s = self.p_s
        self.pt_sx = self.pt_s * self.p_x1s
        for param in self._parameter_dict.values(): param.requires_grad_()

    def parameters(self):
        for param in self._parameter_dict.values(): yield param

    def state_dict(self):
        return self._parameter_dict

    def load_state_dict(self, state_dict: dict):
        for name in list(self._parameter_dict.keys()):
            with tc.no_grad(): self._parameter_dict[name].copy_(state_dict[name])

    def get_lossfn(self, n_mc_q: int=0, reduction: str="mean", mode: str="defl", weight_da: float=None, wlogpi: float=None):
        if reduction == "mean": reducefn = tc.mean
        elif reduction == "sum": reducefn = tc.sum
        elif reduction is None or reduction == "none": reducefn = lambda x: x
        else: raise ValueError(f"unknown `reduction` '{reduction}'")

        if self.q_s1x is not None: # svae, svae-da2
            def lossfn_src(x: tc.Tensor, y: tc.LongTensor) -> tc.Tensor:
                return -reducefn( xds.elbo_z2xy(self.p_sx, self.p_y1s, self.q_s1x, {'x':x, 'y':y}, n_mc_q, wlogpi) )
        else:
            if self.learn_tprior: # svae-da
                def lossfn_src(x: tc.Tensor, y: tc.LongTensor) -> tc.Tensor:
                    return -reducefn( xds.elbo_z2xy_twist(self.pt_sx, self.p_y1s, self.p_s, self.pt_s, self.qt_s1x, {'x':x, 'y':y}, n_mc_q, wlogpi) )
                    # return -reducefn( xds.elbo_z2xy_twist_fixpt(self.p_x1s, self.p_y1s, self.p_s, self.pt_s, self.qt_s1x, {'x':x, 'y':y}, n_mc_q, wlogpi) )
            else: raise ValueError("Use `q_s1x` for the source loss when no new prior")

        def lossfn_tgt(xt: tc.Tensor) -> tc.Tensor:
            return -reducefn( ds.elbo(self.pt_sx, self.qt_s1x, {'x': xt}, n_mc_q) )
            # return -reducefn( xds.elbo_fixllh(self.pt_s, self.p_x1s, self.qt_s1x, {'x': xt}, n_mc_q) )

        if mode == "src": return lossfn_src
        elif mode == "tgt": return lossfn_tgt # may be invalid
        elif not mode or mode == "defl":
            if self.learn_tprior:
                def lossfn(x: tc.Tensor, y: tc.LongTensor, xt: tc.Tensor) -> tc.Tensor:
                    return lossfn_src(x,y) + weight_da * lossfn_tgt(xt)
                return lossfn
            else: return lossfn_src
        else: raise ValueError(f"unknown `mode` '{mode}'")

    # Utilities
    def llh(self, x: tc.Tensor, y: tc.LongTensor=None, n_mc_marg: int=64, use_q: bool=True, mode: str="src") -> float:
        if mode == "src":
            p_joint = self.p_sx
            q_cond = self.q_s1x if self.q_s1x else self.qt_s1x
        elif mode == "tgt":
            p_joint = self.pt_sx
            q_cond = self.qt_s1x if self.qt_s1x else self.q_s1x
        else: raise ValueError(f"unknown `mode` '{mode}'")
        if not use_q:
            if y is None: llh_vals = p_joint.marg({'x'}, n_mc_marg).logp({'x': x})
            else: llh_vals = (p_joint * self.p_y1s).marg({'x', 'y'}, n_mc_marg).logp({'x': x, 'y': y})
        else:
            if y is None: llh_vals = q_cond.expect(lambda dc: p_joint.logp(dc) - q_cond.logp(dc,dc),
                    {'x': x}, n_mc_marg, reducefn=tc.logsumexp) - math.log(n_mc_marg)
            else: llh_vals = q_cond.expect(lambda dc: (p_joint * self.p_y1s).logp(dc) - q_cond.logp(dc,dc),
                    {'x': x, 'y': y}, n_mc_marg, reducefn=tc.logsumexp) - math.log(n_mc_marg)
        return llh_vals.mean().item()

    def logit_y1x_src(self, x: tc.Tensor, n_mc_q: int=0, repar: bool=True):
        dim_y = 2 if self.dim_y == 1 else self.dim_y
        y_eval = ds.expand_front(tc.arange(dim_y, device=x.device), ds.tcsize_div(x.shape, self.shape_x))
        x_eval = ds.expand_middle(x, (dim_y,), -len(self.shape_x))
        obs_xy = ds.edic({'x': x_eval, 'y': y_eval})
        if self.q_s1x is not None:
            logits = (self.q_s1x.expect(lambda dc: self.p_y1s.logp(dc,dc), obs_xy, 0, repar) #, reducefn=tc.logsumexp)
                ) if n_mc_q == 0 else (
                    self.q_s1x.expect(lambda dc: self.p_y1s.logp(dc,dc),
                        obs_xy, n_mc_q, repar, reducefn=tc.logsumexp) - math.log(n_mc_q)
                )
        else:
            vwei_p_y1s_logp = lambda dc: self.p_s.logp(dc,dc) - self.pt_s.logp(dc,dc) + self.p_y1s.logp(dc,dc)
            logits = (self.qt_s1x.expect(vwei_p_y1s_logp, obs_xy, 0, repar) #, reducefn=tc.logsumexp)
                ) if n_mc_q == 0 else (
                    self.qt_s1x.expect(vwei_p_y1s_logp, obs_xy, n_mc_q, repar, reducefn=tc.logsumexp) - math.log(n_mc_q)
                )
        return (logits[..., 1] - logits[..., 0]).squeeze(-1) if self.dim_y == 1 else logits

    def generate(self, shape_mc: tc.Size=tc.Size(), mode: str="src") -> tuple:
        if mode == "src": smp_s = self.p_s.draw(shape_mc)
        elif mode == "tgt": smp_s = self.pt_s.draw(shape_mc)
        else: raise ValueError(f"unknown 'mode' '{mode}'")
        return self.p_x1s.mode(smp_s, False)['x'], self.p_y1s.mode(smp_s, False)['y']

