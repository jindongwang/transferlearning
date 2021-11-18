#!/usr/bin/env python3.6
''' The Semantic-Variation Generative Model.

I.e., the proposed Causal Semantic Generative model (CSG).
'''
import sys
import math
import torch as tc
sys.path.append('..')
import distr as ds
from . import xdistr as xds

__author__ = "Chang Liu"
__email__ = "changliu@microsoft.com"

class SemVar:
    @staticmethod
    def _get_priors(mean_s, std_s, shape_s, mean_v, std_v, shape_v, corr_sv, mvn_prior: bool = False, device = None):
        if not mvn_prior:
            if not callable(mean_s): mean_s_val = mean_s; mean_s = lambda: mean_s_val
            if not callable(std_s): std_s_val = std_s; std_s = lambda: std_s_val
            if not callable(mean_v): mean_v_val = mean_v; mean_v = lambda: mean_v_val
            if not callable(std_v): std_v_val = std_v; std_v = lambda: std_v_val
            if not callable(corr_sv):
                if not corr_sv**2 < 1.: raise ValueError("correlation coefficient larger than 1")
                corr_sv_val = corr_sv; corr_sv = lambda: corr_sv_val

            p_s = ds.Normal('s', mean=mean_s, std=std_s, shape=shape_s)
            dim_s, dim_v = tc.tensor(shape_s).prod(), tc.tensor(shape_v).prod()
            def mean_v1s(s):
                shape_bat = s.shape[:-len(shape_s)] if len(shape_s) else s.shape
                s_normal_flat = ((s - mean_s()) / std_s()).reshape(shape_bat+(dim_s,))
                v_normal_flat = s_normal_flat[..., :dim_v] if dim_v <= dim_s \
                    else tc.cat([s_normal_flat, tc.zeros(shape_bat+(dim_v-dim_s,), dtype=s.dtype, device=s.device)], dim=-1)
                return mean_v() + corr_sv() * std_v() * v_normal_flat.reshape(shape_bat+shape_v)
            def std_v1s(s):
                corr_sv_val = ds.tensorify(device, corr_sv())[0]
                return ( std_v() * (1. - corr_sv_val**2).sqrt()
                        ).expand( (s.shape[:-len(shape_s)] if len(shape_s) else s.shape) + shape_v )
            p_v1s = ds.Normal('v', mean=mean_v1s, std=std_v1s, shape=shape_v)
            p_v = ds.Normal('v', mean=mean_v, std=std_v, shape=shape_v)
            prior_params_list = []
        else:
            if len(shape_s) != 1 or len(shape_v) != 1:
                raise RuntimeError("only 1-dim vectors are supported for `s` and `v` in `mvn_prior` mode")
            dim_s = shape_s[0]; dim_v = shape_v[0]
            mean_s = tc.zeros(shape_s, device=device) if callable(mean_s) else ds.tensorify(device, mean_s)[0].expand(shape_s).clone().detach()
            mean_v = tc.zeros(shape_v, device=device) if callable(mean_v) else ds.tensorify(device, mean_v)[0].expand(shape_v).clone().detach()
            # Sigma_sv = L_sv L_sv^T, L_sv = (L_ss, 0; M_vs, L_vv)
            std_s_offdiag = tc.zeros((dim_s, dim_s), device=device) # lower triangular of L_ss (excl. diag)
            std_v_offdiag = tc.zeros((dim_v, dim_v), device=device) # lower triangular of L_vv (excl. diag)
            if callable(std_s): # for diag of L_ss
                std_s_diag_param = tc.zeros(shape_s, device=device)
            else:
                std_s = ds.tensorify(device, std_s)[0].expand(shape_s)
                std_s_diag_param = std_s.log().clone().detach()
            if callable(std_v): # for diag of L_vv
                std_v_diag_param = tc.zeros(shape_v, device=device)
            else:
                std_v = ds.tensorify(device, std_v)[0].expand(shape_v)
                std_v_diag_param = std_v.log().clone().detach()
            if any(callable(var) for var in [std_s, std_v, corr_sv]): # M_vs
                std_vs_mat = tc.zeros(dim_v, dim_s, device=device)
            else:
                std_vs_mat = tc.eye(dim_v, dim_s, device=device)
                dim_min = min(dim_s, dim_v)
                std_vs_diag = (ds.tensorify(device, corr_sv)[0].expand((dim_min,)) * std_s[:dim_min] * std_v[:dim_min]).sqrt()
                if dim_min == dim_s: std_vs_mat = (std_vs_mat @ std_vs_diag.diagflat()).clone().detach()
                else: std_vs_mat = (std_vs_diag.diagflat() @ std_vs_mat).clone().detach()
            prior_params_list = [mean_s, std_s_diag_param, std_s_offdiag, mean_v, std_v_diag_param, std_v_offdiag, std_vs_mat]

            def std_s_tril(): # L_ss
                return std_s_offdiag.tril(-1) + std_s_diag_param.exp().diagflat()
            p_s = ds.MVNormal('s', mean=mean_s, std_tril=std_s_tril, shape=shape_s)

            def mean_v1s(s):
                return mean_v + ( std_vs_mat @ tc.triangular_solve(
                        (s - mean_s).unsqueeze(-1), std_s_tril(),  upper=False)[0] ).squeeze(-1)
            def std_v1s_tril(s): # L_vv
                return ( std_v_offdiag.tril(-1) + std_v_diag_param.exp().diagflat()
                        ).expand( (s.shape[:-len(shape_s)] if len(shape_s) else s.shape) + (dim_v, dim_v) )
            p_v1s = ds.MVNormal('v', mean=mean_v1s, std_tril=std_v1s_tril, shape=shape_v)

            def cov_v(): # M_vs M_vs^T + L_vv L_vv^T
                L_vv = std_v_offdiag.tril(-1) + std_v_diag_param.exp().diagflat()
                return std_vs_mat @ std_vs_mat.T + L_vv @ L_vv.T
            p_v = ds.MVNormal('v', mean=mean_v, cov=cov_v, shape=shape_v)
        return p_s, p_v1s, p_v, prior_params_list

    def __init__(self, shape_s, shape_v, shape_x, dim_y,
            mean_x1sv, std_x1sv, logit_y1s,
            mean_v1x = None, std_v1x = None, mean_s1vx = None, std_s1vx = None,
            tmean_v1x = None, tstd_v1x = None, tmean_s1vx = None, tstd_s1vx = None,
            mean_s = 0., std_s = 1., mean_v = 0., std_v = 1., corr_sv = .5,
            learn_tprior = False, src_mvn_prior = False, tgt_mvn_prior = False, device = None):
        if device is not None: ds.Distr.default_device = device
        self._parameter_dict = {}
        self.shape_s, self.shape_v, self.shape_x, self.dim_y = shape_s, shape_v, shape_x, dim_y
        self.learn_tprior = learn_tprior

        self.p_x1sv = ds.Normal('x', mean=mean_x1sv, std=std_x1sv, shape=shape_x)
        self.p_y1s = getattr(ds, 'Bern' if dim_y == 1 else 'Catg')('y', logits=logit_y1s)

        self.p_s, self.p_v1s, self.p_v, prior_params_list = self._get_priors(
                mean_s, std_s, shape_s, mean_v, std_v, shape_v, corr_sv, src_mvn_prior, device)
        if src_mvn_prior: self._parameter_dict.update(zip([
                'mean_s', 'std_s_diag_param', 'std_s_offdiag', 'mean_v', 'std_v_diag_param', 'std_v_offdiag', 'std_vs_mat'
            ], prior_params_list))
        self.p_sv = self.p_s * self.p_v1s
        self.p_svx = self.p_sv * self.p_x1sv

        if mean_v1x is not None:
            self.q_v1x = ds.Normal('v', mean=mean_v1x, std=std_v1x, shape=shape_v)
            self.q_s1vx = ds.Normal('s', mean=mean_s1vx, std=std_s1vx, shape=shape_s)
            self.q_sv1x = self.q_v1x * self.q_s1vx
        else: self.q_v1x, self.q_s1vx, self.q_sv1x = None, None, None

        if tmean_v1x is not None:
            self.qt_v1x = ds.Normal('v', mean=tmean_v1x, std=tstd_v1x, shape=shape_v)
            self.qt_s1vx = ds.Normal('s', mean=tmean_s1vx, std=tstd_s1vx, shape=shape_s)
            self.qt_sv1x = self.qt_v1x * self.qt_s1vx
        else: self.qt_v1x, self.qt_s1vx, self.qt_sv1x = None, None, None

        if learn_tprior:
            if not tgt_mvn_prior:
                tmean_s = tc.zeros(shape_s, device=device) if callable(mean_s) else ds.tensorify(device, mean_s)[0].expand(shape_s).clone().detach()
                tmean_v = tc.zeros(shape_v, device=device) if callable(mean_v) else ds.tensorify(device, mean_v)[0].expand(shape_v).clone().detach()
                tstd_s_param = tc.zeros(shape_s, device=device) if callable(std_s) else ds.tensorify(device, std_s)[0].expand(shape_s).log().clone().detach()
                tstd_v_param = tc.zeros(shape_v, device=device) if callable(std_v) else ds.tensorify(device, std_v)[0].expand(shape_v).log().clone().detach()
                if callable(corr_sv): tcorr_sv_param = tc.zeros((), device=device)
                else:
                    val = (ds.tensorify(device, corr_sv)[0].reshape(()) + 1.) / 2.
                    tcorr_sv_param = (val / (1-val)).clone().log().detach()
                self._parameter_dict.update({'tmean_s': tmean_s, 'tmean_v': tmean_v,
                    'tstd_s_param': tstd_s_param, 'tstd_v_param': tstd_v_param, 'tcorr_sv_param': tcorr_sv_param})

                def tstd_s(): return tc.exp(tstd_s_param)
                def tstd_v(): return tc.exp(tstd_v_param)
                def tcorr_sv(): return 2. * tc.sigmoid(tcorr_sv_param) - 1.
                self.pt_s, self.pt_v1s, self.pt_v, tprior_params_list = self._get_priors(
                        tmean_s, tstd_s, shape_s, tmean_v, tstd_v, shape_v, tcorr_sv, False, device)
            else:
                self.pt_s, self.pt_v1s, self.pt_v, tprior_params_list = self._get_priors(
                        mean_s, std_s, shape_s, mean_v, std_v, shape_v, corr_sv, True, device)
                self._parameter_dict.update(zip([
                        'tmean_s', 'tstd_s_diag_param', 'tstd_s_offdiag', 'tmean_v', 'tstd_v_diag_param', 'tstd_v_offdiag', 'tstd_vs_mat'
                    ], tprior_params_list))
        else: self.pt_s, self.pt_v1s, self.pt_v = self.p_s, self.p_v, self.p_v # independent prior
        self.pt_sv = self.pt_s * self.pt_v1s
        self.pt_svx = self.pt_sv * self.p_x1sv
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

        if self.q_sv1x is not None: # svgm, svgm-da2
            def lossfn_src(x: tc.Tensor, y: tc.LongTensor) -> tc.Tensor:
                return -reducefn( xds.elbo_z2xy(self.p_svx, self.p_y1s, self.q_sv1x, {'x':x, 'y':y}, n_mc_q, wlogpi) )
        else:
            if self.learn_tprior: # svgm-da
                def lossfn_src(x: tc.Tensor, y: tc.LongTensor) -> tc.Tensor:
                    return -reducefn( xds.elbo_z2xy_twist(self.pt_svx, self.p_y1s, self.p_sv, self.pt_sv, self.qt_sv1x, {'x':x, 'y':y}, n_mc_q, wlogpi) )
                    # return -reducefn( xds.elbo_z2xy_twist_fixpt(self.p_x1sv, self.p_y1s, self.p_sv, self.pt_sv, self.qt_sv1x, {'x':x, 'y':y}, n_mc_q, wlogpi) )
            else: # svgm-ind
                def lossfn_src(x: tc.Tensor, y: tc.LongTensor) -> tc.Tensor:
                    return -reducefn( xds.elbo_z2xy_twist(self.pt_svx, self.p_y1s, self.p_v1s, self.p_v, self.qt_sv1x, {'x':x, 'y':y}, n_mc_q, wlogpi) )

        def lossfn_tgt(xt: tc.Tensor) -> tc.Tensor:
            return -reducefn( ds.elbo(self.pt_svx, self.qt_sv1x, {'x': xt}, n_mc_q) )
            # return -reducefn( xds.elbo_fixllh(self.pt_sv, self.p_x1sv, self.qt_sv1x, {'x': xt}, n_mc_q) )

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
            p_joint = self.p_svx
            q_cond = self.q_sv1x if self.q_sv1x else self.qt_sv1x
        elif mode == "tgt":
            p_joint = self.pt_svx
            q_cond = self.qt_sv1x if self.qt_sv1x else self.q_sv1x
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
        if self.q_sv1x is not None:
            logits = (self.q_sv1x.expect(lambda dc: self.p_y1s.logp(dc,dc), obs_xy, 0, repar) #, reducefn=tc.logsumexp)
                ) if n_mc_q == 0 else (
                    self.q_sv1x.expect(lambda dc: self.p_y1s.logp(dc,dc),
                        obs_xy, n_mc_q, repar, reducefn=tc.logsumexp) - math.log(n_mc_q)
                )
        else:
            vwei_p_y1s_logp = lambda dc: self.p_sv.logp(dc,dc) - self.pt_sv.logp(dc,dc) + self.p_y1s.logp(dc,dc)
            logits = (self.qt_sv1x.expect(vwei_p_y1s_logp, obs_xy, 0, repar) #, reducefn=tc.logsumexp)
                ) if n_mc_q == 0 else (
                    self.qt_sv1x.expect(vwei_p_y1s_logp, obs_xy, n_mc_q, repar, reducefn=tc.logsumexp) - math.log(n_mc_q)
                )
        return (logits[..., 1] - logits[..., 0]).squeeze(-1) if self.dim_y == 1 else logits

    def generate(self, shape_mc: tc.Size=tc.Size(), mode: str="src") -> tuple:
        if mode == "src": smp_sv = self.p_sv.draw(shape_mc)
        elif mode == "tgt": smp_sv = self.pt_sv.draw(shape_mc)
        else: raise ValueError(f"unknown 'mode' '{mode}'")
        return self.p_x1sv.mode(smp_sv, False)['x'], self.p_y1s.mode(smp_sv, False)['y']

