#!/usr/bin/env python3.6
""" Modification to the `distr` package for the structure of the
    Causal Semantic Generative model.
"""
import sys
import math
import torch as tc
sys.path.append('..')
from distr import Distr, edic

__author__ = "Chang Liu"
__email__ = "changliu@microsoft.com"

def elbo_z2xy(p_zx: Distr, p_y1z: Distr, q_z1x: Distr, obs_xy: edic, n_mc: int=0, wlogpi: float=1., repar: bool=True) -> tc.Tensor:
    """ For supervised VAE with structure x <- z -> y.
    Observations are supervised (x,y) pairs.
    For unsupervised observations of x data, use `elbo(p_zx, q_z1x, obs_x)` as VAE z -> x. """
    if n_mc == 0:
        q_y1x_logpval = q_z1x.expect(lambda dc: p_y1z.logp(dc,dc), obs_xy, 0, repar) #, reducefn=tc.logsumexp)
        if hasattr(q_z1x, "entropy"): # No difference for Gaussian
            expc_val = q_z1x.expect(lambda dc: p_zx.logp(dc,dc), obs_xy, 0, repar) + q_z1x.entropy(obs_xy)
        else:
            expc_val = q_z1x.expect(lambda dc: p_zx.logp(dc,dc) - q_z1x.logp(dc,dc), obs_xy, 0, repar)
        return wlogpi * q_y1x_logpval + expc_val
    else:
        q_y1x_pval = q_z1x.expect(lambda dc: p_y1z.logp(dc,dc).exp(), obs_xy, n_mc, repar)
        expc_val = q_z1x.expect(lambda dc: p_y1z.logp(dc,dc).exp() * (p_zx.logp(dc,dc) - q_z1x.logp(dc,dc)),
                obs_xy, n_mc, repar)
        return wlogpi * q_y1x_pval.log() + expc_val / q_y1x_pval
        # q_y1x_logpval = q_z1x.expect(lambda dc: p_y1z.logp(dc,dc), obs_xy, n_mc, repar,
        #         reducefn=tc.logsumexp) - math.log(n_mc)
        # expc_logval = q_z1x.expect(lambda dc: p_y1z.logp(dc,dc) + (p_zx.logp(dc,dc) - q_z1x.logp(dc,dc)).log(),
        #         obs_xy, n_mc, repar, reducefn=tc.logsumexp) - math.log(n_mc)
        # return wlogpi * q_y1x_logpval + (expc_logval - q_y1x_logpval).exp()

def elbo_z2xy_twist(pt_zx: Distr, p_y1z: Distr, p_z: Distr, pt_z: Distr, qt_z1x: Distr, obs_xy: edic, n_mc: int=0, wlogpi: float=1., repar: bool=True) -> tc.Tensor:
    vwei_p_y1z_logp = lambda dc: p_z.logp(dc,dc) - pt_z.logp(dc,dc) + p_y1z.logp(dc,dc) # z, y:
    if n_mc == 0:
        r_y1x_logpval = qt_z1x.expect(vwei_p_y1z_logp, obs_xy, 0, repar) #, reducefn=tc.logsumexp)
        if hasattr(qt_z1x, "entropy"): # No difference for Gaussian
            expc_val = qt_z1x.expect(lambda dc: pt_zx.logp(dc,dc), obs_xy, 0, repar) + qt_z1x.entropy(obs_xy)
        else:
            expc_val = qt_z1x.expect(lambda dc: pt_zx.logp(dc,dc) - qt_z1x.logp(dc,dc), obs_xy, 0, repar)
        return wlogpi * r_y1x_logpval + expc_val
    else:
        r_y1x_pval = qt_z1x.expect(lambda dc: vwei_p_y1z_logp(dc).exp(), obs_xy, n_mc, repar)
        expc_val = qt_z1x.expect( lambda dc: # z, x, y:
                vwei_p_y1z_logp(dc).exp() * (pt_zx.logp(dc,dc) - qt_z1x.logp(dc,dc)),
            obs_xy, n_mc, repar)
        return wlogpi * r_y1x_pval.log() + expc_val / r_y1x_pval
        # r_y1x_logpval = qt_z1x.expect(vwei_p_y1z_logp, obs_xy, n_mc, repar,
        #         reducefn=tc.logsumexp) - math.log(n_mc) # z, y:
        # expc_logval = qt_z1x.expect(lambda dc: # z, x, y:
        #         vwei_p_y1z_logp(dc) + (pt_zx.logp(dc,dc) - qt_z1x.logp(dc,dc)).log(),
        #     obs_xy, n_mc, repar, reducefn=tc.logsumexp) - math.log(n_mc)
        # return wlogpi * r_y1x_logpval + (expc_logval - r_y1x_logpval).exp()

def elbo_fixllh(p_prior: Distr, p_llh: Distr, q_cond: Distr, obs: edic, n_mc: int=10, repar: bool=True) -> tc.Tensor: # [shape_bat] -> [shape_bat]
    def logp_llh_nograd(dc):
        with tc.no_grad(): return p_llh.logp(dc,dc)
    if hasattr(q_cond, "entropy"):
        return q_cond.expect(lambda dc: p_prior.logp(dc,dc) + logp_llh_nograd(dc),
                obs, n_mc, repar) + q_cond.entropy(obs)
    else:
        return q_cond.expect(lambda dc: p_prior.logp(dc,dc) + logp_llh_nograd(dc) - q_cond.logp(dc,dc),
                obs, n_mc, repar)

def elbo_z2xy_twist_fixpt(p_x1z: Distr, p_y1z: Distr, p_z: Distr, pt_z: Distr, qt_z1x: Distr, obs_xy: edic, n_mc: int=0, wlogpi: float=1., repar: bool=True) -> tc.Tensor:
    def logpt_z_nograd(dc):
        with tc.no_grad(): return pt_z.logp(dc,dc)
    vwei_p_y1z_logp = lambda dc: p_z.logp(dc,dc) - logpt_z_nograd(dc) + p_y1z.logp(dc,dc) # z, y:
    if n_mc == 0:
        r_y1x_logpval = qt_z1x.expect(vwei_p_y1z_logp, obs_xy, 0, repar) #, reducefn=tc.logsumexp)
        if hasattr(qt_z1x, "entropy"):
            expc_val = qt_z1x.expect(lambda dc: logpt_z_nograd(dc) + p_x1z.logp(dc,dc),
                    obs_xy, 0, repar) + qt_z1x.entropy(obs_xy)
        else:
            expc_val = qt_z1x.expect(lambda dc: logpt_z_nograd(dc) + p_x1z.logp(dc,dc) - qt_z1x.logp(dc,dc), obs_xy, 0, repar)
        return wlogpi * r_y1x_logpval + expc_val
    else:
        r_y1x_pval = qt_z1x.expect(lambda dc: vwei_p_y1z_logp(dc).exp(), obs_xy, n_mc, repar)
        expc_val = qt_z1x.expect( lambda dc: # z, x, y:
                vwei_p_y1z_logp(dc).exp() * (logpt_z_nograd(dc) + p_x1z.logp(dc,dc) - qt_z1x.logp(dc,dc)),
            obs_xy, n_mc, repar)
        return wlogpi * r_y1x_pval.log() + expc_val / r_y1x_pval
        # r_y1x_logpval = qt_z1x.expect(vwei_p_y1z_logp, obs_xy, n_mc, repar,
        #         reducefn=tc.logsumexp) - math.log(n_mc) # z, y:
        # expc_logval = qt_z1x.expect(lambda dc: # z, x, y:
        #         vwei_p_y1z_logp(dc) + (logpt_z_nograd(dc) + p_x1z.logp(dc,dc) - qt_z1x.logp(dc,dc)).log(),
        #     obs_xy, n_mc, repar, reducefn=tc.logsumexp) - math.log(n_mc)
        # return wlogpi * r_y1x_logpval + (expc_logval - r_y1x_logpval).exp()

