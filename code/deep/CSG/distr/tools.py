#!/usr/bin/env python3.6
'''Probabilistic Programming Package.

The prototype is distributions, which can be a conditional one with
functions for parameters to define the dependency. Distribution
multiplication is implemented, as well as the mean, expectation,
sampling with backprop capability, and log-probability.
'''

import math
import torch as tc
from .utils import edic
from .base import Distr

__author__ = "Chang Liu"
__version__ = "1.0.1"
__email__ = "changliu@microsoft.com"

def elbo(p_joint: Distr, q_cond: Distr, obs: edic, n_mc: int=10, repar: bool=True) -> tc.Tensor: # [shape_bat] -> [shape_bat]
    if hasattr(q_cond, "entropy"):
        return q_cond.expect(lambda dc: p_joint.logp(dc,dc), obs, n_mc, repar) + q_cond.entropy(obs)
    else:
        return q_cond.expect(lambda dc: p_joint.logp(dc,dc) - q_cond.logp(dc,dc), obs, n_mc, repar)

def elbo_z2xy(p_zx: Distr, p_y1z: Distr, q_z1x: Distr, obs_xy: edic, n_mc: int=0, repar: bool=True) -> tc.Tensor:
    """ For supervised VAE with structure x <- z -> y.
    Observations are supervised (x,y) pairs.
    For unsupervised observations of x data, use `elbo(p_zx, q_z1x, obs_x)` as VAE z -> x. """
    if n_mc == 0:
        q_y1x_logpval = q_z1x.expect(lambda dc: p_y1z.logp(dc,dc), obs_xy, 0, repar) #, reducefn=tc.logsumexp)
        if hasattr(q_z1x, "entropy"): # No difference for Gaussian
            expc_val = q_z1x.expect(lambda dc: p_zx.logp(dc,dc), obs_xy, 0, repar) + q_z1x.entropy(obs_xy)
        else:
            expc_val = q_z1x.expect(lambda dc: p_zx.logp(dc,dc) - q_z1x.logp(dc,dc), obs_xy, 0, repar)
        return q_y1x_logpval + expc_val
    else:
        q_y1x_pval = q_z1x.expect(lambda dc: p_y1z.logp(dc,dc).exp(), obs_xy, n_mc, repar)
        expc_val = q_z1x.expect(lambda dc: p_y1z.logp(dc,dc).exp() * (p_zx.logp(dc,dc) - q_z1x.logp(dc,dc)),
                obs_xy, n_mc, repar)
        return q_y1x_pval.log() + expc_val / q_y1x_pval
        # q_y1x_logpval = q_z1x.expect(lambda dc: p_y1z.logp(dc,dc), obs_xy, n_mc, repar,
        #         reducefn=tc.logsumexp) - math.log(n_mc)
        # expc_logval = q_z1x.expect(lambda dc: p_y1z.logp(dc,dc) + (p_zx.logp(dc,dc) - q_z1x.logp(dc,dc)).log(),
        #         obs_xy, n_mc, repar, reducefn=tc.logsumexp) - math.log(n_mc)
        # return q_y1x_logpval + (expc_logval - q_y1x_logpval).exp()

def elbo_z2xy_twist(pt_zx: Distr, p_y1z: Distr, p_z: Distr, pt_z: Distr, qt_z1x: Distr, obs_xy: edic, n_mc: int=0, repar: bool=True) -> tc.Tensor:
    vwei_p_y1z_logp = lambda dc: p_z.logp(dc,dc) - pt_z.logp(dc,dc) + p_y1z.logp(dc,dc) # z, y:
    if n_mc == 0:
        r_y1x_logpval = qt_z1x.expect(vwei_p_y1z_logp, obs_xy, 0, repar) #, reducefn=tc.logsumexp)
        if hasattr(qt_z1x, "entropy"): # No difference for Gaussian
            expc_val = qt_z1x.expect(lambda dc: pt_zx.logp(dc,dc), obs_xy, 0, repar) + qt_z1x.entropy(obs_xy)
        else:
            expc_val = qt_z1x.expect(lambda dc: pt_zx.logp(dc,dc) - qt_z1x.logp(dc,dc), obs_xy, 0, repar)
        return r_y1x_logpval + expc_val
    else:
        r_y1x_pval = qt_z1x.expect(lambda dc: vwei_p_y1z_logp(dc).exp(), obs_xy, n_mc, repar)
        expc_val = qt_z1x.expect( lambda dc: # z, x, y:
                vwei_p_y1z_logp(dc).exp() * (pt_zx.logp(dc,dc) - qt_z1x.logp(dc,dc)),
            obs_xy, n_mc, repar)
        return r_y1x_pval.log() + expc_val / r_y1x_pval
        # r_y1x_logpval = qt_z1x.expect(vwei_p_y1z_logp, obs_xy, n_mc, repar,
        #         reducefn=tc.logsumexp) - math.log(n_mc) # z, y:
        # expc_logval = qt_z1x.expect(lambda dc: # z, x, y:
        #         vwei_p_y1z_logp(dc) + (pt_zx.logp(dc,dc) - qt_z1x.logp(dc,dc)).log(),
        #     obs_xy, n_mc, repar, reducefn=tc.logsumexp) - math.log(n_mc)
        # return r_y1x_logpval + (expc_logval - r_y1x_logpval).exp()

def elbo_zy2x(p_zyx: Distr, q_y1x: Distr, q_z1xy: Distr, obs_x: edic, n_mc: int=0, repar: bool=True) -> tc.Tensor:
    """ For supervised VAE with structure z -> x <- y (Kingma's semi-supervised VAE, M2). (z,y) correlation also allowed.
    Observations are unsupervised x data.
    For supervised observations of (x,y) pairs, use `elbo(p_zyx, q_z1xy, obs_xy)` as VAE z -> (x,y). """
    if hasattr(q_y1x, "entropy"):
        return q_y1x.expect(lambda dc: elbo(p_zyx, q_z1xy, dc, n_mc, repar),
                obs_x, n_mc, repar) + q_y1x.entropy(obs_x)
    else:
        return q_y1x.expect(lambda dc: elbo(p_zyx, q_z1xy, dc, n_mc, repar) - q_y1x.logp(dc,dc),
                obs_x, n_mc, repar)

