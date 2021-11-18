#!/usr/bin/env python3.6
'''Probabilistic Programming Package.

The prototype is distributions, which can be a conditional one with
functions for parameters to define the dependency. Distribution
multiplication is implemented, as well as the mean, expectation,
sampling with backprop capability, and log-probability.

This file is greatly inspired by `torch.distributions`, with some components adopted.
'''

import warnings
import math
from contextlib import suppress
import torch as tc
import torch.distributions.utils as tcdu
from .base import Distr, DistrElem
from .utils import edic, edicify, expand_front, fargnames, fedic, is_scalar, normalize_logits, normalize_probs, precision_to_scale_tril, reduce_last, tcsize_div, tensorify

__author__ = "Chang Liu"
__version__ = "1.0.1"
__email__ = "changliu@microsoft.com"

class Determ(DistrElem):
    def __init__(self, name: str, val, shape = None, checkval: bool=False, device = None):
        super(Determ, self).__init__(name, shape, device, _fn = val)
        self._checkval = checkval

    def mean(self, conds: edic=edic(), n_mc: int=10, repar: bool=True) -> edic:
        # [shape_bat, shape_cond] -> [shape_bat, shape_var]
        return edic({self.name: self._fn(conds)}) # `repar` is only for `draw()`

    def mode(self, conds: edic=edic(), repar: bool=True) -> edic:
        # [shape_bat, shape_cond] -> [shape_bat, shape_var]
        return self.mean(conds, 0, repar)

    def draw(self, shape_mc: tc.Size=tc.Size(), conds: edic=edic(), repar: bool=False) -> edic: # vals
        # shape_mc, [shape_bat, shape_cond] -> [shape_mc, shape_bat, shape_var]
        with suppress() if repar else tc.no_grad():
            return edic({self.name: expand_front(self._fn(conds), shape_mc)})

    def logp(self, vals: edic, conds: edic=edic()) -> tc.Tensor: # log_probs
        # [shape_bat, shape_var], [shape_bat, shape_cond] -> [shape_bat]
        val = vals[self.name]
        if self._checkval:
            equals_all = (val - self._fn(conds)).abs() <= 1e-8 + 1e-5*val.abs() # shape matches
            probs = reduce_last(tc.all, equals_all, len(self.shape)).type(val.dtype) # 0. or 1.
            return probs.log() # -inf or 0. (probs can be recovered by tc.exp)
        else:
            return tc.zeros(tcsize_div(val.shape, self.shape),
                    dtype=val.dtype, device=val.device, layout=val.layout)

    def entropy(self, conds: edic=edic()) -> tc.Tensor:
        # [shape_bat, shape_cond] -> [shape_bat]
        return tc.zeros(size=Distr.shape_bat(conds), device=self.device)

class Normal(DistrElem):
    def __init__(self, name: str, mean = 0., std = 1., shape = None, device = None):
        super(Normal, self).__init__(name, shape, device, _meanfn = mean, _stdfn = std)
        self._log_const = .5 * math.log(2*math.pi) * tc.tensor(self.shape).prod()

    def mean(self, conds: edic=edic(), n_mc: int=10, repar: bool=True) -> edic:
        # [shape_bat, shape_cond] -> [shape_bat, shape_var]
        return edic({self.name: self._meanfn(conds)})

    def mode(self, conds: edic=edic(), repar: bool=True) -> edic:
        # [shape_bat, shape_cond] -> [shape_bat, shape_var]
        return self.mean(conds, 0, repar)

    def draw(self, shape_mc: tc.Size=tc.Size(), conds: edic=edic(), repar: bool=False) -> edic: # vals
        # shape_mc, [shape_bat, shape_cond] -> [shape_mc, shape_bat, shape_var]
        with suppress() if repar else tc.no_grad():
            meanval = expand_front(self._meanfn(conds), shape_mc) # [shape_mc, shape_bat, shape_var]
            return edic({self.name: self._stdfn(conds) * tc.randn_like(meanval) + meanval})

    def logp(self, vals: edic, conds: edic=edic()) -> tc.Tensor: # log_probs
        # [shape_bat, shape_var], [shape_bat, shape_cond] -> [shape_bat]
        meanval, stdval = self._meanfn(conds), self._stdfn(conds)
        normalized_vals = (vals[self.name] - meanval) / stdval
        quads = reduce_last(tc.sum, normalized_vals ** 2, len(self.shape))
        half_log_det = reduce_last(tc.sum, stdval.log(), len(self.shape))
        return -.5 * quads - half_log_det - self._log_const

    def entropy(self, conds: edic=edic()) -> tc.Tensor:
        # [shape_bat, shape_cond] -> [shape_bat]
        stdval = self._stdfn(conds)
        half_log_det = reduce_last(tc.sum, stdval.log(), len(self.shape))
        return half_log_det + self._log_const + .5 * tc.tensor(self.shape).prod()

class MVNormal(DistrElem):
    def __init__(self, name: str, mean = 0., cov = None, prec = None, std_tril = None, shape = None, device = None):
        input_indicator = (cov is not None) + (prec is not None) + (std_tril is not None)
        if input_indicator > 1: raise ValueError(f"For {self}, at most one of covariance_matrix or precision_matrix or scale_tril can be specified")
        if input_indicator == 0: std_tril = 1.
        if device is None: device = Distr.default_device
        if shape is None and Distr.has_name(name): shape = Distr.shape_var(name)

        def _vecterize_mean(mean):
            if mean.ndim == 0:
                warnings.warn("shape of `mean` expanded by 1 ndim")
                return mean[None]
            else: return mean

        def _matrixize_std(std_arg):
            if std_arg.ndim == 0:
                warnings.warn("shape of `std_arg` expanded by 2 ndims")
                return std_arg[None, None]
            elif std_arg.ndim == 1:
                warnings.warn("shape of `std_arg` expanded by 1 ndim")
                return tc.diag_embed(std_arg)
            else: return std_arg

        if cov is not None: std_arg = cov
        elif prec is not None: std_arg = prec
        else: std_arg = std_tril
        if callable(std_arg):
            if callable(mean):
                if shape is None: raise RuntimeError(f"For {self}, argument `shape` has to be provided when both parameters are functions")
                parents = fargnames(mean) | fargnames(std_arg)
                if fargnames(mean): self._meanfn = fedic(mean)
                else: self._meanfn = lambda conds: tensorify(device, mean())[0].expand(Distr.shape_bat(conds) + shape)
            else:
                parents = fargnames(std_arg)
                mean, = tensorify(device, mean,)
                mean = _vecterize_mean(mean)
                if shape is None: shape = mean.shape
                self._meanfn = lambda conds: mean.expand(Distr.shape_bat(conds) + shape)
            # un-indent
            if fargnames(std_arg): _std_argfn = fedic(std_arg)
            else: _std_argfn = lambda conds: tensorify(device, std_arg())[0].expand(Distr.shape_bat(conds) + shape + shape[-1:])
            if cov is not None: self._std_trilfn = lambda conds: tc.cholesky(_std_argfn(conds))
            elif prec is not None: self._std_trilfn = lambda conds: precision_to_scale_tril(_std_argfn(conds))
            else: self._std_trilfn = _std_argfn
        else:
            if callable(mean):
                parents = fargnames(mean)
                std_arg, = tensorify(device, std_arg,)
                std_arg = _matrixize_std(std_arg)
                if shape is None: shape = std_arg.shape[:-1]
                if parents: self._meanfn = fedic(mean)
                else: self._meanfn = lambda conds: tensorify(device, mean())[0].expand(Distr.shape_bat(conds) + shape)
            else:
                parents = set()
                mean, std_arg = tensorify(device, mean, std_arg)
                mean = _vecterize_mean(mean)
                std_arg = _matrixize_std(std_arg)
                if shape is None: shape = tc.broadcast_tensors(mean.unsqueeze(-1), std_arg)[0].shape[:-1]
                self._meanfn = lambda conds: mean.expand(Distr.shape_bat(conds) + shape)
            # un-indent
            if cov is not None: std_tril = tc.cholesky(std_arg)
            elif prec is not None: std_tril = precision_to_scale_tril(std_arg)
            else: std_tril = std_arg
            self._std_trilfn = lambda conds: std_tril.expand(Distr.shape_bat(conds) + shape + shape[-1:])
        super(DistrElem, self).__init__(names_shapes = {name: shape}, parents = parents)
        self._name, self._shape, self._device = name, shape, device
        self._log_const = .5 * math.log(2*math.pi) * tc.tensor(shape).prod()

    def mean(self, conds: edic=edic(), n_mc: int=10, repar: bool=True) -> edic:
        # [shape_bat, shape_cond] -> [shape_bat, shape_var]
        return edic({self.name: self._meanfn(conds)})

    def mode(self, conds: edic=edic(), repar: bool=True) -> edic:
        # [shape_bat, shape_cond] -> [shape_bat, shape_var]
        return self.mean(conds, 0, repar)

    def draw(self, shape_mc: tc.Size=tc.Size(), conds: edic=edic(), repar: bool=False) -> edic: # vals
        # shape_mc, [shape_bat, shape_cond] -> [shape_mc, shape_bat, shape_var]
        with suppress() if repar else tc.no_grad():
            meanval = expand_front(self._meanfn(conds), shape_mc) # [shape_mc, shape_bat, shape_var]
            eps_ = tc.randn_like(meanval).unsqueeze(-1)
            return edic({self.name: (self._std_trilfn(conds) @ eps_).squeeze(-1) + meanval})

    def logp(self, vals: edic, conds: edic=edic()) -> tc.Tensor: # log_probs
        # [shape_bat, shape_var], [shape_bat, shape_cond] -> [shape_bat]
        meanval, std_trilval = self._meanfn(conds), self._std_trilfn(conds)
        centered_vals_ = (vals[self.name] - meanval).unsqueeze(-1)
        normalized_vals = tc.triangular_solve(centered_vals_, std_trilval, upper=False)[0].squeeze(-1)
        quads = reduce_last(tc.sum, normalized_vals ** 2, len(self.shape))
        half_log_det = reduce_last(tc.sum, std_trilval.diagonal(dim1=-2, dim2=-1).log(), len(self.shape))
        return -.5 * quads - half_log_det - self._log_const

    def entropy(self, conds: edic=edic()) -> tc.Tensor:
        # [shape_bat, shape_cond] -> [shape_bat]
        std_trilval = self._std_trilfn(conds)
        half_log_det = reduce_last(tc.sum, std_trilval.diagonal(dim1=-2, dim2=-1).log(), len(self.shape))
        return half_log_det + self._log_const + .5 * tc.tensor(self.shape).prod()

class Catg(DistrElem):
    def __init__(self, name: str, *, probs = None, logits = None, shape = None, normalized: bool=False, device = None):
        # This logit should be normalized (sumexp == 1). logit == log prob.
        if (probs is None) == (logits is None):
            raise ValueError(f"For {self}, one and only one of `probs` and `logits` required")
        params = logits if probs is None else probs
        if device is None: device = Distr.default_device
        if shape is None and Distr.has_name(name): shape = Distr.shape_var(name)
        if callable(params):
            if shape is None: shape = tc.Size()
            parents = fargnames(params)
            if parents: paramsfn = fedic(params)
            else: paramsfn = lambda conds: tensorify(device, params())[0].expand(Distr.shape_bat(conds) + shape + (-1,))
            if probs is None:
                self._logitsfn = paramsfn if normalized else lambda conds: normalize_logits(paramsfn(conds))
            else:
                self._probsfn = paramsfn if normalized else lambda conds: normalize_probs(paramsfn(conds))
        else:
            params, = tensorify(device, params,)
            if params.ndim < 1 or params.shape[-1] <= 1: raise ValueError(f"For {self}, use `Bern` for binary variables")
            if shape is None: shape = params.shape[:-1]
            parents = set()
            if not normalized:
                params = normalize_logits(params) if probs is None else normalize_probs(params)
            setattr( self, '_logitsfn' if probs is None else '_probsfn',
                    lambda conds: params.expand(Distr.shape_bat(conds) + shape + (-1,)) )
        super(DistrElem, self).__init__(names_shapes = {name: shape}, parents = parents)
        self._name, self._shape, self._device = name, shape, device

    @tcdu.lazy_property
    def _probsfn(self): return lambda conds: tcdu.logits_to_probs(self._logitsfn(conds))
    @tcdu.lazy_property
    def _logitsfn(self): return lambda conds: tcdu.probs_to_logits(self._probsfn(conds))
    # No `mean()`.

    def mode(self, conds: edic=edic(), repar: bool=True) -> edic:
        # [shape_bat, shape_cond] -> [shape_bat, shape_var]
        return edic({self.name: self._logitsfn(conds).argmax(dim=-1)})

    def draw(self, shape_mc: tc.Size=tc.Size(), conds: edic=edic(), repar: bool=False) -> edic: # vals
        # shape_mc, [shape_bat, shape_cond] -> [shape_mc, shape_bat, shape_var]
        if repar: warnings.warn(f"For categorical {self}, reparameterization for `draw` is not allowed")
        with tc.no_grad():
            probs = expand_front(self._probsfn(conds), shape_mc)
            shape_out, n_catg = probs.shape[:-1], probs.shape[-1]
            probs_flat = probs.reshape(-1, n_catg)
            return edic({self.name:
                tc.multinomial(probs_flat, num_samples=1).reshape(shape_out)})

    def logp(self, vals: edic, conds: edic=edic()) -> tc.Tensor: # log_probs
        # [shape_bat, shape_var], [shape_bat, shape_cond] -> [shape_bat]
        val = vals[self.name]
        logits = self._logitsfn(conds) # logit == log prob
        logits = logits.expand(val.shape + logits.shape[-1:])
        logps_all = logits.gather(dim=-1, index=val.unsqueeze(-1)).squeeze(-1)
        return reduce_last(tc.sum, logps_all, len(self.shape))

    def entropy(self, conds: edic=edic()) -> tc.Tensor:
        # [shape_bat, shape_cond] -> [shape_bat]
        logits = self._logitsfn(conds) # logit == log prob
        return - reduce_last(tc.sum, logits.exp() * logits, len(self.shape) + 1)

class Bern(DistrElem):
    def __init__(self, name: str, *, probs = None, logits = None, shape = None, device = None):
        # This logit is NOT normalized (it has the logit of 0 being 0). So logit != log prob.
        if (probs is None) == (logits is None):
            raise ValueError(f"For {self}, one and only one of `probs` and `logits` required")
        super(Bern, self).__init__(name, shape, device, **({'_logitsfn':logits} if probs is None else {'_probsfn':probs}))

    @tcdu.lazy_property
    def _probsfn(self): return lambda conds: tcdu.logits_to_probs(self._logitsfn(conds), is_binary=True)
    @tcdu.lazy_property
    def _logitsfn(self): return lambda conds: tcdu.probs_to_logits(self._probsfn(conds), is_binary=True)
    # No `mean()`.

    def mode(self, conds: edic=edic(), repar: bool=True) -> edic:
        # [shape_bat, shape_cond] -> [shape_bat, shape_var]
        return edic({self.name: (self._logitsfn(conds) > 0).long()})

    def draw(self, shape_mc: tc.Size=tc.Size(), conds: edic=edic(), repar: bool=False) -> edic: # vals
        # shape_mc, [shape_bat, shape_cond] -> [shape_mc, shape_bat, shape_var]
        if repar: warnings.warn(f"For Bernoulli {self}, reparameterization for `draw` is not allowed")
        with tc.no_grad():
            return edic({self.name:
                tc.bernoulli( expand_front(self._probsfn(conds), shape_mc) ).long()}) # tc.bernoulli returns float32, and allows backprop (.long() doesn't)

    def logp(self, vals: edic, conds: edic=edic()) -> tc.Tensor: # log_probs
        # [shape_bat, shape_var], [shape_bat, shape_cond] -> [shape_bat]
        logits = self._logitsfn(conds)
        logprobs = -tc.log1p(tc.exp(-logits)) # logit != log prob
        logps_all = tc.where(vals[self.name].bool(), logprobs, logprobs - logits) # shape matches
        return reduce_last(tc.sum, logps_all, len(self.shape))

    def entropy(self, conds: edic=edic()) -> tc.Tensor:
        # [shape_bat, shape_cond] -> [shape_bat]
        logits = self._logitsfn(conds)
        logprobs = -tc.log1p(tc.exp(-logits)) # logit != log prob
        return reduce_last(tc.sum, -logprobs + logits / (1+tc.exp(logits)), len(self.shape))

