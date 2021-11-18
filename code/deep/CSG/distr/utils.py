#!/usr/bin/env python3.6
'''Probabilistic Programming Package.

The prototype is distributions, which can be a conditional one with
functions for parameters to define the dependency. Distribution
multiplication is implemented, as well as the mean, expectation,
sampling with backprop capability, and log-probability.
'''

import torch as tc
from functools import partial, wraps
from inspect import signature

__author__ = "Chang Liu"
__version__ = "1.0.1"
__email__ = "changliu@microsoft.com"

# enhanced dictionary
class edic(dict):
    def __and__(self, other): return edic({k:other[k] for k in set(self) & set(other)}) # &
    def __rand__(self, other): return edic({k:self[k] for k in set(other) & set(self)}) # &
    def __or__(self, other): return edic({**self, **other}) # |
    def __ror__(self, other): return edic({**other, **self}) # |
    def __sub__(self, other): return edic({k:v for k,v in self.items() if k not in other}) # -
    def __rsub__(self, other): return edic({k:v for k,v in other.items() if k not in self}) # -
    def isdisjoint(self, other) -> bool: return set(self).isdisjoint(set(other))

    def sub(self, it, fn = None):
        return edic({k:self[k] for k in it} if fn is None else {k:fn(self[k]) for k in it})
    def sub_expand_front(self, it, shape: tc.Size):
        return self.sub(it, partial(expand_front, shape = shape))
    def subedic(self, it, fn = None, use_default = False, default = None):
        if not use_default:
            if fn is None: return edic({k:self[k] for k in it if k in self})
            else: return edic({k:fn(self[k]) for k in it if k in self})
        else:
            if fn is None: return edic({k: (self[k] if k in self else default) for k in it})
            else: return edic({k: fn(self[k] if k in self else default) for k in it})
    def sublist(self, it, fn = None, use_default = False, default = None):
        if not use_default:
            if fn is None: return [self[k] for k in it if k in self]
            else: return [fn(self[k]) for k in it if k in self]
        else:
            if fn is None: return [(self[k] if k in self else default) for k in it]
            else: return [fn(self[k] if k in self else default) for k in it]

    def key0(self): return next(iter(self))
    def value0(self): return next(iter(self.values()))
    def item0(self): return next(iter(self.items()))

    def mean(self, dim, keepdim: bool=False):
        return edic({k: v.mean(dim, keepdim) for k,v in self.items()})
    def expand_front(self, shape: tc.Size):
        return edic({k: v.expand(shape + v.shape) for k,v in self.items()})
    def broadcast(self):
        return edic(zip( self.keys(), tc.broadcast_tensors(*self.values()) ))

def edicify(*args) -> tuple:
    return tuple(arg if type(arg) is edic else edic(arg) for arg in args)

# helper functions
def fargnames(fn) -> set:
    # return set(fn.__code__.co_varnames) - {'self'} # Also includes temporary local variables
    return set(signature(fn).parameters.keys()) # Do not need to substract 'self'

def fedic(fn):
    return wraps(fn)( lambda dc: fn(**( fargnames(fn) & edicify(dc)[0] )) )
    # pms = signature(fn).parameters
    # return wraps(fn)( lambda dc: fn(**(
    #     edic({k:v.default for k,v in pms.items() if v.default is not v.empty})
    #     | (set(pms.keys()) & edicify(dc)[0])  )) )

def wrap4_multi_batchdims(fn, ndim_vars = 1):
    """
    Function decorator to allow multiple batch dims at the front of input tensors.
    For incorporating functions (e.g., `torch.nn.Conv*`, `torch.nn.BatchNorm*`) that require only one batch dim
    into the `distr` package.
    """
    allowed_types = [int, list, tuple]
    if type(ndim_vars) not in allowed_types:
        raise ValueError("`ndim_vars` must be within types {allowed_types}")

    def fn_new(*args, **kwargs):
        keys = tuple(kwargs.keys())
        args += tuple(kwargs.values())
        if type(ndim_vars) is int: ndims = [ndim_vars] * len(args)
        else: ndims = ndim_vars
        shapes_bat_var = [
                ( (arg.shape[:-ndim], arg.shape[-ndim:]) if ndim else (args.shape, tc.Size()) )
                    if type(arg) is tc.Tensor else
                (None, None)
            for arg, ndim in zip(args, ndims)]
        args_batflat = [
                arg.reshape(-1, *shape_var)
                    if type(arg) is tc.Tensor and len(shape_bat) > 1 else
                arg
            for arg, (shape_bat, shape_var) in zip(args, shapes_bat_var)]

        if len(keys):
            outs_batflat = fn( *args_batflat[:-len(keys)],
                    **dict(zip(keys, args_batflat[-len(keys):])) )
        else:
            outs_batflat = fn(*args_batflat)
        single_out = type(outs_batflat) not in {list, tuple}
        if single_out: outs_batflat = (outs_batflat,)
        outs = tuple(
                out.reshape(*shapes_bat_var[0][0], *out.shape[1:]) # use `shape_bat` of the first input var
                    if type(out) is tc.Tensor else
                out
            for out in outs_batflat)
        if single_out: return outs[0]
        else: return outs
    return fn_new

def append_attrs(obj, vardict, attrs): # obj = self, vardict = locals(), attrs = set(locals.keys()) - {'self'}
    for attr in attrs: setattr(obj, attr, vardict[attr])

# for tc.Tensor
def tensorify(device=None, *args) -> tuple:
    return tuple(arg.to(device) if type(arg) is tc.Tensor else tc.tensor(arg, device=device) for arg in args)

def is_scalar(ten: tc.Tensor) -> bool:
    return ten.squeeze().ndim == 0

def is_same_tensor(ten1: tc.Tensor, ten2: tc.Tensor) -> bool:
    return (ten1 is ten2) or (
            type(ten1) == type(ten2) == tc.Tensor
            and ten1.data_ptr() == ten2.data_ptr() and ten1.shape == ten2.shape)

def expand_front(ten: tc.Tensor, shape: tc.Size) -> tc.Tensor:
    return ten.expand(shape + ten.shape)

def flatten_last(ten: tc.Tensor, ndims: int):
    if ndims <= 1: return ten
    else: return ten.reshape(ten.shape[:-ndims] + (-1,))

def reduce_last(reducefn, ten: tc.Tensor, ndims: int): # tc.distributions.utils
    if ndims < 1: return ten
    else: return reducefn(ten.reshape(ten.shape[:-ndims] + (-1,)), dim=-1)
    # return reducefn(ten, dim=list(range(-1, -ndims-1, -1))) if ndims else ten # doesn't work for `tc.all`

def swap_dim_ranges(ten: tc.Tensor, dims1: tuple, dims2: tuple) -> tc.Tensor:
    if len(dims1) != 2 or len(dims2) != 2:
        raise ValueError("`dims1` and `dims2` must be 2-tuples of integers")
    dims1 = tuple(dim if dim >= 0 else dim+ten.ndim for dim in dims1)
    dims2 = tuple(dim if dim >= 0 else dim+ten.ndim for dim in dims2)
    if dims1[0] > dims1[1]: dims1 = (dims1[1], dims1[0])
    if dims2[0] > dims2[1]: dims2 = (dims2[1], dims2[0])
    if dims2[0] < dims1[1] and dims1[0] < dims2[1]:
        raise ValueError("`dims1` and `dims2` must define disjoint intevals")
    if dims2[1] <= dims1[0]: dims1, dims2 = dims2, dims1
    dimord = list(range(0, dims1[0])) + list(range(*dims2)) \
            + list(range(dims1[1], dims2[0])) \
            + list(range(*dims1)) + list(range(dims2[1], ten.ndim))
    return ten.permute(*dimord)

def expand_middle(ten: tc.Tensor, shape: tc.Size, pos: int) -> tc.Tensor:
    # Expand with `shape` in front of dim `pos`.
    if len(shape) == 0: return ten
    if pos < 0: pos += ten.ndim
    ten_expd = expand_front(ten, shape)
    if pos == 0: return ten_expd
    else: return swap_dim_ranges(ten_expd, (0, len(shape)), (len(shape), len(shape)+pos))

# for tc.Size
def tcsizeify(*args) -> tuple:
    return tuple(arg if type(arg) is tc.Size else tc.Size(arg) for arg in args)

def tcsize_div(sz1: tc.Size, sz2: tc.Size) -> tc.Size:
    if not sz2 or sz1[-len(sz2):] == sz2: return sz1[:(len(sz1)-len(sz2))]
    else: raise ValueError("sizes not match")

def tcsize_broadcast(*sizes) -> tc.Size:
    szfinal = tc.Size()
    for sz in sizes:
        szlong, szshort = (szfinal, sz) if len(szfinal) >= len(sz) else (sz, szfinal)
        for i in range(1, 1 + len(szshort)):
            if szshort[-i] != 1:
                if szlong[-i] == 1: szlong[-i] = szshort[-i]
                elif szshort[-i] != szlong[-i]: raise ValueError("sizes not match")
        szfinal = szlong
    return szfinal

# specific distribution utilities
def normalize_probs(probs: tc.Tensor) -> tc.Tensor:
    return probs / probs.sum(dim=-1, keepdim=True)

def normalize_logits(logits: tc.Tensor) -> tc.Tensor:
    return logits - logits.logsumexp(dim=-1, keepdim=True)

def precision_to_scale_tril(P: tc.Tensor) -> tc.Tensor:
    # Ref: https://nbviewer.jupyter.org/gist/fehiepsi/5ef8e09e61604f10607380467eb82006#Precision-to-scale_tril
    Lf = tc.cholesky(tc.flip(P, dims=(-2, -1)))
    L_inv = tc.transpose(tc.flip(Lf, dims=(-2, -1)), -2, -1)
    L = tc.triangular_solve(tc.eye(P.shape[-1], dtype=P.dtype, device=P.device),
            L_inv, upper=False)[0]
    return L

