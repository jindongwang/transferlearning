#!/usr/bin/env python3.6
'''Probabilistic Programming Package.

The prototype is distributions, which can be a conditional one with
functions for parameters to define the dependency. Distribution
multiplication is implemented, as well as the mean, expectation,
sampling with backprop capability, and log-probability.
'''

import math
import torch as tc
from .utils import edic, edicify, expand_front, fargnames, fedic, swap_dim_ranges, tcsizeify, tcsize_broadcast, tcsize_div, tensorify

__author__ = "Chang Liu"
__version__ = "1.0.1"
__email__ = "changliu@microsoft.com"

class Distr:
    '''
    names:   set <-> vals:  edic(name:tensor)
    parents: set <-> conds: edic(name:tensor)
    '''
    default_device = None
    _shapes_all_vars = {} # dict

    @staticmethod
    def all_names() -> set: return set(Distr._shapes_all_vars)
    @staticmethod
    def clear(): Distr._shapes_all_vars.clear()
    @staticmethod
    def has_name(name: str) -> bool: return name in Distr._shapes_all_vars
    @staticmethod
    def has_names(names: set) -> bool:
        return all(name in Distr._shapes_all_vars for name in names)
    @staticmethod
    def shape_var(name: str) -> tc.Size: return Distr._shapes_all_vars[name]
    @staticmethod
    def shapes_var(names: set) -> dict:
        return {name: Distr._shapes_all_vars[name] for name in names}
    @staticmethod
    def shape_bat(conds: edic) -> tc.Size:
        for name, ten in conds.items():
            if Distr.has_name(name): return tcsize_div(ten.shape, Distr.shape_var(name))
        return tc.Size()
    @staticmethod
    def shape_bat_broadcast(conds: edic) -> tc.Size:
        shapes_bat = [tcsize_div(ten.shape, Distr.shape_var(name))
                for name, ten in conds.items() if Distr.has_name(name)]
        return tcsize_broadcast(*shapes_bat)
    @staticmethod
    def broadcast_vars(conds: edic) -> edic:
        shape_bat = Distr.shape_bat_broadcast(conds)
        return edic({name: ten.expand(shape_bat + Distr.shape_var(name))
                for name, ten in conds.items() if Distr.has_name(name)})

    def __init__(self, *, names: set=set(), names_shapes: dict={}, parents: set=set()):
        for name, shape in names_shapes.items():
            shape, = tcsizeify(shape,)
            if Distr.has_name(name):
                if shape != Distr.shape_var(name): raise ValueError(f"shape not match for existing variable '{name}'")
            else: Distr._shapes_all_vars[name] = shape
        for name in names:
            if not Distr.has_name(name): raise ValueError(f"new variable '{name}' needs a shape")
        names = names | set(names_shapes)
        if not names: raise ValueError("name(s) have to be provided")
        names_comm = names & parents
        if names_comm: raise ValueError(f"common variable(s) '{names_comm}' found in `names` and `parents`")
        self._names, self._parents, self._is_root = names, parents, (not bool(parents))

    @property
    def names(self) -> set: return self._names
    @property
    def parents(self) -> set: return self._parents
    @property
    def is_root(self) -> bool: return self._is_root

    def __repr__(self) -> str:
        return "p(" + ", ".join(self.names) + (" | " + ", ".join(self.parents) + ")" if self.parents else ")")

    def mean(self, conds: edic=edic(), n_mc: int=10, repar: bool=True) -> edic:
        # [shape_bat, shape_cond] -> [shape_bat, shape_var]
        raise NotImplementedError

    def expect(self, fn, conds: edic=edic(), n_mc: int=10, repar: bool=True, reducefn = tc.mean) -> tc.Tensor:
        # [shape_bat] -> [shape_bat]
        if n_mc == 0:
            vals = self.mean(conds, 0, repar)
            return fn(conds|vals)
        elif n_mc > 0:
            vals = self.draw(tc.Size((n_mc,)), conds, repar)
            return reducefn(fn( edicify(conds)[0].expand_front((n_mc,)) | vals ), dim=0)
        else: raise ValueError(f"For {self}, negative `n_mc` {n_mc} encountered")

    def rdraw(self, shape_mc: tc.Size=tc.Size(), conds: edic=edic()) -> edic: # vals
        # shape_mc, [shape_bat, shape_cond] -> [shape_mc, shape_bat, shape_var]
        return self.draw(shape_mc, conds, True)

    def draw(self, shape_mc: tc.Size=tc.Size(), conds: edic=edic(), repar: bool=False) -> edic: # vals
        # shape_mc, [shape_bat, shape_cond] -> [shape_mc, shape_bat, shape_var]
        raise NotImplementedError

    def logp(self, vals: edic, conds: edic=edic()) -> tc.Tensor: # log_probs
        # [shape_bat, shape_var], [shape_bat, shape_cond] -> [shape_bat]
        raise NotImplementedError

    def logp_cartes(self, vals: edic, conds: edic=edic()) -> tc.Tensor: # log_probs
        # [shape_mc, shape_var], [shape_bat, shape_cond] -> [shape_mc, shape_bat]
        vals, conds = edicify(vals, conds)
        shape_mc, shape_bat = Distr.shape_bat(vals), Distr.shape_bat(conds)
        len_mc, len_bat = len(shape_mc), len(shape_bat)
        vals_expd = vals.sub( self.names,
                lambda v: swap_dim_ranges(expand_front(v, shape_bat), (0, len_bat), (len_bat, len_bat+len_mc)) )
        conds_expd = conds.sub_expand_front(self.parents, shape_mc)
        return self.logp(vals_expd, conds_expd)

    def __mul__(self, other): # p(z|w1) * p(x|z,w2) -> p(z,x|w1,w2)
        if not self.names.isdisjoint(other.names):
            raise ValueError(f"cycle found for {self} * {other}: common variable")
        if not self.parents.isdisjoint(other.names):
            if not self.names.isdisjoint(other.parents):
                raise ValueError(f"cycle found for {self} * {other}: cyclic generation")
            else: p_marg, p_cond = other, self
        else: p_marg, p_cond = self, other
        indep = p_marg.names.isdisjoint(p_cond.parents)
        p_joint = Distr(names = p_marg.names | p_cond.names,
                parents = p_marg.parents | p_cond.parents - p_marg.names)
        p_joint._p_marg, p_joint._p_cond, p_joint._indep = p_marg, p_cond, indep

        def _mean(conds: edic=edic(), n_mc: int=10, repar: bool=True) -> edic:
            mean_marg = p_marg.mean(conds, n_mc, repar) # [shape_bat, shape_var_marg]
            # mean_cond = p_marg.expect(lambda dc: p_cond.mean(dc,dc), conds, n_mc, repar, tc.mean) # Commented for the lack of tc.mean(edic) and efficiency concern
            if indep:
                mean_cond = p_cond.mean(conds, n_mc, repar) # [shape_bat, shape_var_cond]
            else:
                if n_mc == 0:
                    mean_cond = p_cond.mean(conds|mean_marg, 0, repar) # [shape_bat, shape_var_cond]
                elif n_mc > 0:
                    vals_marg = p_marg.draw((n_mc,), conds, repar) # [n_mc, shape_bat, shape_var_marg]
                    mean_cond = p_cond.mean(edicify(conds)[0].sub_expand_front(p_joint.parents, (n_mc,)) | vals_marg,
                            n_mc, repar).mean(dim=0) # [shape_bat, shape_var_cond]
                else: raise ValueError(f"For {p_joint}, negative `n_mc` {n_mc} encountered")
            return mean_marg | mean_cond
        p_joint.mean = _mean

        def _draw(shape_mc: tc.Size=tc.Size(), conds: edic=edic(), repar: bool=False) -> edic: # vals
            vals_marg = p_marg.draw(shape_mc, conds, repar) # [shape_mc, shape_bat, shape_var_marg]
            if indep:
                vals_cond = p_cond.draw(shape_mc, conds, repar) # [shape_mc, shape_bat, shape_var_cond]
            else:
                vals_cond = p_cond.draw(tc.Size(),
                        edicify(conds)[0].sub_expand_front(p_joint.parents, shape_mc) | vals_marg,
                    repar) # [shape_mc, shape_bat, shape_var_cond]
            return vals_marg | vals_cond
        p_joint.draw = _draw

        def _logp(vals: edic, conds: edic=edic()) -> tc.Tensor: # log_probs
            return p_marg.logp(vals, conds) + p_cond.logp(vals, edicify(conds)[0]|vals) # [shape_bat] + [shape_bat] guaranteed
        p_joint.logp = _logp

        def _entropy(conds: edic=edic(), n_mc: int=10, repar: bool=True) -> tc.Tensor:
            # [shape_bat, shape_cond] -> [shape_bat]
            return p_marg.entropy(conds) + (
                    p_cond.entropy(conds) if indep else
                    p.marg.expect(p_cond.entropy, conds, n_mc, repar, tc.mean) )
        if indep: p_joint.entropy = _entropy

        return p_joint

    def marg(self, mnames: set, n_mc: int=10):
        if mnames == self.names: return self
        if not mnames: raise ValueError(f"'mnames' empty for {self}")
        names_irrelev = mnames - self.names;
        if names_irrelev: raise ValueError(f"irrelevant variable(s) {names_irrelev} found for {self}")
        # To filter out non-elem distr that is formed by MC marginalization
        if hasattr(self, '_p_marg') and hasattr(self, '_p_cond'):
            # Marg p(ym, yo, xm, xo, zm, zo) = p(xm, xo, zm, zo) p(ym, yo | xm, xo)
            # for mnames = {ym, xm, zm}. Other conditioned vars omitted.
            p_marg, p_cond = self._p_marg, self._p_cond
            # ym empty. Marg p(xm, xo, zm, zo) for {xm, zm}
            if mnames <= p_marg.names: return p_marg.marg(mnames, n_mc)
            # ym not empty
            mnames_marg, mnames_cond = (mnames & p_marg.names), (mnames & p_cond.names) # {xm, zm}, {ym}
            p_condm = p_cond.marg(mnames_cond, n_mc) # p(ym | xm, xo)
            names_intsec = p_marg.names & p_condm.parents # {xm, xo}
            if not names_intsec: # {xm, xo} empty. Marg p(zm, zo) p(ym) for {ym, zm}
                return p_marg.marg(mnames_marg, n_mc) * p_condm if mnames_marg else p_condm # p(zm) p(ym) if zm not empty else p(ym)
            else: # {xm, xo} not empty
                p_margm = p_marg.marg(mnames_marg | names_intsec, n_mc) # p(xm, xo, zm)
                if not mnames_marg: # label: L0
                    # {xm, zm} empty. Marg p(xo) p(ym | xo) for ym
                    p_joint = p_margm * p_condm # p(xo) p(ym | xo)
                    p_res = Distr(names = mnames, parents = p_joint.parents)
                    def _mean(conds: edic=edic(), n_mc: int=10, repar: bool=True) -> edic:
                        return p_joint.mean(conds, n_mc, repar).sub(mnames)
                    p_res.mean = _mean
                    def _draw(shape_mc: tc.Size=tc.Size(), conds: edic=edic(), repar: bool=False) -> edic: # vals
                        return p_joint.draw(shape_mc, conds, repar).sub(mnames)
                    p_res.draw = _draw
                    def _logp(vals: edic, conds: edic=edic()) -> tc.Tensor: # log_probs
                        return p_margm.expect(lambda dc: p_condm.logp(dc,dc), # `dc` contains properly expanded `conds` and `vals`
                                conds|vals, n_mc, True, tc.logsumexp) - math.log(n_mc)
                    p_res.logp = _logp
                    return p_res
                else: # {xm, zm} not empty. Marg p(xm, xo, zm) p(ym | xm, xo) for {ym, xm, zm}
                    if names_intsect <= mnames_marg: # xo is empty. Marg p(xm, zm) p(ym | xm) for {ym, xm, zm}
                        return p_margm * p_condm # p(xm, zm) p(ym | xm)
                    else: # xo is not empty
                        if hasattr(p_margm, '_p_marg') and hasattr(p_margm, '_p_cond'):
                            if p_margm._p_marg.names == mnames_marg: # p(xm, xo, zm) = p(xm, zm) p(xo | xm, zm)
                                # Marg p(xm, zm) p(xo | xm, zm) p(ym | xm, xo) for {ym, xm, zm} is p(xm, zm) p(ym | xm, zm),
                                # where p(ym | xm, zm) is from marg p(xo | xm, zm) p(ym | xm, xo) for ym
                                return p_margm._p_marg * (p_margm._p_cond * p_condm).marg(mnames_cond, n_mc) # goto L0
                            elif p_margm._p_cond.names == mnames_marg:
                                if p_margm._indep: # p(xm, xo, zm) = p(xo) p(xm, zm)
                                    # Similar to the above. p(xm, zm) p(ym | xm, zm),
                                    # where p(ym | xm, zm) is from marg p(xo) p(ym | xm, xo) for ym
                                    return p_margm._p_cond * (p_margm._p_marg * p_condm).marg(mnames_cond, n_mc) # goto L0
                                else: # p(xm, xo, zm) = p(xo) p(xm, zm | xo)
                                    # Marg p(xo) p(xm, zm | xo) p(ym | xm, xo) for {ym, xm, zm}
                                    return (p_margm._p_marg * (p_margm._p_cond * p_condm)).marg(mnames, n_mc) # goto L0
        raise RuntimeError(f"Unable to marginalize {self} for {mnames}. Check the model or try other factorizations.")

class DistrElem(Distr):
    def __init__(self, name: str, shape: tc.Size, device = None, **params):
        # for distributions whose parameter and random variable (or, one sample) have the same shape
        if device is None: device = Distr.default_device
        fnnames, fnvals, tennames, tenvals = [], [], [], []
        for pmname, pmval in params.items():
            if callable(pmval):
                fnnames.append(pmname)
                fnvals.append(pmval)
            else:
                tennames.append(pmname)
                tenvals.append(pmval)
        tenvals = tensorify(device, *tenvals)
        if shape is None:
            if Distr.has_name(name): shape = Distr.shape_var(name)
            elif tenvals: shape = tc.broadcast_tensors(*tenvals)[0].shape
            else: shape = tc.Size()
        parents = set()
        for fname, fval in zip(fnnames, fnvals):
            parents_inc = fargnames(fval)
            if parents_inc:
                parents |= parents_inc
                setattr(self, fname, fedic(fval))
            else:
                setattr( self, fname,
                        lambda conds, fval=fval: tensorify(device, fval())[0].expand(Distr.shape_bat(conds) + shape) )
        for tname, tval in zip(tennames, tenvals):
            setattr( self, tname,
                    lambda conds, tval=tval: tval.expand(Distr.shape_bat(conds) + shape) )
        super(DistrElem, self).__init__(names_shapes = {name: shape}, parents = parents)
        self._name, self._shape, self._device = name, shape, device

    @property
    def name(self): return self._name
    @property
    def shape(self): return self._shape
    @property
    def device(self): return self._device

    def mode(self, conds: edic=edic(), repar: bool=True) -> edic:
        # [shape_bat, shape_cond] -> [shape_bat, shape_var]
        raise NotImplementedError

