#!/usr/bin/env python3.6
import os
import warnings
from itertools import product, chain
import math
import torch as tc
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

__author__ = "Chang Liu"
__email__ = "changliu@microsoft.com"

# General Utilities
def unique_filename(prefix: str="", suffix: str="", n_digits: int=2, count_start: int=0) -> str:
    fmt = "{:0" + str(n_digits) + "d}"
    if prefix and prefix[-1] not in {"/", "\\"}: prefix += "_"
    while True:
        filename = prefix + fmt.format(count_start) + suffix
        if not os.path.exists(filename): return filename
        else: count_start += 1

class Averager:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self._val = 0
        self._avg = 0
        self._sum = 0
        self._count = 0

    def update(self, val, nrep = 1):
        self._val = val
        self._sum += val * nrep
        self._count += nrep
        self._avg = self._sum / self._count

    @property
    def val(self): return self._val
    @property
    def avg(self): return self._avg
    @property
    def sum(self): return self._sum
    @property
    def count(self): return self._count

def repeat_iterator(itr, n_repeat):
    # The built-in `itertools.cycle` stores all results over `itr` and does not initialize `itr` again.
    return chain.from_iterable([itr] * n_repeat)

class RepeatIterator:
    def __init__(self, itr, n_repeat):
        self.itr = itr
        self.n_repeat = n_repeat
        self.len = len(itr) * n_repeat

    def __iter__(self):
        return chain.from_iterable([self.itr] * self.n_repeat)

    def __len__(self): return self.len

def zip_longer(itr1, itr2):
    len_ratio = len(itr1) / len(itr2)
    if len_ratio > 1:
        return zip(itr1, repeat_iterator(itr2, math.ceil(len_ratio)))
    elif len_ratio < 1:
        return zip(repeat_iterator(itr1, math.ceil(1/len_ratio)), itr2)
    else: return zip(itr1, itr2)

def zip_longest(*itrs):
    itr_longest = max(itrs, key=len)
    len_longest = len(itr_longest)
    return zip(*[itr if len(itr) == len_longest
            else repeat_iterator(itr, math.ceil(len_longest / len(itr)))
        for itr in itrs])

class ZipLongest:
    def __init__(self, *itrs):
        self.itrs = itrs
        self.itr_longest = max(itrs, key=len)
        self.len = len(self.itr_longest)

    def __iter__(self):
        return zip(*[itr if len(itr) == self.len
                else repeat_iterator(itr, math.ceil(self.len / len(itr)))
            for itr in self.itrs])

    def __len__(self): return self.len


class CyclicLoader:
    def __init__(self, datalist: list, shuffle: bool=True, cycle: int=None):
        self.len = len(datalist[0])
        if shuffle:
            ids = np.random.permutation(self.len)
            self.datalist = [data[ids] for data in datalist]
        else: self.datalist = datalist
        self.cycle = self.len if cycle is None else cycle
        self.head = 0
    def iter(self):
        self.head = 0
        return self
    def next(self, n: int) -> tuple:
        ids = [i % self.cycle for i in range(self.head, self.head + n)]
        self.head = (self.head + n) % self.cycle
        return tuple(data[ids] for data in self.datalist)
    def back(self, n: int):
        self.head = (self.head - n) % self.cycle
        return self

def boolstr(s: str) -> bool:
    # for argparse argument of type bool
    if isinstance(s, str):
        true_strings = {'1', 'true', 'True', 'T', 'yes', 'Yes', 'Y'}
        false_strings = {'0', 'false', 'False', 'F', 'no', 'No', 'N'}
        if s not in true_strings | false_strings:
            raise ValueError('Not a valid boolean string')
        return s in true_strings
    else:
        return bool(s)

## For lists/tuples
def getlist(ls: list, ids: list) -> list:
    return [ls[i] for i in ids]

def interleave(*lists) -> list:
    return [v for row in zip(*lists) for v in row]

def flatten(ls: list, depth: int=None) -> list:
    i = 0
    while (depth is None or i < depth) and bool(ls) and all(
            type(row) is list or type(row) is tuple for row in ls):
        ls = [v for row in ls if bool(row) for v in row]
        i += 1
    return ls

class SlicesAt:
    def __init__(self, axis: int, ndim: int):
        if ndim <= 0: raise ValueError(f"`ndim` (which is {ndim}) should be a positive integer")
        if not -ndim <= axis < ndim: raise ValueError(f"`axis` (which is {axis}) should be within [{-ndim}, {ndim})")
        self._axis, self._ndim = axis % ndim, ndim

    def __getitem__(self, idx):
        slices = [slice(None)] * self._ndim
        slices[self._axis] = idx
        return tuple(slices)

# For numpy/torch
def moving_average_slim(arr: np.ndarray, n_win: int=2, axis: int=-1) -> np.ndarray:
    # `(n_win-1)` shorter. Good for any positive `n_win`. Good if `arr` is empty in `axis`
    if n_win <= 0: raise ValueError(f"nonpositive `n_win` {n_win} not allowed")
    slc = SlicesAt(axis, arr.ndim)
    concatfn = tc.cat if type(arr) is tc.Tensor else np.concatenate
    cum = arr.cumsum(axis) # , dtype=float)
    return concatfn([ cum[slc[n_win-1:n_win]], cum[slc[n_win:]]-cum[slc[:-n_win]] ], axis) / float(n_win)

def moving_average_full(arr: np.ndarray, n_win: int=2, axis: int=-1) -> np.ndarray:
    # Same length as `arr`. Good for any positive `n_win`. Good if `arr` is empty in `axis`
    if n_win <= 0: raise ValueError(f"nonpositive `n_win` {n_win} not allowed")
    slc = SlicesAt(axis, arr.ndim)
    concatfn = tc.cat if type(arr) is tc.Tensor else np.concatenate
    cum = arr.cumsum(axis) # , dtype=float)
    stem = concatfn([ cum[slc[n_win-1:n_win]], cum[slc[n_win:]]-cum[slc[:-n_win]] ], axis) / float(n_win)
    length = arr.shape[axis]
    lwid = (n_win - 1) // 2
    rwid = n_win//2 + 1
    return concatfn([
            *[ cum[slc[j-1: j]] / float(j) for i in range(min(lwid, length)) for j in [min(i+rwid, length)] ],
            stem,
            *[ (cum[slc[-1:]] - cum[slc[i-lwid-1: i-lwid]] if i-lwid > 0 else cum[slc[-1:]]) / float(length-i+lwid)
                for i in range(max(length-rwid+1, lwid), length) ]
        ], axis)

def moving_average_full_checker(arr: np.ndarray, n_win: int=2, axis: int=-1) -> np.ndarray:
    # Same length as `arr`. Good for any positive `n_win`. Good if `arr` is empty in `axis`
    if n_win <= 0: raise ValueError(f"nonpositive `n_win` {n_win} not allowed")
    if arr.shape[axis] < 2: return arr
    slc = SlicesAt(axis, arr.ndim)
    concatfn = tc.cat if type(arr) is tc.Tensor else np.concatenate
    lwid = (n_win - 1) // 2
    rwid = n_win//2 + 1
    return concatfn([ arr[slc[max(0, i-lwid): (i+rwid)]].mean(axis, keepdims=True) for i in range(arr.shape[axis]) ], axis)

# Plotting Utilities
class Plotter:
    def __init__(self, var_xlab: dict, metr_ylab: dict, tab_items: list=None,
            check_var: bool=False, check_tab: bool=False, loader = tc.load):
        for var in var_xlab:
            if not var_xlab[var]: var_xlab[var] = var
        for metr in metr_ylab:
            if not metr_ylab[metr]: metr_ylab[metr] = metr
        self.var_xlab, self.metr_ylab = var_xlab, metr_ylab
        self.variables, self.metrics = list(var_xlab), list(metr_ylab)
        if tab_items is None: self.tab_items = []
        else: self.tab_items = tab_items
        self.check_var, self.check_tab = check_var, check_tab
        self.loader = loader
        self._plt_data, self._tab_data = [], []

    def _get_res(self, dataholder): # does not change `self`
        res_x = {var: [] for var in self.variables}
        res_ymean = {metr: np.array([]) for metr in self.metrics}
        res_ystd = {metr: np.array([]) for metr in self.metrics}
        res_tab = [None for item in self.tab_items]
        if type(dataholder) is not dict: # treated as a list of data file names
            resfiles = []
            for file in dataholder:
                if os.path.isfile(file): resfiles.append(file)
                else: warnings.warn(f"file '{file}' does not exist")
            dataholder = dict()
            for file in resfiles:
                ckp = self.loader(file)
                for name in self.metrics + self.variables + self.tab_items:
                    if name not in ckp: warnings.warn(f"metric or variable or item '{name}' not found in file '{file}'")
                    else:
                        if name not in dataholder: dataholder[name] = []
                        dataholder[name].append(ckp[name])
        for metr in self.metrics:
            if metr not in dataholder or not dataholder[metr]:
                warnings.warn(f"metric '{metr}' not found or empty")
                continue
            n_align = min((len(line) for line in dataholder[metr]), default=0)
            if n_align:
                vals = np.array([line[:n_align] for line in dataholder[metr]])
                res_ymean[metr] = vals.mean(0)
                res_ystd[metr] = vals.std(0)
        for var in self.variables:
            if var not in dataholder or not dataholder[var]:
                warnings.warn(f"variable '{var}' not found or empty")
                continue
            n_align = min((len(line) for line in dataholder[var]), default=0)
            if n_align:
                res_x[var] = dataholder[var][0]
                if self.check_var:
                    for line in dataholder[var][1:]:
                        if line != res_x[x]: raise RuntimeError(f"variable '{var}' not match")
        for i, item in enumerate(self.tab_items):
            if item not in dataholder or not dataholder[item]:
                warnings.warn(f"item '{item}' not found or empty")
                continue
            res_tab[i] = dataholder[item][0]
            if self.check_tab:
                for val in dataholder[item][1:]:
                    if val != res_tab[i]: raise RuntimeError(f"item '{item}' not match")
        return res_x, res_ymean, res_ystd, res_tab

    def load(self, *triplets):
        # each triplet = (legend, pltsty, [filename1, filename2]), or
        # (legend, pltsty, {var1: [val1_1, val1_2], metr2: [val2_1, val2_2, val2_3]})
        data = [(legend, pltsty, *self._get_res(dataholder)) for legend, pltsty, dataholder in triplets]
        self._plt_data += [entry[:-1] for entry in data]
        self._tab_data += [[entry[0]] + entry[-1] for entry in data]

    def clear(self):
        self._plt_data.clear()
        self._tab_data.clear()

    def plot(self, variables: list=None, metrics: list=None,
            var_xlim: dict=None, metr_ylim: dict=None,
            n_start: int=None, n_stop: int=None, n_step: int=None, n_win: int=1,
            plot_err: bool=True, ncol: int=None,
            fontsize: int=20, figheight: int=8, linewidth: int=4, alpha: float=.2, show_legend: bool=True):
        if variables is None: variables = self.variables
        if metrics is None: metrics = self.metrics
        if var_xlim is None: var_xlim = {}
        if metr_ylim is None: metr_ylim = {}
        slc = slice(n_start, n_stop, n_step)
        if ncol is None: ncol = max(2, len(variables))
        nfig = len(variables) * len(metrics)
        nrow = (nfig-1) // ncol + 1
        if nfig < ncol: ncol = nfig
        plt.rcParams.update({'font.size': fontsize})

        fig, axes0 = plt.subplots(nrow, ncol, figsize=(ncol*figheight, nrow*figheight))
        if nfig == 1: axes = [axes0]
        elif nrow > 1: axes = [ax for row in axes0 for ax in row][:nfig]
        else: axes = axes0[:nfig]
        for ax, (metr, var) in zip(axes, product(metrics, variables)):
            plotted = False
            for legend, pltsty, res_x, res_ymean, res_ystd in self._plt_data:
                y, std = res_ymean[metr], res_ystd[metr]
                x = res_x[var] if var is not None else list(range(min(len(y), len(std))))
                n_align = min(len(x), len(y), len(std))
                x, y, std = x[:n_align], y[:n_align], std[:n_align]
                if n_win > 1:
                    y = moving_average_full(y, n_win)
                    if plot_err: std = moving_average_full(std, n_win) # Not precise! std and averaging is not interchangeable, since sqrt(sum ^2) is not linear
                x, y, std = x[slc], y[slc], std[slc]
                if len(x):
                    if plot_err:
                        ax.fill_between(x, y-std, y+std, facecolor=pltsty[0], alpha=alpha, linewidth=0)
                    ax.plot(x, y, pltsty, label=legend, linewidth=linewidth)
                    plotted = True
            if show_legend and plotted: ax.legend()
            if var in var_xlim: ax.set_xlim(var_xlim[var])
            if metr in metr_ylim: ax.set_ylim(metr_ylim[metr])
            ax.set_xlabel(self.var_xlab[var] if var is not None else "index")
            ax.set_ylabel(self.metr_ylab[metr])
        return fig, axes0

    def inspect(self, metr: str, ids: list=None, var: str=None, vals: list=None,
            show_std: bool=True, **tbformat):
        if (ids is None) == (var is None and vals is None):
            raise ValueError("exactly one of `ids`, or `var` and `vals`, should be provided")
        if ids is not None:
            if not show_std:
                table = [[legend, *getlist(res_ymean[metr], ids)] for legend, pltsty, res_x, res_ymean, res_ystd in self._plt_data]
                print(tabulate(table, headers = ["indices"] + ids, **tbformat))
            else:
                table = [[legend, *interleave(getlist(res_ymean[metr], ids), getlist(res_ystd[metr], ids))]
                        for legend, pltsty, res_x, res_ymean, res_ystd in self._plt_data]
                print(tabulate(table, headers = ["indices"] + interleave(ids, ids), **tbformat))
        else:
            if not show_std:
                table = [[legend, *[res_ymean[metr][res_x[var].index(val)] for val in vals]]
                        for legend, pltsty, res_x, res_ymean, res_ystd in self._plt_data]
                print(tabulate(table, headers = [var] + vals, **tbformat))
            else:
                ids_list = [[res_x[var].index(val) for val in vals] for _, _, res_x, _, _ in self._plt_data]
                table = [[legend, *interleave(getlist(res_ymean[metr], ids), getlist(res_ystd[metr], ids))]
                        for ids, (legend, pltsty, res_x, res_ymean, res_ystd) in zip(ids_list, self._plt_data)]
                print(tabulate(table, headers = [var] + interleave(vals, vals), **tbformat))
        return table

    def tabulate(self, **tbformat):
        print(tabulate(self._tab_data, headers = ["legend"] + self.tab_items, **tbformat))

