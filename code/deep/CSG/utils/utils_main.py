#!/usr/bin/env python3.6
import os, sys
import argparse
from copy import deepcopy
from functools import partial
import tqdm
import torch as tc
import torchvision as tv

sys.path.append('..')
from distr import edic
from arch import mlp, cnn
from methods import CNBBLoss, SemVar, SupVAE
from utils import Averager, unique_filename, boolstr, zip_longer # This imports from 'utils/__init__.py'

from dalib.modules.domain_discriminator import DomainDiscriminator
from dalib.modules.kernels import GaussianKernel
from dalib.adaptation.dann import DomainAdversarialLoss
from dalib.adaptation.cdan import ConditionalDomainAdversarialLoss
from dalib.adaptation.dan import MultipleKernelMaximumMeanDiscrepancy
from dalib.adaptation.mdd import MarginDisparityDiscrepancy

__author__ = "Chang Liu"
__email__ = "changliu@microsoft.com"
# tc.autograd.set_detect_anomaly(True)

MODES_OOD_NONGEN = {"discr", "cnbb"}
MODES_OOD_GEN = {"svae", "svgm", "svgm-ind"}
MODES_DA_NONGEN = {"dann", "cdan", "dan", "mdd", "bnm"}
MODES_DA_GEN = {"svae-da", "svgm-da", "svae-da2", "svgm-da2"}

MODES_OOD = MODES_OOD_NONGEN | MODES_OOD_GEN
MODES_DA = MODES_DA_NONGEN | MODES_DA_GEN
MODES_GEN = MODES_OOD_GEN | MODES_DA_GEN

MODES_TWIST = {"svgm-ind", "svae-da", "svgm-da"}

# Init models
def auto_load(dc_vars, names, ckpt):
    if ckpt:
        if type(names) is str: names = [names]
        for name in names:
            model = dc_vars[name]
            model.load_state_dict(ckpt[name+'_state_dict'])
            if hasattr(model, 'eval'): model.eval()

def get_frame(discr, gen, dc_vars, device = None, discr_src = None):
    if type(dc_vars) is not edic: dc_vars = edic(dc_vars)
    shape_x = dc_vars['shape_x'] if 'shape_x' in dc_vars else (dc_vars['dim_x'],)
    shape_s = discr.shape_s if hasattr(discr, "shape_s") else (dc_vars['dim_s'],)
    shape_v = discr.shape_v if hasattr(discr, "shape_v") else (dc_vars['dim_v'],)
    std_v1x = discr.std_v1x if hasattr(discr, "std_v1x") else dc_vars['qstd_v']
    std_s1vx = discr.std_s1vx if hasattr(discr, "std_s1vx") else dc_vars['qstd_s']
    std_s1x = discr.std_s1x if hasattr(discr, "std_s1x") else dc_vars['qstd_s']
    mode = dc_vars['mode']

    if mode.startswith("svgm"):
        q_args_stem = (discr.v1x, std_v1x, discr.s1vx, std_s1vx)
    elif mode.startswith("svae"):
        q_args_stem = (discr.s1x, std_s1x)
    else: return None
    if mode == "svgm-da2" and discr_src is not None:
        q_args = ( discr_src.v1x, discr_src.std_v1x if hasattr(discr_src, "std_v1x") else dc_vars['qstd_v'],
                discr_src.s1vx, discr_src.std_s1vx if hasattr(discr_src, "std_s1vx") else dc_vars['qstd_s'],
            ) + q_args_stem
    elif mode == "svae-da2" and discr_src is not None:
        q_args = ( discr_src.s1x, discr_src.std_s1x if hasattr(discr_src, "std_s1x") else dc_vars['qstd_s'],
            ) + q_args_stem
    elif mode in MODES_TWIST: # svgm-ind, svae-da, svgm-da
        q_args = (None,)*len(q_args_stem) + q_args_stem
    else: # svae, svgm
        q_args = q_args_stem + (None,)*len(q_args_stem)

    if mode.startswith("svgm"):
        frame = SemVar( shape_s, shape_v, shape_x, dc_vars['dim_y'],
                gen.x1sv, dc_vars['pstd_x'], discr.y1s, *q_args,
                *dc_vars.sublist(['mu_s', 'sig_s', 'mu_v', 'sig_v', 'corr_sv']),
                mode in MODES_DA, *dc_vars.sublist(['src_mvn_prior', 'tgt_mvn_prior']), device )
    elif mode.startswith("svae"):
        frame = SupVAE( shape_s, shape_x, dc_vars['dim_y'],
                gen.x1s, dc_vars['pstd_x'], discr.y1s, *q_args,
                *dc_vars.sublist(['mu_s', 'sig_s']),
                mode in MODES_DA, *dc_vars.sublist(['src_mvn_prior', 'tgt_mvn_prior']), device )
    return frame

def get_discr(archtype, dc_vars):
    if archtype == "mlp":
        discr = mlp.create_discr_from_json(
                *dc_vars.sublist([
                    'discrstru', 'dim_x', 'dim_y', 'actv',
                    'qstd_v', 'qstd_s', 'after_actv']),
                jsonfile=dc_vars['mlpstrufile']
            )
    elif archtype == "cnn":
        discr = cnn.CNNsvy1x(
                *dc_vars.sublist([
                    'discrstru', 'dim_btnk', 'dim_s', 'dim_y', 'dim_v',
                    'qstd_v', 'qstd_s']),
                *dc_vars.sublist([
                    'dims_bb2bn', 'dims_bn2s', 'dims_s2y',
                    'vbranch', 'dims_bn2v'], use_default = True, default = None)
            )
    else: raise ValueError(f"unknown `archtype` '{archtype}'")
    return discr

def get_gen(archtype, dc_vars, discr):
    if dc_vars['mode'].startswith("svgm"):
        if archtype == "mlp":
            gen = mlp.create_gen_from_json(
                    "MLPx1sv", discr, dc_vars['genstru'], jsonfile=dc_vars['mlpstrufile'] )
        elif archtype == "cnn":
            gen = cnn.CNNx1sv(
                    dc_vars['shape_x'][-1], *dc_vars.sublist(['dim_s', 'dim_v', 'dim_feat', 'genstru']) )
    elif dc_vars['mode'].startswith("svae"):
        if archtype == "mlp":
            gen = mlp.create_gen_from_json(
                    "MLPx1s", discr, dc_vars['genstru'], jsonfile=dc_vars['mlpstrufile'] )
        elif archtype == "cnn":
            gen = cnn.CNNx1s(
                    dc_vars['shape_x'][-1], *dc_vars.sublist(['dim_s', 'dim_feat', 'genstru']) )
    return gen

def get_models(archtype, dc_vars, ckpt = None, device = None):
    if type(dc_vars) is not edic: dc_vars = edic(dc_vars)
    discr = get_discr(archtype, dc_vars)
    if ckpt is not None: auto_load(locals(), 'discr', ckpt)
    discr.to(device)
    if dc_vars['mode'] in MODES_GEN:
        gen = get_gen(archtype, dc_vars, discr)
        if ckpt is not None: auto_load(locals(), 'gen', ckpt)
        gen.to(device)
        if dc_vars['mode'].endswith("-da2"):
            discr_src = get_discr(archtype, dc_vars)
            if ckpt is not None: auto_load(locals(), 'discr_src', ckpt)
            discr_src.to(device)
            frame = get_frame(discr, gen, dc_vars, device, discr_src)
            if ckpt is not None: auto_load(locals(), 'frame', ckpt)
            return discr, gen, frame, discr_src
        else:
            frame = get_frame(discr, gen, dc_vars, device)
            if ckpt is not None: auto_load(locals(), 'frame', ckpt)
            return discr, gen, frame
    else: return discr, None, None

def load_ckpt(filename: str, loadmodel: bool=False, device: tc.device=None, archtype: str="mlp", map_location: tc.device=None):
    ckpt = tc.load(filename, map_location)
    if loadmodel:
        return (ckpt,) + get_models(archtype, ckpt, ckpt, device)
    else: return ckpt

# Built methods
def get_ce_or_bce_loss(discr, dim_y: int, reduction: str="mean"):
    if dim_y == 1:
        celossobj = tc.nn.BCEWithLogitsLoss(reduction=reduction)
        celossfn = lambda x, y: celossobj(discr(x), y.float())
    else:
        celossobj = tc.nn.CrossEntropyLoss(reduction=reduction)
        celossfn = lambda x, y: celossobj(discr(x), y)
    return celossobj, celossfn

def add_ce_loss(lossobj, celossfn, ag):
    shrink_sup = ShrinkRatio(w_iter=ag.wsup_wdatum*ag.n_bat, decay_rate=ag.wsup_expo)
    def lossfn(*x_y_maybext_niter):
        loss = ag.wgen * lossobj(*x_y_maybext_niter[:-1])
        if ag.wsup:
            loss += ag.wsup * shrink_sup(x_y_maybext_niter[-1]) * celossfn(*x_y_maybext_niter[:2])
        return loss
    return lossfn

def ood_methods(discr, frame, ag, dim_y, cnbb_actv):
    if ag.mode not in MODES_GEN:
        if ag.mode == "discr":
            lossfn = get_ce_or_bce_loss(discr, dim_y, ag.reduction)[1]
        elif ag.mode == "cnbb":
            lossfn = CNBBLoss(discr.s1x, cnbb_actv, discr.forward, dim_y, ag.reg_w, ag.reg_s, ag.lr_w, ag.n_iter_w)
    else: # should be in MODES_GEN
        celossfn = get_ce_or_bce_loss( partial(frame.logit_y1x_src, n_mc_q=ag.n_mc_q)
                if ag.mode in MODES_TWIST and ag.true_sup else discr,
            dim_y, ag.reduction )[1]
        lossobj = frame.get_lossfn(ag.n_mc_q, ag.reduction, "defl", wlogpi=ag.wlogpi/ag.wgen)
        lossfn = add_ce_loss(lossobj, celossfn, ag)
    return lossfn

def da_methods(discr, frame, ag, dim_x, dim_y, device, ckpt, discr_src = None):
    if ag.mode not in MODES_GEN:
        celossobj, celossfn = get_ce_or_bce_loss(discr, dim_y, ag.reduction)
        if ag.mode == "dann":
            domdisc = DomainDiscriminator(in_feature=discr.dim_s, hidden_size=ag.domdisc_dimh).to(device)
            dalossobj = DomainAdversarialLoss(domdisc, reduction=ag.reduction).to(device)
            auto_load(locals(), ['domdisc', 'dalossobj'], ckpt)
            def lossfn(x, y, xt):
                logit, feat = discr.ys1x(x)
                featt = discr.s1x(xt)
                if feat.shape[0] < featt.shape[0]: featt = featt[:feat.shape[0]]
                elif feat.shape[0] > featt.shape[0]: feat = feat[:featt.shape[0]]
                return celossobj(logit, y.float() if dim_y == 1 else y
                        ) + ag.wda * dalossobj(feat, featt)
        elif ag.mode == "cdan":
            # In both randomized and not randomized versions, the code has problems.
            # For randomized, the dim_s*num_cls is fed to domdisc.
            # For not rand, `tc.mm` receives the wrong input order in `RandomizedMultiLinearMap.forward()`
            num_classes = (2 if dim_y == 1 else dim_y)
            domdisc = DomainDiscriminator(in_feature = discr.dim_s * (1 if ag.cdan_rand else num_classes), # confusing `in_feature`
                    hidden_size=ag.domdisc_dimh).to(device)
            dalossobj = ConditionalDomainAdversarialLoss(domdisc, reduction=ag.reduction, randomized=ag.cdan_rand,
                    num_classes=num_classes, features_dim=discr.dim_s, randomized_dim=discr.dim_s).to(device)
            auto_load(locals(), ['domdisc', 'dalossobj'], ckpt)
            def lossfn(x, y, xt):
                logit, feat = discr.ys1x(x)
                logitt, featt = discr.ys1x(xt)
                logit_stack = tc.stack([tc.zeros_like(logit), logit], dim=-1) if dim_y == 1 else logit
                logitt_stack = tc.stack([tc.zeros_like(logitt), logitt], dim=-1) if dim_y == 1 else logitt
                return celossobj(logit, y.float() if dim_y == 1 else y
                        ) + ag.wda * dalossobj(logit_stack, feat, logitt_stack, featt)
        elif ag.mode == "dan":
            domdisc = None
            dalossobj = MultipleKernelMaximumMeanDiscrepancy(
                    [GaussianKernel(alpha=alpha) for alpha in ag.ker_alphas] ).to(device)
            def lossfn(x, y, xt):
                logit, feat = discr.ys1x(x)
                featt = discr.s1x(xt)
                if feat.shape[0] < featt.shape[0]: featt = featt[:feat.shape[0]]
                elif feat.shape[0] > featt.shape[0]: feat = feat[:featt.shape[0]]
                return celossobj(logit, y.float() if dim_y == 1 else y
                        ) + ag.wda * dalossobj(feat, featt)
        elif ag.mode == "mdd":
            num_classes = (2 if dim_y == 1 else dim_y)
            domdisc = mlp.MLP([dim_x, ag.domdisc_dimh, ag.domdisc_dimh, num_classes]).to(device) # actually not domain discriminator but an auxiliary (adversarial) classifier
            dalossobj = MarginDisparityDiscrepancy(margin=ag.mdd_margin, reduction=ag.reduction).to(device)
            auto_load(locals(), ['domdisc', 'dalossobj'], ckpt)
            def lossfn(x, y, xt):
                logit, logitt = discr(x), discr(xt)
                logit_adv, logitt_adv = domdisc(x.reshape(-1,dim_x)), domdisc(xt.reshape(-1,dim_x))
                logit_stack = tc.stack([tc.zeros_like(logit), logit], dim=-1) if dim_y == 1 else logit
                logitt_stack = tc.stack([tc.zeros_like(logitt), logitt], dim=-1) if dim_y == 1 else logitt
                return celossobj(logit, y.float() if dim_y == 1 else y
                        ) + ag.wda * dalossobj(logit_stack, logit_adv, logitt_stack, logitt_adv)
        elif ag.mode == "bnm":
            domdisc = None
            dalossobj = None
            def lossfn(x, y, xt):
                logit, logitt = discr(x), discr(xt)
                logitt_stack = tc.stack([tc.zeros_like(logitt), logitt], dim=-1) if dim_y == 1 else logitt
                softmax_tgt = logitt_stack.softmax(dim=1)
                _, s_tgt, _ = tc.svd(softmax_tgt)
                # if config["method"]=="BNM":
                transfer_loss = -tc.mean(s_tgt)
                # elif config["method"]=="BFM":
                #     transfer_loss = -tc.sqrt(tc.sum(s_tgt*s_tgt)/s_tgt.shape[0])
                # elif config["method"]=="ENT":
                #     transfer_loss = -tc.mean(tc.sum(softmax_tgt*tc.log(softmax_tgt+1e-8),dim=1))/tc.log(softmax_tgt.shape[1])
                return celossobj(logit, y.float() if dim_y == 1 else y
                        ) + ag.wda * transfer_loss
        else: pass
        for obj in [dalossobj, domdisc]:
            if obj is not None: obj.train()
    else:
        if ag.mode.endswith("-da2") and discr_src is not None: true_discr = discr_src
        elif ag.mode in MODES_TWIST and ag.true_sup: true_discr = partial(frame.logit_y1x_src, n_mc_q=ag.n_mc_q)
        else: true_discr = discr
        celossfn = get_ce_or_bce_loss(true_discr, dim_y, ag.reduction)[1]
        lossobj = frame.get_lossfn(ag.n_mc_q, ag.reduction, "defl", weight_da=ag.wda/ag.wgen, wlogpi=ag.wlogpi/ag.wgen)
        lossfn = add_ce_loss(lossobj, celossfn, ag)
        domdisc, dalossobj = None, None
    return lossfn, domdisc, dalossobj

# Training utilities
class ParamGroupsCollector:
    def __init__(self, lr):
        self.reset(lr)

    def reset(self, lr):
        self.lr = lr
        self.param_groups = []

    def collect_params(self, *models):
        for model in models:
            if hasattr(model, 'parameter_groups'):
                groups_inc = list(model.parameter_groups())
                for grp in groups_inc:
                    if 'lr_ratio' in grp:
                        grp['lr'] = self.lr * grp['lr_ratio']
                    elif 'lr' not in grp: # Do not overwrite existing lr assignments
                        grp['lr'] = self.lr
                self.param_groups += groups_inc
            else:
                self.param_groups += [
                        {'params': model.parameters(), 'lr': self.lr} ]

class ShrinkRatio:
    def __init__(self, w_iter, decay_rate):
        self.w_iter = w_iter
        self.decay_rate = decay_rate

    def __call__(self, n_iter):
        return (1 + self.w_iter * n_iter) ** (-self.decay_rate)

# Test and save utilities
def acc_with_logits(model: tc.nn.Module, x: tc.Tensor, y: tc.LongTensor, is_binary: bool, u = None, use_u = False) -> float:
    with tc.no_grad(): logits = model(x) if not use_u else model(x,u)
    ypred = (logits > 0).long() if is_binary else logits.argmax(dim=-1)
    return (ypred == y).float().mean().item()

def evaluate_acc(discr, input_loader, is_binary, device):
    avgr = Averager()
    for x, y in input_loader:
        x, y = x.to(device), y.to(device)
        avgr.update(acc_with_logits(discr, x, y, is_binary), nrep = len(y))
    return avgr.avg

def evaluate_llhx(frame, input_loader, n_marg_llh, use_q_llh, mode, device):
    avgr = Averager()
    for x, y in input_loader:
        x = x.to(device)
        avgr.update(frame.llh(x, None, n_marg_llh, use_q_llh, mode), nrep = len(x))
    return avgr.avg

class ResultsContainer:
    def __init__(self, len_ts, frame, ag, is_binary, device, ckpt = None):
        for k,v in locals().items():
            if k not in {"self", "ckpt"}: setattr(self, k, v)
        self.dc = dict( epochs = [], losses = [],
                accs_tr = [], llhs_tr = [], accs_val = [], llhs_val = [] )
        if len_ts:
            ls_empty = [[] for _ in range(len_ts)]
            self.dc.update( ls_accs_ts = ls_empty, ls_llhs_ts = deepcopy(ls_empty) )
        else:
            self.dc.update( accs_ts = [], llhs_ts = [] )
        if ckpt is not None:
            for k in self.dc.keys():
                if not k.startswith('ls_'): self.dc[k] = ckpt[k]
                else: self.dc[k] = [ckpt[k[3:]]]

    def update(self, *, epk, loss):
        self.dc['epochs'].append(epk)
        self.dc['losses'].append(loss)

    def evaluate(self, discr, dname, dpart, loader, llh_mode, i = None):
        acc = evaluate_acc(discr, loader, self.is_binary, self.device)
        if i is None: self.dc['accs_'+dpart].append(acc)
        else: self.dc['ls_accs_'+dpart][i].append(acc)
        print(f"On {dname}, acc: {acc:.3f}", end="", flush=True)
        if self.frame is not None and self.ag.eval_llh:
            llh = evaluate_llhx(
                    self.frame, loader, self.ag.n_marg_llh, self.ag.use_q_llh, llh_mode, self.device)
            if i is None: self.dc['llhs_'+dpart].append(llh)
            else: self.dc['ls_llhs_'+dpart][i].append(llh)
            print(f", llhs: {llh:.3e}. ", end="", flush=True)
        else: print(". ", end="", flush=True)

    def summary(self, dname, dpart, i = None):
        if i is None:
            accs = self.dc['accs_'+dpart]
            llhs = self.dc['llhs_'+dpart]
        else:
            accs = self.dc['ls_accs_'+dpart][i]
            llhs = self.dc['ls_llhs_'+dpart][i]
        acc_fin = tc.tensor(accs[-self.ag.avglast:]).mean().item()
        llh_fin = tc.tensor(llhs[-self.ag.avglast:]).mean().item() if llhs[-self.ag.avglast:] else None
        acc_max = tc.tensor(accs).max().item()
        llh_max = tc.tensor(llhs).max().item() if llhs else None
        print(f"On {dname}, final acc: {acc_fin:.3f}" + (
                f", llh: {llh_fin:.3f}" if llh_fin else ""
            ) + f", max acc: {acc_max:.3f}" + (
                f", llh: {llh_max:.3f}" if llh_max else "") + ".")

def dc_state_dict(dc_vars, *name_list):
    return {name+"_state_dict" : dc_vars[name].state_dict()
            for name in name_list if hasattr(dc_vars[name], 'state_dict')}

# Main
def is_ood(mode):
    return mode in MODES_OOD

def process_continue_run(ag):
    # Process if continue running
    if ag.init_model not in {"rand", "fix"}: # continue running
        ckpt = load_ckpt(ag.init_model, loadmodel=False)
        if ag.mode != ckpt['mode']: raise RuntimeError("mode not match")
        for k in vars(ag):
            if k not in {"testdoms", "n_epk", "gpu"}: # use the new final number of epochs
                setattr(ag, k, ckpt[k])
        ag.testdoms = [ckpt['testdom']] # overwrite the input `testdoms`
    else: ckpt = None
    return ag, ckpt

def main_stem(ag, ckpt, archtype, shape_x, dim_y,
        tr_src_loader, val_src_loader,
        ls_ts_tgt_loader = None, # for ood
        tr_tgt_loader = None, ts_tgt_loader = None, testdom = None # for da
    ):
    print(ag)
    print_infrstru_info()
    IS_OOD = is_ood(ag.mode)
    device = tc.device("cuda:"+str(ag.gpu) if tc.cuda.is_available() else "cpu")

    # Datasets
    dim_x = tc.tensor(shape_x).prod().item()
    if IS_OOD: n_per_epk = len(tr_src_loader)
    else: n_per_epk = max(len(tr_src_loader), len(tr_tgt_loader))

    # Models
    res = get_models(archtype, edic(locals()) | vars(ag), ckpt, device)
    if ag.mode.endswith("-da2"):
        discr, gen, frame, discr_src = res
        discr_src.train()
    else:
        discr, gen, frame = res
    discr.train()
    if gen is not None: gen.train()

    # Methods and Losses
    if IS_OOD:
        lossfn = ood_methods(discr, frame, ag, dim_y, cnbb_actv="Sigmoid") # Actually the activation is ReLU, but there is no `is_treat` rule for ReLU in CNBB.
        domdisc = None
    else:
        lossfn, domdisc, dalossobj = da_methods(discr, frame, ag, dim_x, dim_y, device, ckpt,
                discr_src if ag.mode.endswith("-da2") else None)

    # Optimizer
    pgc = ParamGroupsCollector(ag.lr)
    pgc.collect_params(discr)
    if ag.mode.endswith("-da2"): pgc.collect_params(discr_src)
    if gen is not None: pgc.collect_params(gen, frame)
    if domdisc is not None: pgc.collect_params(domdisc)
    if ag.optim == "SGD":
        opt = getattr(tc.optim, ag.optim)(pgc.param_groups, weight_decay=ag.wl2, momentum=ag.momentum, nesterov=ag.nesterov)
        shrink_opt = ShrinkRatio(w_iter=ag.lr_wdatum*ag.n_bat, decay_rate=ag.lr_expo)
        lrsched = tc.optim.lr_scheduler.LambdaLR(opt, shrink_opt)
        auto_load(locals(), 'lrsched', ckpt)
    else: opt = getattr(tc.optim, ag.optim)(pgc.param_groups, weight_decay=ag.wl2)
    auto_load(locals(), 'opt', ckpt)

    # Training
    epk0 = 1; i_bat0 = 1
    if ckpt is not None:
        epk0 = ckpt['epochs'][-1] + 1 if ckpt['epochs'] else 1
        i_bat0 = ckpt['i_bat']
    res = ResultsContainer( len(ag.testdoms) if IS_OOD else None,
            frame, ag, dim_y==1, device, ckpt )
    print(f"Run in mode '{ag.mode}' for {ag.n_epk:3d} epochs:")
    try:
        for epk in range(epk0, ag.n_epk+1):
            pbar = tqdm.tqdm(total=n_per_epk, desc=f"Train epoch = {epk:3d}", ncols=80, leave=False)
            for i_bat, data_bat in enumerate(
                    tr_src_loader if IS_OOD else zip_longer(tr_src_loader, tr_tgt_loader), start=1):
                if i_bat < i_bat0: continue
                if IS_OOD:
                    x, y = data_bat
                    data_args = (x.to(device), y.to(device))
                else:
                    (x, y), (xt, yt) = data_bat
                    data_args = (x.to(device), y.to(device), xt.to(device))
                opt.zero_grad()
                if ag.mode in MODES_GEN:
                    n_iter_tot = (epk-1)*n_per_epk + i_bat-1
                    loss = lossfn(*data_args, n_iter_tot)
                else: loss = lossfn(*data_args)
                loss.backward()
                opt.step()
                if ag.optim == "SGD": lrsched.step()
                pbar.update(1)
            # end for
            pbar.close()
            i_bat = 1; i_bat0 = 1

            if epk % ag.eval_interval == 0:
                res.update(epk=epk, loss=loss.item())
                print(f"Mode '{ag.mode}': Epoch {epk:.1f}, Loss = {loss:.3e},")
                discr.eval()
                if ag.mode.endswith("-da2"): discr_src.eval(); true_discr = discr_src
                elif ag.mode in MODES_TWIST and ag.true_sup_val: true_discr = partial(frame.logit_y1x_src, n_mc_q=ag.n_mc_q)
                else: true_discr = discr
                res.evaluate(true_discr, "val "+str(ag.traindom), 'val', val_src_loader, 'src')
                if IS_OOD:
                    for i, (testdom, ts_tgt_loader) in enumerate(zip(ag.testdoms, ls_ts_tgt_loader)):
                        res.evaluate(discr, "test "+str(testdom), 'ts', ts_tgt_loader, 'tgt', i)
                else:
                    res.evaluate(discr, "test "+str(testdom), 'ts', ts_tgt_loader, 'tgt')
                print()
                discr.train()
                if ag.mode.endswith("-da2"): discr_src.train()
        # end for
    except (KeyboardInterrupt, SystemExit): pass
    res.summary("val "+str(ag.traindom), 'val')
    if IS_OOD:
        for i, testdom in enumerate(ag.testdoms): res.summary("test "+str(testdom), 'ts', i)
    else:
        res.summary("test "+str(testdom), 'ts')

    if not ag.no_save:
        dirname = "ckpt_" + ag.mode + "/"
        os.makedirs(dirname, exist_ok=True)
        for i, testdom in enumerate(
                ag.testdoms if IS_OOD else [testdom]
            ):
            filename = unique_filename(
                    dirname + ("ood" if IS_OOD else "da"), ".pt", n_digits = 3
                ) if ckpt is None else ckpt['filename']
            dc_vars = edic(locals()).sub([
                    'dirname', 'filename', 'testdom',
                    'shape_x', 'dim_x', 'dim_y', 'i_bat'
                ]) | ( edic(vars(ag)) - {'testdoms'}
                ) | dc_state_dict(locals(), "discr", "opt")
            if ag.mode.endswith("-da2"):
                dc_vars.update( dc_state_dict(locals(), "discr_src") )
            if ag.optim == "SGD":
                dc_vars.update( dc_state_dict(locals(), "lrsched") )
            if ag.mode in MODES_GEN:
                dc_vars.update( dc_state_dict(locals(), "gen", "frame") )
            elif ag.mode in {"dann", "cdan", "dan", "mdd"}:
                dc_vars.update( dc_state_dict(locals(), "domdisc", "dalossobj") )
            else: pass
            if IS_OOD:
                dc_vars.update( edic({
                        k:v for k,v in res.dc.items() if not k.startswith('ls_')
                    }) | { k[3:]:v[i] for k,v in res.dc.items() if k.startswith('ls_')
                    })
            else:
                dc_vars.update(res.dc)
            tc.save(dc_vars, filename)
            print(f"checkpoint saved to '{filename}'.")

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type = str, choices = MODES_OOD | MODES_DA)

    # Data
    parser.add_argument("--tr_val_split", type = float, default = .8)
    parser.add_argument("--n_bat", type = int, default = 32)

    # Model
    parser.add_argument("--init_model", type = str, default = "rand") # or a model file name to continue running
    parser.add_argument("--discrstru", type = str)
    parser.add_argument("--genstru", type = str)

    # Process
    parser.add_argument("--n_epk", type = int, default = 800)
    parser.add_argument("--eval_interval", type = int, default = 5)
    parser.add_argument("--avglast", type = int, default = 4)
    parser.add_argument("-ns", "--no_save", action = "store_true")

    # Optimization
    parser.add_argument("--optim", type = str)
    parser.add_argument("--lr", type = float)
    parser.add_argument("--wl2", type = float)
    parser.add_argument("--reduction", type = str, default = "mean")
    parser.add_argument("--momentum", type = float, default = 0.) # only when "lr" is "SGD"
    parser.add_argument("--nesterov", type = boolstr, default = False) # only when "lr" is "SGD"
    parser.add_argument("--lr_expo", type = float, default = .75) # only when "lr" is "SGD"
    parser.add_argument("--lr_wdatum", type = float, default = 6.25e-6) # only when "lr" is "SGD"

    # For generative models only
    parser.add_argument("--mu_s", type = float, default = 0.)
    parser.add_argument("--sig_s", type = float, default = 1.)
    parser.add_argument("--mu_v", type = float, default = 0.) # for svgm only
    parser.add_argument("--sig_v", type = float, default = 1.) # for svgm only
    parser.add_argument("--corr_sv", type = float, default = .7) # for svgm only
    parser.add_argument("--pstd_x", type = float, default = 3e-2)
    parser.add_argument("--qstd_s", type = float, default = 3e-2)
    parser.add_argument("--qstd_v", type = float, default = 3e-2) # for svgm only
    parser.add_argument("--wgen", type = float, default = 1.)
    parser.add_argument("--wsup", type = float, default = 1.)
    parser.add_argument("--wsup_expo", type = float, default = 0.) # only when "wsup" is not 0
    parser.add_argument("--wsup_wdatum", type = float, default = 6.25e-6) # only when "wsup" and "wsup_expo" are not 0
    parser.add_argument("--wlogpi", type = float, default = None)
    parser.add_argument("--n_mc_q", type = int, default = 0)
    parser.add_argument("--eval_llh", action = "store_true")
    parser.add_argument("--use_q_llh", type = boolstr, default = True)
    parser.add_argument("--n_marg_llh", type = int, default = 16)
    parser.add_argument("--true_sup", type = boolstr, default = False, help = "for 'svgm-ind', 'svgm-da', 'svae-da' only")
    parser.add_argument("--true_sup_val", type = boolstr, default = True, help = "for 'svgm-ind', 'svgm-da', 'svae-da' only")
    ## For OOD
    parser.add_argument("--mvn_prior", type = boolstr, default = False)
    ## For DA
    parser.add_argument("--src_mvn_prior", type = boolstr, default = False)
    parser.add_argument("--tgt_mvn_prior", type = boolstr, default = False)

    # For OOD
    ## For cnbb only
    parser.add_argument("--reg_w", type = float, default = 1e-4)
    parser.add_argument("--reg_s", type = float, default = 3e-6)
    parser.add_argument("--lr_w", type = float, default = 1e-3)
    parser.add_argument("--n_iter_w", type = int, default = 4)

    # For DA
    parser.add_argument("--wda", type = float, default = .25)
    ## For {dann, cdan, dan, mdd} only
    parser.add_argument("--domdisc_dimh", type = int, default = 1024) # for {dann, cdan, mdd} only
    parser.add_argument("--cdan_rand", type = boolstr, default = False) # for cdan only
    parser.add_argument("--ker_alphas", type = float, nargs = '+', default = [.5, 1., 2.]) # for dan only
    parser.add_argument("--mdd_margin", type = float, default = 4.) # for mdd only
    parser.add_argument("--gpu", type=int, default = 0)

    return parser

def print_infrstru_info():
    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(tc.__version__))
    print("\tTorchvision: {}".format(tv.__version__))
    print("\tCUDA: {}".format(tc.version.cuda))
    print("\tCUDNN: {}".format(tc.backends.cudnn.version()))
    # print("\tNumPy: {}".format(np.__version__))
    # print("\tPIL: {}".format(PIL.__version__))

