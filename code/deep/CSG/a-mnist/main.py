#!/usr/bin/env python3.6
import warnings
import sys
import torch as tc
sys.path.append("..")
from utils.utils_main import main_stem, get_parser, is_ood, process_continue_run
from utils.utils import boolstr

__author__ = "Chang Liu"
__email__ = "changliu@microsoft.com"
# tc.autograd.set_detect_anomaly(True)

if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument("--data_root", type = str, default = "./data/MNIST/processed/")
    parser.add_argument("--traindom", type = str) # 12665 = 5923 (46.77%) + 6742
    parser.add_argument("--testdoms", type = str, nargs = '+') # 2115 = 980 (46.34%) + 1135
    parser.add_argument("--shuffle", type = boolstr, default = True)

    parser.add_argument("--mlpstrufile", type = str, default = "../arch/mlpstru.json")
    parser.add_argument("--actv", type = str, default = "Sigmoid")
    parser.add_argument("--after_actv", type = boolstr, default = True)

    parser.set_defaults(discrstru = "lite", genstru = None,
            n_bat = 128, n_epk = 100,
            mu_s = .5, mu_v = .5,
            pstd_x = 3e-2, qstd_s = 3e-2, qstd_v = 3e-2,
            optim = "RMSprop", lr = 1e-3, wl2 = 1e-5,
            momentum = 0., nesterov = False, lr_expo = .5, lr_wdatum = 6.25e-6, # only when "lr" is "SGD"
            wda = 1.,
            domdisc_dimh = 1024, # for {dann, cdan, mdd} only
            cdan_rand = False, # for cdan only
            ker_alphas = [.5, 1., 2.], # for dan only
            mdd_margin = 4. # for mdd only
        )
    ag = parser.parse_args()
    if ag.wlogpi is None: ag.wlogpi = ag.wgen
    ag, ckpt = process_continue_run(ag)
    IS_OOD = is_ood(ag.mode)

    archtype = "mlp"
    # Dataset
    src_x, src_y = tc.load(ag.data_root+ag.traindom) # `x` already tc.Tensor in range [0., 1.]
    dim_x = tc.tensor(src_x.shape[1:]).prod().item()
    src_x = src_x.reshape(-1, dim_x)
    shape_x = (dim_x,)
    dim_y = src_y.max().long().item() + 1
    if dim_y == 2: dim_y = 1
    ## tr-val split
    len_src = len(src_x)
    assert len_src == len(src_y)
    ids_src = tc.randperm(len_src)
    len_tr_src = int(len_src * ag.tr_val_split)
    ids_tr_src, ids_val_src = ids_src[:len_tr_src], ids_src[len_tr_src:]
    tr_src_x, tr_src_y = src_x[ids_tr_src], src_y[ids_tr_src]
    val_src_x, val_src_y = src_x[ids_val_src], src_y[ids_val_src]
    ## dataloaders
    kwargs = {'num_workers': 4, 'pin_memory': True}
    tr_src_loader = tc.utils.data.DataLoader(
            tc.utils.data.TensorDataset(tr_src_x, tr_src_y),
            ag.n_bat, ag.shuffle, **kwargs )
    val_src_loader = tc.utils.data.DataLoader(
            tc.utils.data.TensorDataset(val_src_x, val_src_y),
            ag.n_bat, ag.shuffle, **kwargs )
    ## tgt (ts) domain
    ls_tgt_xy_raw = [tc.load(ag.data_root+testdom) for testdom in ag.testdoms] # (x,y). `x` already tc.Tensor in range [0., 1.]
    ls_tgt_xy = [(xy[0].reshape(len(xy[0]), dim_x), xy[1]) for xy in ls_tgt_xy_raw]
    ls_ts_tgt_loader = [tc.utils.data.DataLoader(
            tc.utils.data.TensorDataset(*xy),
            ag.n_bat, ag.shuffle, **kwargs )
        for xy in ls_tgt_xy]
    if not IS_OOD:
        ls_tr_tgt_loader = [tc.utils.data.DataLoader(
                tc.utils.data.TensorDataset(*xy),
                ag.n_bat, ag.shuffle, **kwargs )
            for xy in ls_tgt_xy]

    if IS_OOD:
        main_stem( ag, ckpt, archtype, shape_x, dim_y,
                tr_src_loader, val_src_loader, ls_ts_tgt_loader )
    else:
        for testdom, tr_tgt_loader, ts_tgt_loader in zip(
                ag.testdoms, ls_tr_tgt_loader, ls_ts_tgt_loader):
            if testdom != ag.traindom:
                main_stem( ag, ckpt, archtype, shape_x, dim_y,
                        tr_src_loader, val_src_loader, None,
                        tr_tgt_loader, ts_tgt_loader, testdom )
            else:
                warnings.warn("same domain adaptation ignored")

