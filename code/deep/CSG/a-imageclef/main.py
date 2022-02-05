#!/usr/bin/env python3.6
import warnings
import sys
sys.path.append("..")
from utils.utils_main import main_stem, get_parser, is_ood, process_continue_run
from utils.preprocess import data_loader
from utils.utils import boolstr

__author__ = "Chang Liu"
__email__ = "changliu@microsoft.com"
# tc.autograd.set_detect_anomaly(True)

if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument("--data_root", type = str, default = "./data/image_CLEF/")
    parser.add_argument("--traindom", type = str, default = "b")
    parser.add_argument("--testdoms", type = str, nargs = '+', default = ["b", "c", "i", "p"])

    parser.add_argument("--dim_s", type = int, default = 1024)
    parser.add_argument("--dim_v", type = int, default = 256)
    parser.add_argument("--dim_btnk", type = int, default = 1024) # for discr_model
    parser.add_argument("--dims_bb2bn", type = int, nargs = '*') # for discr_model
    parser.add_argument("--dims_bn2s", type = int, nargs = '*') # for discr_model
    parser.add_argument("--dims_s2y", type = int, nargs = '*') # for discr_model
    parser.add_argument("--dims_bn2v", type = int, nargs = '*') # for discr_model
    parser.add_argument("--vbranch", type = boolstr, default = False) # for discr_model
    parser.add_argument("--dim_feat", type = int, default = 128) # for gen_model

    parser.set_defaults(discrstru = "ResNet50", genstru = "DCGANvar",
            n_bat = 32, n_epk = 100,
            optim = "SGD", lr = 4e-3, wl2 = 5e-4,
            momentum = .9, nesterov = True, lr_expo = .75, lr_wdatum = 6.25e-6, # only when "lr" is "SGD"
            wda = .25,
            domdisc_dimh = 1024, # for {dann, cdan, mdd} only
            cdan_rand = False, # for cdan only
            ker_alphas = [.5, 1., 2.], # for dan only
            mdd_margin = 4. # for mdd only
        )
    ag = parser.parse_args()
    if ag.wlogpi is None: ag.wlogpi = ag.wgen
    ag, ckpt = process_continue_run(ag)
    IS_OOD = is_ood(ag.mode)

    archtype = "cnn"
    # Dataset
    shape_x = (3, 224, 224) # determined by the loader
    dim_y = 12
    kwargs = {'num_workers': 4, 'pin_memory': True}
    tr_src_loader, val_src_loader = data_loader.load_training(
            ag.data_root, ag.traindom, ag.n_bat, kwargs,
            ag.tr_val_split, rand_split=True ) # needs to rand split otherwise some classes are unseen in training.
    ls_ts_tgt_loader = [data_loader.load_testing(
            ag.data_root, testdom, ag.n_bat, kwargs)
        for testdom in ag.testdoms]
    if not IS_OOD:
        ls_tr_tgt_loader = [data_loader.load_training(
                ag.data_root, testdom, ag.n_bat, kwargs, -1)
            for testdom in ag.testdoms]

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

