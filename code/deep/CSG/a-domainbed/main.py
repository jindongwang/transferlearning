#!/usr/bin/env python3.6
import warnings
import sys
import torch as tc
sys.path.append("..")
from utils.utils_main import main_stem, get_parser, is_ood, process_continue_run
from utils.preprocess import data_loader
from utils.utils import boolstr, ZipLongest
from DomainBed.domainbed import datasets
from DomainBed.domainbed.lib import misc
from DomainBed.domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader

__author__ = "Chang Liu"
__email__ = "changliu@microsoft.com"
# tc.autograd.set_detect_anomaly(True)

class MergeIters:
    def __init__(self, *itrs):
        self.itrs = itrs
        self.zipped = ZipLongest(*itrs)
        self.len = len(self.zipped)

    def __iter__(self):
        for vals in self.zipped:
            yield tuple(tc.cat([val[i] for val in vals]) for i in range(len(vals[0])))

    def __len__(self): return self.len


if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument("--data_root", type = str, default = "./DomainBed/domainbed/data/")
    parser.add_argument('--dataset', type = str, default = "PACS")
    parser.add_argument("--testdoms", type = int, nargs = '+', default = [0])
    parser.add_argument("--n_bat_test", type = int, default = None)
    parser.add_argument("--traindoms", type = int, nargs = '+', default = None) # default: 'other' if `excl_test` else 'all'
    parser.add_argument("--excl_test", type = boolstr, default = True) # only active when `traindoms` is None (by default)
    parser.add_argument("--uda_frac", type = float, default = 1.)
    parser.add_argument("--data_aug", type = boolstr, default = True)

    parser.add_argument("--dim_s", type = int, default = 512)
    parser.add_argument("--dim_v", type = int, default = 128)
    parser.add_argument("--dim_btnk", type = int, default = 1024) # for discr_model
    parser.add_argument("--dims_bb2bn", type = int, nargs = '*') # for discr_model
    parser.add_argument("--dims_bn2s", type = int, nargs = '*') # for discr_model
    parser.add_argument("--dims_s2y", type = int, nargs = '*') # for discr_model
    parser.add_argument("--dims_bn2v", type = int, nargs = '*') # for discr_model
    parser.add_argument("--vbranch", type = boolstr, default = False) # for discr_model
    parser.add_argument("--dim_feat", type = int, default = 256) # for gen_model

    parser.set_defaults(discrstru = "DBresnet50", genstru = "DCGANpretr",
            n_bat = 32, n_epk = 40, eval_interval = 1,
            optim = "Adam", lr = 5e-5, wl2 = 5e-4,
            # momentum = .9, nesterov = True, lr_expo = .75, lr_wdatum = 6.25e-6, # only when "lr" is "SGD"
            sig_s = 3e+1, sig_v = 3e+1, corr_sv = .7, tgt_mvn_prior = True, src_mvn_prior = True,
            pstd_x = 1e-1, qstd_s = -1., qstd_v = -1.,
            wgen = 1e-7, wsup = 0., wlogpi = 1.,
            wda = .25,
            domdisc_dimh = 1024, # for {dann, cdan, mdd} only
            cdan_rand = False, # for cdan only
            ker_alphas = [.5, 1., 2.], # for dan only
            mdd_margin = 4. # for mdd only
        )
    ag = parser.parse_args()
    if ag.wlogpi is None: ag.wlogpi = ag.wgen
    if ag.n_bat_test is None: ag.n_bat_test = ag.n_bat
    ag, ckpt = process_continue_run(ag)
    IS_OOD = is_ood(ag.mode)

    ag.data_dir = ag.data_root
    ag.test_envs = ag.testdoms
    ag.holdout_fraction = 1. - ag.tr_val_split
    ag.uda_holdout_fraction = ag.uda_frac
    ag.trial_seed = 0.
    hparams = {'batch_size': ag.n_bat, 'class_balanced': False, 'data_augmentation': ag.data_aug}

    # BEGIN: from 'domainbed.scripts.train.py'
    if ag.dataset in vars(datasets):
        dataset = vars(datasets)[ag.dataset](ag.data_dir,
            ag.test_envs, hparams)
    else:
        raise NotImplementedError

    # (customed plugin)
    if ag.traindoms is None:
        ag.traindoms = list(i for i in range(len(dataset)) if not ag.excl_test or i not in ag.test_envs)
    ag.traindom = ag.traindoms # for printing info in `main_stem`
    # (end)

    # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.

    # To allow unsupervised domain adaptation experiments, we split each test
    # env into 'in-split', 'uda-split' and 'out-split'. The 'in-split' is used
    # by collect_results.py to compute classification accuracies.  The
    # 'out-split' is used by the Oracle model selectino method. The unlabeled
    # samples in 'uda-split' are passed to the algorithm at training time if
    # args.task == "domain_adaptation". If we are interested in comparing
    # domain generalization and domain adaptation results, then domain
    # generalization algorithms should create the same 'uda-splits', which will
    # be discared at training.
    in_splits = []
    out_splits = []
    uda_splits = []
    for env_i, env in enumerate(dataset):
        uda = []

        out, in_ = misc.split_dataset(env,
            int(len(env)*ag.holdout_fraction),
            misc.seed_hash(ag.trial_seed, env_i))

        if env_i in ag.test_envs:
            uda, in_ = misc.split_dataset(in_,
                int(len(in_)*ag.uda_holdout_fraction),
                misc.seed_hash(ag.trial_seed, env_i))

        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
            if uda is not None:
                uda_weights = misc.make_weights_for_balanced_classes(uda)
        else:
            in_weights, out_weights, uda_weights = None, None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
        if len(uda):
            uda_splits.append((uda, uda_weights))
    # Now `in_splits` and `out_splits` contain used-validation splits for all envs, and `uda_splits` contains the part of `in_splits` for uda for test envs only.

    if len(uda_splits) == 0: # args.task == "domain_adaptation" and len(uda_splits) == 0:
        raise ValueError("Not enough unlabeled samples for domain adaptation.")

    train_loaders = [FastDataLoader( # InfiniteDataLoader(
        dataset=env,
        # weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(in_splits)
        if i in ag.traindoms]

    val_loaders = [FastDataLoader( # InfiniteDataLoader(
        dataset=env,
        # weights=env_weights,
        batch_size=ag.n_bat_test, # hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(out_splits)
        if i in ag.traindoms]

    uda_loaders = [FastDataLoader( # InfiniteDataLoader(
        dataset=env,
        # weights=env_weights,
        batch_size = hparams['batch_size'] * len(train_loaders), # =hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(uda_splits)
        # if i in args.test_envs
    ]

    # eval_loaders = [FastDataLoader(
    #     dataset=env,
    #     batch_size=64,
    #     num_workers=dataset.N_WORKERS)
    #     for env, _ in (in_splits + out_splits + uda_splits)]
    # eval_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]
    # eval_loader_names = ['env{}_in'.format(i)
    #     for i in range(len(in_splits))]
    # eval_loader_names += ['env{}_out'.format(i)
    #     for i in range(len(out_splits))]
    # eval_loader_names += ['env{}_uda'.format(i)
    #     for i in range(len(uda_splits))]
    # END

    archtype = "cnn"
    shape_x = dataset.input_shape
    dim_y = dataset.num_classes
    tr_src_loader = MergeIters(*train_loaders)
    val_src_loader = MergeIters(*val_loaders)
    ls_ts_tgt_loader = uda_loaders
    if not IS_OOD:
        ls_tr_tgt_loader = uda_loaders

    if IS_OOD:
        main_stem( ag, ckpt, archtype, shape_x, dim_y,
                tr_src_loader, val_src_loader, ls_ts_tgt_loader )
    else:
        for testdom, tr_tgt_loader, ts_tgt_loader in zip(
                ag.testdoms, ls_tr_tgt_loader, ls_ts_tgt_loader):
            main_stem( ag, ckpt, archtype, shape_x, dim_y,
                    tr_src_loader, val_src_loader, None,
                    tr_tgt_loader, ts_tgt_loader, testdom )

