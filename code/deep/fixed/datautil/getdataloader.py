# coding=utf-8
from torch.utils.data import DataLoader
import datautil.actdata.util as actutil
from datautil.util import make_weights_for_balanced_classes, split_trian_val_test
from datautil.mydataloader import InfiniteDataLoader
import datautil.actdata.cross_people as cross_people

task_act = {'cross_people': cross_people}


def get_dataloader(args, trdatalist, tedatalist):
    in_splits, out_splits = [], []
    for tr in trdatalist:
        if args.class_balanced:
            in_weights = make_weights_for_balanced_classes(tr)
        else:
            in_weights = None
        in_splits.append((tr, in_weights))
    for te in tedatalist:
        if args.class_balanced:
            out_weights = make_weights_for_balanced_classes(te)
        else:
            out_weights = None
        out_splits.append((te, out_weights))

    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=args.batch_size,
        num_workers=args.N_WORKERS)
        for i, (env, env_weights) in enumerate(in_splits)]

    tr_loaders = [DataLoader(
        dataset=env,
        batch_size=64,
        num_workers=args.N_WORKERS,
        drop_last=False,
        shuffle=False)
        for i, (env, env_weights) in enumerate(in_splits)]

    eval_loaders = [DataLoader(
        dataset=env,
        batch_size=64,
        num_workers=args.N_WORKERS,
        drop_last=False,
        shuffle=False)
        for i, (env, env_weights) in enumerate(in_splits + out_splits)]
    eval_weights = [None for _, weights in (in_splits + out_splits)]
    return train_loaders, tr_loaders, eval_loaders, in_splits, out_splits, eval_weights


def get_act_dataloader(args):
    train_datasetlist = []
    eval_datasetlist = []
    pcross_act = task_act[args.task]
    in_names, out_names = [], []
    trl = []
    tmpp = args.act_people[args.dataset]
    args.domain_num = len(tmpp)
    for i, item in enumerate(tmpp):
        tdata = pcross_act.ActList(
            args, args.dataset, args.data_dir, item, i, transform=actutil.act_train())
        if i in args.test_envs:
            eval_datasetlist.append(tdata)
            out_names.append('eval%d_out' % (i))
        else:
            in_names.append('eval%d_in' % (i))
            out_names.append('eval%d_out' % (i))
            trl.append(tdata)
            tr, te = split_trian_val_test(args, tdata)
            train_datasetlist.append(tr)
            eval_datasetlist.append(te)
    eval_loader_names = in_names
    eval_loader_names.extend(out_names)
    train_loaders, tr_loaders, eval_loaders, in_splits, out_splits, eval_weights = get_dataloader(
        args, train_datasetlist, eval_datasetlist)
    return train_loaders, tr_loaders, eval_loaders, in_splits, out_splits, eval_loader_names, eval_weights, trl
