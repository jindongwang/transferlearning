# coding=utf-8
import collections
import os
import time
import numpy as np

from alg.opt import *
from alg import alg, modelopera
from utils.util import set_random_seed, get_args, init_args, print_row, train_valid_target_eval_names
from datautil.getdataloader import get_act_dataloader

if __name__ == '__main__':
    args = get_args()
    init_args(args)
    set_random_seed(args.seed1)

    if os.path.exists(os.path.join(args.output, 'newdone')):
        exit()

    train_loaders, tr_loaders, eval_loaders, in_splits, out_splits, eval_loader_names, eval_weights, trl = get_act_dataloader(
        args)
    algorithm_class = alg.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(args).cuda()

    algorithm.train()

    eval_dict = train_valid_target_eval_names(args)
    print_key = ['step', 'epoch']
    print_key.extend([item+'_acc' for item in eval_dict.keys()])
    print_key.append('total_cost_time')

    best_valid_acc, target_acc = 0, 0

    train_minibatches_iterator = zip(*train_loaders)

    start_step = 0
    steps_per_epoch = min(
        [len(env)/args.batch_size for env, _ in in_splits])
    n_steps = int(args.max_epoch*steps_per_epoch)+1
    checkpoint_freq = args.checkpoint_freq
    checkpoint_vals = collections.defaultdict(lambda: [])
    args.steps_per_epoch = steps_per_epoch

    opt = get_optimizer(algorithm, args)
    sch = get_scheduler(opt, args)

    print_row(print_key, colwidth=15)
    sss = time.time()
    for step in range(start_step, n_steps):
        step_start_time = time.time()
        minibatches_device = [(data)
                              for data in next(train_minibatches_iterator)]
        step_vals = algorithm.update(minibatches_device, opt, sch)
        checkpoint_vals['step_time'].append(
            time.time() - step_start_time)

        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        if (step % checkpoint_freq == 0) or (step == n_steps - 1):
            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            evals = zip(eval_loader_names, eval_loaders, eval_weights)
            for name, loader, weights in evals:
                acc = modelopera.accuracy(algorithm, loader, weights)
                results[name+'_acc'] = acc

            for key in eval_dict.keys():
                results[key+'_acc'] = np.mean(
                    np.array([results[item+'_acc'] for item in eval_dict[key]]))
            if results['valid_acc'] > best_valid_acc:
                best_valid_acc = results['valid_acc']
                target_acc = results['target_acc']
            results['total_cost_time'] = time.time()-sss
            print_row([results[key] for key in print_key], colwidth=15)

            results.update({
                'args': vars(args)
            })
            checkpoint_vals = collections.defaultdict(lambda: [])

    print('target acc:%.4f' % target_acc)
    with open(os.path.join(args.output, 'newdone'), 'w') as f:
        f.write('done\n')
        f.write('total cost time:%s\n' % (str(time.time()-sss)))
        f.write('target acc:%.4f\n' % (target_acc))
        f.write('valid acc:%.4f' % (best_valid_acc))
