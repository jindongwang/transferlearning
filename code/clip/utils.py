import logging
import torch
import random
import sys
import numpy as np

def load_csv(folder, src_domain, tar_domain):
    data_s = np.loadtxt(f'{folder}/{src_domain}.csv', delimiter=' ')
    data_t = np.loadtxt(f'{folder}/{tar_domain}.csv', delimiter=' ')
    Xs, Ys = data_s[:, :-1], data_s[:, -1]
    Xt, Yt = data_t[:, :-1], data_t[:, -1]
    return Xs, Ys, Xt, Yt

def kernel(ker, X1, X2, gamma):
    K = None
    from sklearn.metrics.pairwise import linear_kernel, rbf_kernel
    import numpy as np
    if ker == 'linear':
        if X2 is not None:
            K = linear_kernel(
                np.asarray(X1), np.asarray(X2))
        else:
            K = linear_kernel(np.asarray(X1))
    elif ker == 'rbf':
        if X2 is not None:
            K = rbf_kernel(
                np.asarray(X1), np.asarray(X2), gamma)
        else:
            K = rbf_kernel(
                np.asarray(X1), None, gamma)
    elif ker == 'primal':
        K = X1
    return K

def set_seed(seed):
    random.seed(seed)    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_gpu(gpuid):
    torch.cuda.set_device(gpuid)


def get_logger(logger_name, log_file, level=logging.INFO):
    formatter = logging.Formatter('[%(asctime)s - %(levelname)s] %(message)s', "%Y-%m-%d %H:%M:%S")
    fileHandler = logging.FileHandler(log_file, mode='a')
    fileHandler.setFormatter(formatter)

    vlog = logging.getLogger(logger_name)
    vlog.setLevel(level)
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    vlog.addHandler(fileHandler)
    vlog.addHandler(stdout_handler)

    return vlog


def gather_res(model_name_lst, data_name_lst):
    res = []
    for model_name in model_name_lst:
        for data_name in data_name_lst:
            with open('/home/jindwang/mine/clipood/log/{}_{}.txt'.format(model_name, data_name), 'r') as f:
                lines = f.readlines()
                line = lines[-1].strip().split(' ')[-1]
            line = '{},{},{}'.format(model_name, data_name, line)
            res.append(line)
    return res
                

def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float()