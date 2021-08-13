import collections
import torch
import os
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import numpy as np

EPS = 1e-12

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.list = []

    def update(self, val, n=1):
        self.val = val
        self.list.append(val)
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def average_params(params_list):
    assert isinstance(params_list, (tuple, list, collections.deque))
    n = len(params_list)
    if n == 1:
        return params_list[0]
    new_params = collections.OrderedDict()
    keys = None
    for i, params in enumerate(params_list):
        if keys is None:
            keys = params.keys()
        for k, v in params.items():
            if k not in keys:
                raise ValueError('the %d-th model has different params'%i)
            if k not in new_params:
                new_params[k] = v / n
            else:
                new_params[k] += v / n
    return new_params

def zscore(x):
    return (x - x.mean(dim=0, keepdim=True)) / x.std(dim=0, keepdim=True, unbiased=False)


def calc_loss(pred, label):
    return torch.mean((zscore(pred) - label) ** 2)


def calc_corr(pred, label):
    return (zscore(pred) * zscore(label)).mean()


def test_ic(model_list, data_list, device, verbose=True, ic_type='spearman'):
    '''
    model_list: [model1, model2, ...]
    datalist: [loader1, loader2, ...]
    return: unified ic, specific ic (all values), loss
    '''
    spec_ic = []
    loss_test = AverageMeter()
    loss_fn = torch.nn.MSELoss()
    label_true, label_pred = torch.empty(0).to(device), torch.empty(0).to(device)
    for i in range(len(model_list)):
        label_spec_true, label_spec_pred = torch.empty(0).to(device), torch.empty(0).to(device)
        model_list[i].eval()
        with torch.no_grad():
            for _, (feature, label_actual, _, _) in enumerate(data_list[i]):
                # feature = torch.tensor(feature, dtype=torch.float32, device=device)
                label_actual = label_actual.clone().detach().view(-1, 1)
                label_actual, mask = handle_nan(label_actual)
                label_predict = model_list[i].predict(feature).view(-1, 1)
                label_predict = label_predict[mask]
                loss = loss_fn(label_actual, label_predict)
                loss_test.update(loss.item())
                # Concat them for computing IC later
                label_true = torch.cat([label_true, label_actual])
                label_pred = torch.cat([label_pred, label_predict])
                label_spec_true = torch.cat([label_spec_true, label_actual])
                label_spec_pred = torch.cat([label_spec_pred, label_predict])
        ic = calc_ic(label_spec_true, label_spec_pred, ic_type)
        spec_ic.append(ic.item())
    unify_ic = calc_ic(label_true, label_pred, ic_type).item()
    # spec_ic.append(sum(spec_ic) / len(spec_ic))
    loss = loss_test.avg
    if verbose:
        print('[IC] Unified IC: {:.6f}, specific IC: {}, loss: {:.6f}'.format(unify_ic, spec_ic, loss))
    return unify_ic, spec_ic, loss

def test_ic_daily(model_list, data_list, device, verbose=True, ic_type='spearman'):
    '''
    model_list: [model1, model2, ...]
    datalist: [loader1, loader2, ...]
    return: unified ic, specific ic (all values + avg), loss
    '''
    spec_ic = []
    loss_test = AverageMeter()
    loss_fn = torch.nn.MSELoss()
    label_true, label_pred = torch.empty(0).to(device), torch.empty(0).to(device)
    for i in range(len(model_list)):
        label_spec_true, label_spec_pred = torch.empty(0).to(device), torch.empty(0).to(device)
        model_list[i].eval()
        with torch.no_grad():
            for slc in tqdm(data_list[i].iter_daily(), total=data_list[i].daily_length):
                feature, label_actual, _, _ = data_list[i].get(slc)
            # for _, (feature, label_actual, _, _) in enumerate(data_list[i]):
            #     feature = torch.tensor(feature, dtype=torch.float32, device=device)
                label_actual = torch.tensor(label_actual, dtype=torch.float32, device=device).view(-1, 1)
                label_actual, mask = handle_nan(label_actual)
                label_predict = model_list[i].predict(feature).view(-1, 1)
                label_predict = label_predict[mask]
                loss = loss_fn(label_actual, label_predict)
                loss_test.update(loss.item())
                # Concat them for computing IC later
                label_true = torch.cat([label_true, label_actual])
                label_pred = torch.cat([label_pred, label_predict])
                label_spec_true = torch.cat([label_spec_true, label_actual])
                label_spec_pred = torch.cat([label_spec_pred, label_predict])
        ic = calc_ic(label_spec_true, label_spec_pred, ic_type)
        spec_ic.append(ic.item())
    unify_ic = calc_ic(label_true, label_pred, ic_type).item()
    # spec_ic.append(sum(spec_ic) / len(spec_ic))
    loss = loss_test.avg
    if verbose:
        print('[IC] Unified IC: {:.6f}, specific IC: {}, loss: {:.6f}'.format(unify_ic, spec_ic, loss))
    return unify_ic, spec_ic, loss

def test_ic_uni(model, data_loader, model_path=None, ic_type='spearman', verbose=False):
    if model_path:
        model.load_state_dict(torch.load(model_path))
    model.eval()
    loss_all = []
    ic_all = []
    for slc in tqdm(data_loader.iter_daily(), total=data_loader.daily_length):
        data, label, _, _ = data_loader.get(slc)
        with torch.no_grad():
            pred = model.predict(data)
        mask = ~torch.isnan(label)
        pred = pred[mask]
        label = label[mask]
        loss = torch.mean(torch.log(torch.cosh(pred - label)))
        if ic_type == 'spearman':
            ic = spearman_corr(pred, label)
        elif ic_type == 'pearson':
            ic = pearson_corr(pred, label)
        loss_all.append(loss.item())
        ic_all.append(ic)
    loss, ic = np.mean(loss_all), np.mean(ic_all)
    if verbose:
        print('IC: ', ic)
    return loss, ic

def calc_ic(x, y, ic_type='pearson'):
    ic = -100
    if ic_type == 'pearson':
        ic = pearson_corr(x, y)
    elif ic_type == 'spearman':
        ic = spearman_corr(x, y)
    return ic

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def handle_nan(x):
    mask = ~torch.isnan(x)
    return x[mask], mask

class Log_Loss(nn.Module):
    def __init__(self):
        super(Log_Loss, self).__init__()
    
    def forward(self, ytrue, ypred):
        delta = ypred - ytrue
        return torch.mean(torch.log(torch.cosh(delta)))

def spearman_corr(x, y):
    X = pd.Series(x.cpu())
    Y = pd.Series(y.cpu())
    spearman = X.corr(Y, method='spearman')
    return spearman

def spearman_corr2(x, y):
    X = pd.Series(x)
    Y = pd.Series(y)
    spearman = X.corr(Y, method='spearman')
    return spearman

def pearson_corr(x, y):
    X = pd.Series(x.cpu())
    Y = pd.Series(y.cpu())
    spearman = X.corr(Y, method='pearson')
    return spearman

def dir_exist(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)