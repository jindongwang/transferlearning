import torch

EPS = 1e-12

def robust_zscore_tensor(x):
    x -= torch.median(x, dim=0, keepdim=True)[0]
    x /= torch.median(torch.abs(x), dim=0, keepdim=True)[0] * 1.4826 + EPS
    x.clamp_(-3, 3)
    x -= torch.mean(x, dim=0, keepdim=True)
    x /= torch.std(x, unbiased=False, dim=0, keepdim=True) + EPS
    return x


def calc_ic(pred, label):
    pred = robust_zscore_tensor(pred)
    label = robust_zscore_tensor(label)
    return torch.mean(pred * label)


def calc_r2(pred, label):
    return 1 - (pred - label).pow(2).sum() / label.pow(2).sum()


def metric_fn(pred, label, metric='IC'):
    # robust pearson correlation
    mask = ~torch.isnan(label)
    if metric == 'IC':
        return calc_ic(pred[mask], label[mask])
    elif metric == 'R2':
        return calc_r2(pred[mask], label[mask])
    


def robust_zscore(x):
    # MAD based robust zscore
    x = x - x.median() # copy
    x /= x.abs().median() * 1.4826
    x.clip(-3, 3, inplace=True)
    return x


def calc_all_metrics(pred):
    """pred is a pandas dataframe that has two attributes: score (pred) and label (real)"""
    res = {}
    ic = pred.groupby(level='datetime').apply(
            lambda x: robust_zscore(x.label).corr(robust_zscore(x.score)))
    raw_ic = pred.groupby(level='datetime').apply(
        lambda x: x.label.corr(x.score))
    rank_ic = pred.groupby(level='datetime').apply(
        lambda x: x.label.corr(x.score, method='spearman'))

    print('Robust IC %.3f, Robust ICIR %.3f, Rank IC %.3f, Rank ICIR %.3f, Raw IC %.3f, Raw ICIR %.3f'%(
        ic.mean(), ic.mean()/ic.std(), rank_ic.mean(), rank_ic.mean()/rank_ic.std(), raw_ic.mean(), raw_ic.mean() / raw_ic.std()))

    res['IC'] = ic.mean()
    res['ICIR'] = ic.mean() / ic.std()
    res['RankIC'] = rank_ic.mean()
    res['RankICIR'] = rank_ic.mean() / rank_ic.std()
    res['RawIC'] = raw_ic.mean()
    res['RawICIR'] = raw_ic.mean() / raw_ic.std()
    return res

def logcosh(pred, label):
    # logcosh
    mask = ~torch.isnan(label)
    loss = torch.log(torch.cosh(pred[mask] - label[mask]))
    return torch.mean(loss)