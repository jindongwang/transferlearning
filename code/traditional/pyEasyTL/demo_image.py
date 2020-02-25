# -*- coding: utf-8 -*-
import os
import time
import scipy.stats
import numpy as np
from EasyTL import EasyTL
import pandas as pd

if __name__ == "__main__":
    img_dataset = 'image-clef' # 'image-clef' or 'office-home'

    if img_dataset == 'image-clef':
        str_domains = ['c', 'i', 'p']
        datadir = r"D:\Datasets\EasyTL\imageCLEF_resnet50"
    elif img_dataset == 'office-home':
        str_domains = ['Art', 'Clipart', 'Product', 'RealWorld']
        datadir = r"D:\Datasets\EasyTL\officehome_resnet50"
    
    list_acc = []
    
    for i in range(len(str_domains)):
        for j in range(len(str_domains)):
            if i == j:
                continue
            
            print("{} - {}".format(str_domains[i], str_domains[j]))
            src = str_domains[i]
            tar = str_domains[j]
            x1file = "{}_{}.csv".format(src, src)
            x2file = "{}_{}.csv".format(src, tar)
            
            df1 = pd.read_csv(os.path.join(datadir, x1file), header=None)
            Xs = df1.values[:, :-1]
            Ys = df1.values[:, -1] + 1
    
            df2 = pd.read_csv(os.path.join(datadir, x2file), header=None)
            Xt = df2.values[:, :-1]
            Yt = df2.values[:, -1] + 1
            
            Xs = Xs / np.tile(np.sum(Xs,axis=1).reshape(-1,1), [1, Xs.shape[1]])
            Xs = scipy.stats.mstats.zscore(Xs)
            Xt = Xt / np.tile(np.sum(Xt,axis=1).reshape(-1,1), [1, Xt.shape[1]])
            Xt = scipy.stats.mstats.zscore(Xt)
            
            t0 = time.time()
            Acc1, _ = EasyTL(Xs,Ys,Xt,Yt,'raw')
            t1 = time.time()
            print("Time Elapsed: {:.2f} sec".format(t1 - t0))            
            Acc2, _ = EasyTL(Xs,Ys,Xt,Yt)
            t2 = time.time()
            print("Time Elapsed: {:.2f} sec".format(t2 - t1))
            
            print('EasyTL(c) Acc: {:.1f} % || EasyTL Acc: {:.1f} %'.format(Acc1*100, Acc2*100))
            list_acc.append([Acc1,Acc2])
            
    acc = np.array(list_acc)
    avg = np.mean(acc, axis=0)
    print('EasyTL(c) AVG Acc: {:.1f} %'.format(avg[0]*100))
    print('EasyTL AVG Acc: {:.1f} %'.format(avg[1]*100))
