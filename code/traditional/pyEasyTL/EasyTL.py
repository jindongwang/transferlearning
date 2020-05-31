import numpy as np
from intra_alignment import CORAL_map, GFK_map, PCA_map
# from label_prop import label_prop
from label_prop_v2 import label_prop

def get_cosine_dist(A, B):
    B = np.reshape(B, (1, -1))
    
    if A.shape[1] == 1:
        A = np.hstack((A, np.zeros((A.shape[0], 1))))
        B = np.hstack((B, np.zeros((B.shape[0], 1))))
    
    aa = np.sum(np.multiply(A, A), axis=1).reshape(-1, 1)
    bb = np.sum(np.multiply(B, B), axis=1).reshape(-1, 1)
    ab = A @ B.T
    
    # to avoid NaN for zero norm
    aa[aa==0] = 1
    bb[bb==0] = 1
    
    D = np.real(np.ones((A.shape[0], B.shape[0])) - np.multiply((1/np.sqrt(np.kron(aa, bb.T))), ab))
    
    return D
    
def get_ma_dist(A, B):
    Y = A.copy()
    X = B.copy()
    
    S = np.cov(X.T)
    try:
        SI = np.linalg.inv(S)
    except:
        print("Singular Matrix: using np.linalg.pinv")
        SI = np.linalg.pinv(S)
    mu = np.mean(X, axis=0)
    
    diff = Y - mu
    Dct_c = np.diag(diff @ SI @ diff.T)
    
    return Dct_c
    
def get_class_center(Xs,Ys,Xt,dist):
	
    source_class_center = np.array([])
    Dct = np.array([])
    for i in np.unique(Ys):
        sel_mask = Ys == i
        X_i = Xs[sel_mask.flatten()]
        mean_i = np.mean(X_i, axis=0)
        if len(source_class_center) == 0:
            source_class_center = mean_i.reshape(-1, 1)
        else:
            source_class_center = np.hstack((source_class_center, mean_i.reshape(-1, 1)))
		
        if dist == "ma":
            Dct_c = get_ma_dist(Xt, X_i)
        elif dist == "euclidean":
            Dct_c = np.sqrt(np.nansum((mean_i - Xt)**2, axis=1))
        elif dist == "sqeuc":
            Dct_c = np.nansum((mean_i - Xt)**2, axis=1)
        elif dist == "cosine":
            Dct_c = get_cosine_dist(Xt, mean_i)
        elif dist == "rbf":
            Dct_c = np.nansum((mean_i - Xt)**2, axis=1)
            Dct_c = np.exp(- Dct_c / 1);
        
        if len(Dct) == 0:
            Dct = Dct_c.reshape(-1, 1)
        else:
            Dct = np.hstack((Dct, Dct_c.reshape(-1, 1)))
    
    return source_class_center, Dct

def EasyTL(Xs,Ys,Xt,Yt,intra_align="coral",dist="euclidean",lp="linear"):
# Inputs:
#   Xs          : source data, ns * m
#   Ys          : source label, ns * 1
#   Xt          : target data, nt * m
#   Yt          : target label, nt * 1
# The following inputs are not necessary
#   intra_align : intra-domain alignment: coral(default)|gfk|pca|raw
#   dist        : distance: Euclidean(default)|ma(Mahalanobis)|cosine|rbf
#   lp          : linear(default)|binary

# Outputs:
#   acc         : final accuracy
#   y_pred      : predictions for target domain
    
# Reference:
# Jindong Wang, Yiqiang Chen, Han Yu, Meiyu Huang, Qiang Yang.
# Easy Transfer Learning By Exploiting Intra-domain Structures.
# IEEE International Conference on Multimedia & Expo (ICME) 2019.

    C = len(np.unique(Ys))
    if C > np.max(Ys):
        Ys += 1
        Yt += 1
	
    m = len(Yt)
	
    if intra_align == "raw":
        print('EasyTL using raw feature...')
    elif intra_align == "pca":
        print('EasyTL using PCA...')
        print('Not implemented yet, using raw feature')
		#Xs, Xt = PCA_map(Xs, Xt)
    elif intra_align == "gfk":
        print('EasyTL using GFK...')
        print('Not implemented yet, using raw feature')
        #Xs, Xt = GFK_map(Xs, Xt)
    elif intra_align == "coral":
        print('EasyTL using CORAL...')
        Xs = CORAL_map(Xs, Xt)
	
    _, Dct = get_class_center(Xs,Ys,Xt,dist)
    print('Start intra-domain programming...')
    Mcj = label_prop(C,m,Dct,lp)
    y_pred = np.argmax(Mcj, axis=1) + 1
    acc = np.mean(y_pred == Yt.flatten());

    return acc, y_pred
	
