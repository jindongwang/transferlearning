import numpy as np
import scipy
from sklearn.decomposition import PCA
import math

def GFK_map(Xs, Xt):
    pass

def gsvd(A, B):
    pass

def getAngle(Ps, Pt, DD):
    
    Q = np.hstack((Ps, scipy.linalg.null_space(Ps.T)))
    dim = Pt.shape[1]
    QPt = Q.T @ Pt
    A, B = QPt[:dim, :], QPt[dim:, :]
    U,V,X,C,S = gsvd(A, B)
    alpha = np.zeros([1, DD])
    for i in range(DD):
        alpha[0][i] = math.sin(np.real(math.acos(C[i][i]*math.pi/180)))
    
    return alpha

def getGFKDim(Xs, Xt):
    Pss = PCA().fit(Xs).components_.T
    Pts = PCA().fit(Xt).components_.T
    Psstt = PCA().fit(np.vstack((Xs, Xt))).components_.T
    
    DIM = round(Xs.shape[1]*0.5)
    res = -1
    
    for d in range(1, DIM+1):
        Ps = Pss[:, :d]
        Pt = Pts[:, :d]
        Pst = Psstt[:, :d]
        alpha1 = getAngle(Ps, Pst, d)
        alpha2 = getAngle(Pt, Pst, d)
        D = (alpha1 + alpha2) * 0.5
        check = [round(D[1, dd]*100) == 100 for dd in range(d)]
        if True in check:
            res = list(map(lambda i: i == True, check)).index(True) 
            return res

def PCA_map(Xs, Xt):
    dim = getGFKDim(Xs, Xt)
    X = np.vstack((Xs, Xt))
    X_new = PCA().fit_transform(X)[:, :dim]
    Xs_new = X_new[:Xs.shape[0], :]
    Xt_new = X_new[Xs.shape[0]:, :]
    return Xs_new, Xt_new

def CORAL_map(Xs,Xt):
    Ds = Xs.copy()
    Dt = Xt.copy()
      
    cov_src = np.ma.cov(Ds.T) + np.eye(Ds.shape[1])
    cov_tar = np.ma.cov(Dt.T) + np.eye(Dt.shape[1])
    
    Cs = scipy.linalg.sqrtm(np.linalg.inv(np.array(cov_src)))
    Ct = scipy.linalg.sqrtm(np.array(cov_tar))
    A_coral = np.dot(Cs, Ct)
    
    Xs_new = np.dot(Ds, A_coral)
        
    return Xs_new