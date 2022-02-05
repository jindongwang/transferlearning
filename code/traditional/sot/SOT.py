#coding utf-8
from numpy.lib.function_base import rot90
from scipy.spatial.distance import cdist
from sklearn.neighbors import KNeighborsClassifier
from sklearn import mixture
from collections import Counter
import json
import random
import numpy as np
from sklearn.metrics import euclidean_distances
import ot
import os
import joblib
from ot.optim import line_search_armijo

def norm_max(x):
    for i in range(x.shape[1]):
        tmax=x[:,i].max()
        x[:,i]=x[:,i]/tmax
    return x

def load_from_file(root_dir,filename,ss,ts):
    f1=root_dir+filename
    with open(f1,'r') as f:
        s=f.read()
        data=json.loads(s)
        xs,ys,xt,yt = np.array(data[ss]['x']), np.array(data[ss]['y']), np.array(data[ts]['x']),np.array(data[ts]['y'])
    xs=norm_max(xs)
    xt=norm_max(xt)
    ys=np.squeeze(ys)
    yt=np.squeeze(yt)
    ttty=min(Counter(ys).keys())
    ys=ys-ttty
    yt=yt-ttty
    return xs,ys,xt,yt

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def gmm_target(X,w,c,rootdir,filepath,modelpath,targetname,covtype='diag'):
    if not os.path.exists(rootdir+'data/'):
        os.mkdir(rootdir+'data/')
    if not os.path.exists(rootdir+'model/'):
        os.mkdir(rot90+'model/')
    if os.path.exists(filepath):
        pass
    else:
        gmm = mixture.GaussianMixture(n_components=c,covariance_type=covtype)
        gmm.fit(X)
        x1=[]
        x2=[]
        xmu=[]
        for i in range(len(gmm.weights_)):
            xmu.append(gmm.weights_[i]*w)
            x1.append(gmm.means_[i])
            x2.append(np.sqrt(gmm.covariances_[i]))
        data={'xntmu':np.array(xmu),'xnt1':np.array(x1),'xnt2':np.array(x2)}
        record={}
        record[targetname]=data
        with open(filepath,'w') as f:
            json.dump(record,f,cls=MyEncoder)
        joblib.dump(gmm,modelpath)

def gmm_source_class(X,w,slist,covtype='diag'):
    bicr=10000000000
    c=0
    gmm = mixture.GaussianMixture(n_components=slist[c],covariance_type=covtype)
    gmm.fit(X)
    while (c<len(slist))and(gmm.bic(X)<bicr):
        c+=1
        bicr=gmm.bic(X)
        gmm = mixture.GaussianMixture(n_components=slist[c],covariance_type=covtype)
        gmm.fit(X)
    c=c-1
    gmm = mixture.GaussianMixture(n_components=slist[c],covariance_type=covtype)
    gmm.fit(X)
    x1=[]
    x2=[]
    xmu=[]
    for i in range(len(gmm.weights_)):
        xmu.append(gmm.weights_[i]*w)
        x1.append(gmm.means_[i])
        x2.append(np.sqrt(gmm.covariances_[i]))
    return np.array(xmu),np.array(x1),np.array(x2),gmm

def gmm_source(xs,ys,filepath,sourcename,slist=[],covtype='diag'):
    if not os.path.exists(filepath):
        ty = Counter(ys)
        lys = len(ys)
        lc = len(Counter(ys))
        ws = {}
        for i in range(lc):
            ws[i] = ty[i] / lys
        if len(slist)==0:
            slist=np.arange(1,lys+1)
        for i in range(lc):
            xtmu, xt1, xt2, gmmt = gmm_source_class(xs[np.where(ys == i)[0]], ws[i],slist,covtype=covtype)
            yts = np.ones(len(xt1)) * i
            if i == 0:
                xn1, xn2, xmu = xt1, xt2, xtmu
                yns = yts
            else:
                xmu = np.hstack((xmu, xtmu))
                xn1 = np.vstack((xn1, xt1))
                xn2 = np.vstack((xn2, xt2))
                yns = np.hstack((yns, yts))
        data={'xmu':xmu,'xn1':xn1,'xn2':xn2,'yns':yns}
        record={sourcename:data}
        with open(filepath,'w') as f:
            json.dump(record,f,cls=MyEncoder)

def entropic_partial_wasserstein(a, b, M, reg, m=1, numItermax=500,
                                 stopThr=1e-100, verbose=False, log=False):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)

    dim_a, dim_b = M.shape

    if len(a) == 0:
        a = np.ones(dim_a, dtype=np.float64) / dim_a
    if len(b) == 0:
        b = np.ones(dim_b, dtype=np.float64) / dim_b

    if m > np.min((np.sum(a), np.sum(b))):
        m=np.min((np.sum(a),np.sum(b)))

    K = np.empty(M.shape, dtype=M.dtype)
    np.divide(M, -reg, out=K)
    np.exp(K, out=K)
    np.multiply(K, m / np.sum(K), out=K)
    K2 = np.dot(K, np.diag(b / np.sum(K, axis=0)))
    return K2

class SOT:
    def __init__(self,taskname='ACT',root_dir='./clustertemp/',d=200, 
                reg_e=0.1, reg_cl=0.1, reg_ce=0.1,rule='median'):
        self.taskname=taskname
        self.root_dir=root_dir
        self.d=d
        self.reg_e=reg_e
        self.reg_cl=reg_cl
        self.reg_ce=reg_ce
        self.rule=rule

    def get_target(self,filepath,modelpath,targetname):
        with open(filepath, 'r') as f:
            s = f.read()
            record = json.loads(s)
        self.diag_t = record[targetname]
        self.diag_g = joblib.load(modelpath)

    def get_source(self,filepath,sourcename):
        with open(filepath,'r') as f:
            s=f.read()
            record1=json.loads(s)
        self.diag_s=record1[sourcename]

    def partot_DA(self,Sx,Sy,Tx,b,xt1,ttt=1):
        a1 = np.ones(len(Sx))
        M = cdist(Sx, Tx, metric='sqeuclidean')
        M = M / np.median(M)
        b=np.ones(len(Tx))/len(Tx)
        T = entropic_partial_wasserstein(a1, b, M, self.reg_ce,m=1)
        if np.sum(T)<0.5:
            a=np.ones(len(Sx))/len(Sx)
            T=np.outer(a,b)
        gmm=self.diag_g
        index=gmm.predict(xt1)

        a=T.dot(np.ones(len(Tx)))
        b=(T.T).dot(np.ones(len(Sx)))
        G=ot.da.sinkhorn_lpl1_mm(a,Sy,b,M,self.reg_e,self.reg_cl)
        if np.sum(G)<0.5:
            a=np.ones(len(Sx))/len(Sx)
            G=np.outer(a,b)
        transp_Xs_lpl1 = np.diag(1 / G.dot(np.ones(len(Tx)))) @ G.dot(Tx)
        knn_clf = KNeighborsClassifier(n_neighbors=1)
        knn_clf.fit(transp_Xs_lpl1, Sy)
        Cls2 = knn_clf.predict(Tx)
        return Cls2[index]

    def fit_predict(self, Sx, Sy, Tx, Ty,sfilepath,sourcename,tfilepath,tmodelpath,targetname):
        gmm_source(Sx,Sy,sfilepath,sourcename)
        self.get_source(sfilepath,sourcename)
        gmm_target(Tx,1,self.d,self.root_dir,tfilepath,tmodelpath,targetname)
        self.get_target(tfilepath,tmodelpath,targetname)
        ss1 = self.diag_s
        tt1 = self.diag_t
        xns,yns=ss1['xn1'],ss1['yns']
        xntmu,xnt=tt1['xntmu'],tt1['xnt1']
        pred= self.partot_DA(xns,yns,xnt,xntmu,Tx,ttt=2)
        acc= np.mean(Ty==pred)
        return pred,acc

