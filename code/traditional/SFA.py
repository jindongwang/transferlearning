import sys
import math
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from scipy.sparse.linalg import svds as SVD
from sklearn import svm
from sklearn.metrics import accuracy_score


class SFA:
    '''
    spectrual feature alignment
    '''

    def __init__(self, l=500, K=100, base_classifer=svm.SVC()):
        self.l = l
        self.K = K
        self.m = 0
        self.ut = None
        self.phi = 1
        self.base_classifer = base_classifer
        self.ix = None
        self._ix = None
        return

    def fit(self, Xs, Xt):

        ix_s = np.argsort(np.sum(Xs, axis=0))
        ix_t = np.argsort(np.sum(Xt, axis=0))

        ix_s = ix_s[::-1][:self.l]
        ix_t = ix_t[::-1][:self.l]
        ix = np.intersect1d(ix_s, ix_t)
        _ix = np.setdiff1d(range(Xs.shape[1]), ix)
        self.ix = ix
        self._ix = _ix
        self.m = len(_ix)
        self.l = len(ix)

        X = np.concatenate((Xs, Xt), axis=0)
        DI = (X[:, ix] > 0).astype('float')
        DS = (X[:, _ix] > 0).astype('float')

        # construct co-occurrence matrix DSxDI
        M = np.zeros((self.m, self.l))
        for i in range(X.shape[0]):
            tem1 = np.reshape(DS[i], (1, self.m))
            tem2 = np.reshape(DI[i], (1, self.l))
            M += np.matmul(tem1.T, tem2)
        M = M/np.linalg.norm(M, 'fro')
        # #construct A matrix
        # tem_1 = np.zeros((self.m, self.m))
        # tem_2 = np.zeros((self.l, self.l))
        # A1 = np.concatenate((tem_1, M.T), axis=0)
        # A2 = np.concatenate((M, tem_2), axis=0)
        # A = np.concatenate((A1, A2), axis=1)
        # # compute laplace
        # D = np.zeros((A.shape[0], A.shape[1]))
        # for i in range(self.l+self.m):
        # 	D[i,i] = 1.0/np.sqrt(np.sum(A[i,:]))
        # L = (D.dot(A)).dot(D)
        # ut, _, _ = np.linalg.svd(L)
        M = sp.lil_matrix(M)
        D1 = sp.lil_matrix((self.m, self.m))
        D2 = sp.lil_matrix((self.l, self.l))
        for i in range(self.m):
            D1[i, i] = 1.0/np.sqrt(np.sum(M[1, :]).data[0])
        for i in range(self.l):
            D2[i, i] = 1.0/np.sqrt(np.sum(M[:, i]).T.data[0])
        B = (D1.tocsr().dot(M.tocsr())).dot(D2.tocsr())
        # print("Done.")
        # print("Computing SVD...")
        ut, s, vt = SVD(B.tocsc(), k=self.K)
        self.ut = ut
        return ut

    def transform(self, X):
        return np.concatenate((X, X[:, self._ix].dot(self.ut)), axis=1)

    def fit_predict(self, Xs, Xt, X_test, Ys, Y_test):
        ut = self.fit(Xs, Xt)
        Xs = self.transform(Xs)
        self.base_classifer.fit(Xs, Ys)
        X_test = self.transform(X_test)
        y_pred = self.base_classifer.predict(X_test)
        acc = accuracy_score(Y_test, y_pred)
        return acc
