# encoding=utf-8
"""
    Created on 10:40 2018/11/14 
    @author: Jindong Wang
"""

import numpy as np
import scipy.io
from sklearn import metrics
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

import GFK


def kernel(ker, X1, X2, gamma):
    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2 is not None:
            K = metrics.pairwise.linear_kernel(np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = metrics.pairwise.linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2 is not None:
            K = metrics.pairwise.rbf_kernel(np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = metrics.pairwise.rbf_kernel(np.asarray(X1).T, None, gamma)
    return K


def proxy_a_distance(source_X, target_X):
    """
    Compute the Proxy-A-Distance of a source/target representation
    """
    nb_source = np.shape(source_X)[0]
    nb_target = np.shape(target_X)[0]
    train_X = np.vstack((source_X, target_X))
    train_Y = np.hstack((np.zeros(nb_source, dtype=int), np.ones(nb_target, dtype=int)))
    clf = svm.LinearSVC(random_state=0)
    clf.fit(train_X, train_Y)
    y_pred = clf.predict(train_X)
    error = metrics.mean_absolute_error(train_Y, y_pred)
    dist = 2 * (1 - 2 * error)
    return dist


class MEDA:
    def __init__(self, kernel_type='primal', dim=30, lamb=1, rho=1.0, eta=0.1, p=10, gamma=1, T=10):
        '''
        Init func
        :param kernel_type: kernel, values: 'primal' | 'linear' | 'rbf'
        :param dim: dimension after transfer
        :param lamb: lambda value in equation
        :param rho: rho in equation
        :param eta: eta in equation
        :param p: number of neighbors
        :param gamma: kernel bandwidth for rbf kernel
        :param T: iteration number
        '''
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.rho = rho
        self.eta = eta
        self.gamma = gamma
        self.p = p
        self.T = T

    def estimate_mu(self, _X1, _Y1, _X2, _Y2):
        adist_m = proxy_a_distance(_X1, _X2)
        C = len(np.unique(_Y1))
        epsilon = 1e-3
        list_adist_c = []
        for i in range(1, C + 1):
            ind_i, ind_j = np.where(_Y1 == i), np.where(_Y2 == i)
            Xsi = _X1[ind_i[0], :]
            Xtj = _X2[ind_j[0], :]
            adist_i = proxy_a_distance(Xsi, Xtj)
            list_adist_c.append(adist_i)
        adist_c = sum(list_adist_c) / C
        mu = adist_c / (adist_c + adist_m)
        if mu > 1:
            mu = 1
        if mu < epsilon:
            mu = 0
        return mu

    def fit_predict(self, Xs, Ys, Xt, Yt):
        '''
        Transform and Predict
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt: nt * n_feature, target feature
        :param Yt: nt * 1, target label
        :return: acc, y_pred, list_acc
        '''
        gfk = GFK.GFK(dim=self.dim)
        _, Xs_new, Xt_new = gfk.fit(Xs, Xt)
        Xs_new, Xt_new = Xs_new.T, Xt_new.T
        X = np.hstack((Xs_new, Xt_new))
        n, m = Xs_new.shape[1], Xt_new.shape[1]
        C = len(np.unique(Ys))
        list_acc = []
        YY = np.zeros((n, C))
        for c in range(1, C + 1):
            ind = np.where(Ys == c)
            YY[ind, c - 1] = 1
        YY = np.vstack((YY, np.zeros((m, C))))
        YY[0, 1:] = 0

        X /= np.linalg.norm(X, axis=0)
        L = 0  # Graph Laplacian is on the way...
        knn_clf = KNeighborsClassifier(n_neighbors=1)
        knn_clf.fit(X[:, :n].T, Ys.ravel())
        Cls = knn_clf.predict(X[:, n:].T)
        K = kernel(self.kernel_type, X, X2=None, gamma=self.gamma)
        E = np.diagflat(np.vstack((np.ones((n, 1)), np.zeros((m, 1)))))
        for t in range(1, self.T + 1):
            mu = self.estimate_mu(Xs_new.T, Ys, Xt_new.T, Cls)
            e = np.vstack((1 / n * np.ones((n, 1)), -1 / m * np.ones((m, 1))))
            M = e * e.T * C
            N = 0
            for c in range(1, C + 1):
                e = np.zeros((n + m, 1))
                tt = Ys == c
                e[np.where(tt == True)] = 1 / len(Ys[np.where(Ys == c)])
                yy = Cls == c
                ind = np.where(yy == True)
                inds = [item + n for item in ind]
                e[tuple(inds)] = -1 / len(Cls[np.where(Cls == c)])
                e[np.isinf(e)] = 0
                N += np.dot(e, e.T)
            M = (1 - mu) * M + mu * N
            M /= np.linalg.norm(M, 'fro')
            left = np.dot(E + self.lamb * M + self.rho * L, K) + self.eta * np.eye(n + m, n + m)
            Beta = np.dot(np.linalg.inv(left), np.dot(E, YY))
            F = np.dot(K, Beta)
            Cls = np.argmax(F, axis=1) + 1
            Cls = Cls[n:]
            acc = np.mean(Cls == Yt.ravel())
            list_acc.append(acc)
            print('MEDA iteration [{}/{}]: mu={:.2f}, Acc={:.4f}'.format(t, self.T, mu, acc))
        return acc, Cls, list_acc


if __name__ == '__main__':
    domains = ['caltech.mat', 'amazon.mat', 'webcam.mat', 'dslr.mat']
    for i in range(1):
        for j in range(2):
            if i != j:
                src, tar = 'data/' + domains[i], 'data/' + domains[j]
                src_domain, tar_domain = scipy.io.loadmat(src), scipy.io.loadmat(tar)
                Xs, Ys, Xt, Yt = src_domain['feas'], src_domain['label'], tar_domain['feas'], tar_domain['label']
                meda = MEDA(kernel_type='rbf', dim=20, lamb=10, rho=1.0, eta=0.1, p=10, gamma=1, T=10)
                acc, ypre, list_acc = meda.fit_predict(Xs, Ys, Xt, Yt)
                print(acc)
