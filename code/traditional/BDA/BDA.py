#encoding=utf-8
"""
    Created on 9:52 2018/11/14 
    @author: Jindong Wang
"""

# encoding=utf-8
"""
    Created on 15:09 2018/11/13 
    @author: Jindong Wang
"""

import numpy as np
import scipy.io
import scipy.linalg
import sklearn.neighbors
import sklearn.metrics


def kernel(ker, X, X2, gamma):
    if not ker or ker == 'primal':
        return X
    elif ker == 'linear':
        if not X2:
            K = np.dot(X.T, X)
        else:
            K = np.dot(X.T, X2)
    elif ker == 'rbf':
        n1sq = np.sum(X ** 2, axis=0)
        n1 = X.shape[1]
        if not X2:
            D = (np.ones((n1, 1)) * n1sq).T + np.ones((n1, 1)) * n1sq - 2 * np.dot(X.T, X)
        else:
            n2sq = np.sum(X2 ** 2, axis=0)
            n2 = X2.shape[1]
            D = (np.ones((n2, 1)) * n1sq).T + np.ones((n1, 1)) * n2sq - 2 * np.dot(X.T, X)
        K = np.exp(-gamma * D)
    elif ker == 'sam':
        if not X2:
            D = np.dot(X.T, X)
        else:
            D = np.dot(X.T, X2)
        K = np.exp(-gamma * np.arccos(D) ** 2)
    return K

from sklearn import svm

def proxy_a_distance(source_X, target_X, verbose=False):
    """
    Compute the Proxy-A-Distance of a source/target representation
    """
    nb_source = np.shape(source_X)[0]
    nb_target = np.shape(target_X)[0]

    if verbose:
        print('PAD on', (nb_source, nb_target), 'examples')

    C_list = np.logspace(-5, 4, 10)

    half_source, half_target = int(nb_source/2), int(nb_target/2)
    train_X = np.vstack((source_X[0:half_source, :], target_X[0:half_target, :]))
    train_Y = np.hstack((np.zeros(half_source, dtype=int), np.ones(half_target, dtype=int)))

    test_X = np.vstack((source_X[half_source:, :], target_X[half_target:, :]))
    test_Y = np.hstack((np.zeros(nb_source - half_source, dtype=int), np.ones(nb_target - half_target, dtype=int)))

    best_risk = 1.0
    for C in C_list:
        clf = svm.SVC(C=C, kernel='linear', verbose=False)
        clf.fit(train_X, train_Y)

        train_risk = np.mean(clf.predict(train_X) != train_Y)
        test_risk = np.mean(clf.predict(test_X) != test_Y)

        if verbose:
            print('[ PAD C = %f ] train risk: %f  test risk: %f' % (C, train_risk, test_risk))

        if test_risk > .5:
            test_risk = 1. - test_risk

        best_risk = min(best_risk, test_risk)

    return 2 * (1. - 2 * best_risk)


class BDA:
    def __init__(self, Xs, Ys, Xt, Yt, kernel_type='primal', dim=30, lamb=1, mu=-1.0, gamma=1, T=10, mode='BDA'):
        '''
        Init func
        :param Xs: ns * n_feature
        :param Ys: ns * 1
        :param Xt: nt * n_feature
        :param Yt: nt * 1
        :param kernel_type: kernel, values: 'primal' | 'linear' | 'rbf' | 'sam'
        :param dim: dimension after transfer
        :param lamb: lambda value in equation
        :param mu: mu. Default is -1, if not specificied, it calculates using A-distance
        :param gamma: kernel bandwidth for rbf kernel
        :param T: iteration number
        :param mode: 'BDA' | 'WBDA'
        '''
        self.Xs, self.Ys, self.Xt, self.Yt = Xs, Ys, Xt, Yt
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.mu = mu
        self.gamma = gamma
        self.T = T
        self.mode = mode

    def fit_predict(self):
        '''
        Transform and Predict using 1NN as JDA paper did
        :return: acc, y_pred, list_acc
        '''
        list_acc = []
        X = np.hstack((self.Xs.T, self.Xt.T))
        X = np.dot(X, np.diag(1 / (np.sum(X ** 2, axis=0) ** 0.5)))
        m, n = X.shape
        ns, nt = len(self.Xs), len(self.Xt)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        C = len(np.unique(Ys))
        H = np.eye(n) - 1 / n * np.ones((n, n))
        mu = self.mu
        M = e * e.T * C
        Y_tar_pseudo = None
        Xs_new = None
        for t in range(self.T):
            N = 0
            if Y_tar_pseudo is not None and len(Y_tar_pseudo) == nt:
                for c in range(1, C + 1):
                    e = np.zeros((n, 1))
                    if self.mode == 'WBDA':
                        Ns = len(self.Ys[np.where(self.Ys == c)])
                        Nt = len(Y_tar_pseudo[np.where(Y_tar_pseudo == c)])
                        Ps = Ns / len(self.Ys)
                        Pt = Nt / len(Y_tar_pseudo)
                    else:
                        Ps, Pt = 1, 1

                    tt = Ys == c
                    e[np.where(tt == True)] = np.sqrt(Ps) / len(self.Ys[np.where(self.Ys == c)])
                    yy = Y_tar_pseudo == c
                    ind = np.where(yy == True)
                    inds = [item + ns for item in ind]
                    e[tuple(inds)] = -np.sqrt(Pt) / len(Y_tar_pseudo[np.where(Y_tar_pseudo == c)])
                    e[np.isinf(e)] = 0
                    N = N + np.dot(e, e.T)
            if self.mu == -1.0:
                if Xs_new is not None:
                    mu = proxy_a_distance(Xs_new, Xt_new)
                else:
                    mu = 0
            M = (1 - mu) * M + mu * N
            M = M / np.linalg.norm(M, 'fro')
            K = kernel(self.kernel_type, X, None, gamma=self.gamma)
            n_eye = m if self.kernel_type == 'primal' else n
            a, b = np.linalg.multi_dot([K, M, K.T]) + self.lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
            w, V = scipy.linalg.eig(a, b)
            ind = np.argsort(w)
            A = V[:, ind[:self.dim]]
            Z = np.dot(A.T, K)
            Z = np.dot(Z, np.diag(1 / (np.sum(Z ** 2, axis=0) ** 0.5)))
            Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T
            clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1)
            clf.fit(Xs_new, self.Ys.ravel())
            Y_tar_pseudo = clf.predict(Xt_new)
            acc = sklearn.metrics.accuracy_score(Yt, Y_tar_pseudo)
            list_acc.append(acc)
            print('{} iteration [{}/{}]: Acc: {:.4f}'.format(self.mode, t + 1, self.T, acc))
        return acc, Y_tar_pseudo, list_acc


if __name__ == '__main__':
    domains = ['caltech.mat', 'amazon.mat', 'webcam.mat', 'dslr.mat']
    for i in range(1):
        for j in range(2):
            if i != j:
                src, tar = 'data/' + domains[i], 'data/' + domains[j]
                src_domain, tar_domain = scipy.io.loadmat(src), scipy.io.loadmat(tar)
                Xs, Ys, Xt, Yt = src_domain['feas'], src_domain['label'], tar_domain['feas'], tar_domain['label']
                bda = BDA(Xs, Ys, Xt, Yt, kernel_type='primal', dim=30, lamb=1, mu=0.5, mode='WBDA', gamma=1)
                acc, ypre, list_acc = bda.fit_predict()
                print(acc)
