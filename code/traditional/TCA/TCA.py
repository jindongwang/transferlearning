# encoding=utf-8
"""
    Created on 21:29 2018/11/12 
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


class TCA:
    def __init__(self, Xs, Ys, Xt, Yt, kernel_type='primal', dim=30, lamb=1, gamma=1):
        '''
        Init func
        :param Xs: ns * n_feature
        :param Ys: ns * 1
        :param Xt: nt * n_feature
        :param Yt: nt * 1
        :param kernel_type: kernel, values: 'primal' | 'linear' | 'rbf' | 'sam'
        :param dim: dimension after transfer
        :param lamb: lambda value in equation
        :param gamma: kernel bandwidth for rbf kernel
        '''
        self.Xs, self.Ys, self.Xt, self.Yt = Xs, Ys, Xt, Yt
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma

    def fit(self):
        '''
        Transform Xs and Xt
        :return: Xs_new and Xt_new after TCA
        '''
        X = np.hstack((self.Xs.T, self.Xt.T))
        X = np.dot(X, np.diag(1 / (np.sum(X ** 2, axis=0) ** 0.5)))
        m, n = X.shape
        ns, nt = len(self.Xs), len(self.Xt)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        M = e * e.T
        M = M / np.linalg.norm(M, 'fro')
        H = np.eye(n) - 1 / n * np.ones((n, n))
        K = kernel(self.kernel_type, X, None, gamma=self.gamma)
        n_eye = m if self.kernel_type == 'primal' else n
        a, b = np.linalg.multi_dot([K, M, K.T]) + self.lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
        w, V = scipy.linalg.eig(a, b)
        ind = np.argsort(w)
        A = V[:, ind[:self.dim]]
        Z = np.dot(A.T, K)
        Z = np.dot(Z, np.diag(1 / (np.sum(Z ** 2, axis=0) ** 0.5)))
        Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T
        return Xs_new, Xt_new

    def fit_predict(self):
        '''
        Transform Xs and Xt, then make predictions on target using 1NN
        :return: Accuracy and predicted_labels on the target domain
        '''
        Xs_new, Xt_new = self.fit()
        clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1)
        clf.fit(Xs_new, Ys.ravel())
        y_pred = clf.predict(Xt_new)
        acc = sklearn.metrics.accuracy_score(Yt, y_pred)
        return acc, y_pred


if __name__ == '__main__':
    # Assume you have the office-caltech10 dataset and put them into the 'data' folder
    # Data can be downloaded here: https://github.com/jindongwang/transferlearning/blob/master/data/dataset.md#officecaltech
    domains = ['caltech.mat', 'amazon.mat', 'webcam.mat', 'dslr.mat']
    for i in range(4):
        for j in range(4):
            if i != j:
                src, tar = 'data/' + domains[i], 'data/' + domains[j]
                src_domain, tar_domain = scipy.io.loadmat(src), scipy.io.loadmat(tar)
                Xs, Ys, Xt, Yt = src_domain['feas'], src_domain['label'], tar_domain['feas'], tar_domain['label']
                tca = TCA(Xs, Ys, Xt, Yt, kernel_type='primal', dim=30, lamb=1, gamma=1)
                acc, ypre = tca.fit_predict()
                print(acc)
