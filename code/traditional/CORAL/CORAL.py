# encoding=utf-8
"""
    Created on 16:31 2018/11/13 
    @author: Jindong Wang
"""

import numpy as np
import scipy.io
import scipy.linalg
import sklearn.metrics
import sklearn.neighbors


class CORAL:
    def __init__(self, Xs, Ys, Xt, Yt):
        '''
        Init func
        :param Xs: ns * n_feature
        :param Ys: ns * 1
        :param Xt: nt * n_feature
        :param Yt: nt * 1
        '''
        self.Xs, self.Ys, self.Xt, self.Yt = Xs, Ys, Xt, Yt

    def fit(self):
        '''
        Perform CORAL on the source domain features
        :return: New source domain features
        '''
        cov_src = np.cov(Xs.T) + np.eye(Xs.shape[1])
        cov_tar = np.cov(Xt.T) + np.eye(Xt.shape[1])
        A_coral = np.dot(scipy.linalg.fractional_matrix_power(cov_src, -0.5),
                         scipy.linalg.fractional_matrix_power(cov_tar, 0.5))
        Xs_new = np.dot(Xs, A_coral)
        return Xs_new

    def fit_predict(self):
        '''
        Perform CORAL, then predict using 1NN classifier
        :return: Accuracy and predicted labels of target domain
        '''
        Xs_new = self.fit()
        clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1)
        clf.fit(Xs_new, Ys.ravel())
        y_pred = clf.predict(Xt)
        acc = sklearn.metrics.accuracy_score(Yt, y_pred)
        return acc, y_pred


if __name__ == '__main__':
    domains = ['caltech.mat', 'amazon.mat', 'webcam.mat', 'dslr.mat']
    for i in range(4):
        for j in range(4):
            if i != j:
                src, tar = 'data/' + domains[i], 'data/' + domains[j]
                src_domain, tar_domain = scipy.io.loadmat(src), scipy.io.loadmat(tar)
                Xs, Ys, Xt, Yt = src_domain['feas'], src_domain['label'], tar_domain['feas'], tar_domain['label']
                coral = CORAL(Xs, Ys, Xt, Yt)
                acc, ypre = coral.fit_predict()
                print(acc)
