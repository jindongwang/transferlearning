# encoding=utf-8
"""
    Created on 14:52 2017/4/30 
    @author: Jindong Wang
"""

import numpy as np


class TCA:
    dim = 5
    kerneltype = 'rbf'
    kernelparam = 1
    mu = 1

    def __init__(self, dim=5, kerneltype='rbf', kernelparam=1, mu=1):
        '''
        Init function
        :param dim: dims after tca (dim <= d)
        :param kerneltype: 'rbf' | 'linear' | 'poly' (default is 'rbf')
        :param kernelparam: kernel param
        :param mu: param
        '''
        self.dim = dim
        self.kernelparam = kernelparam
        self.kerneltype = kerneltype
        self.mu = mu

    def get_L(self, n_src, n_tar):
        '''
        Get index matrix
        :param n_src: num of source domain 
        :param n_tar: num of target domain
        :return: index matrix L
        '''
        L_ss = (1. / (n_src * n_src)) * np.full((n_src, n_src), 1)
        L_st = (-1. / (n_src * n_tar)) * np.full((n_src, n_tar), 1)
        L_ts = (-1. / (n_tar * n_src)) * np.full((n_tar, n_src), 1)
        L_tt = (1. / (n_tar * n_tar)) * np.full((n_tar, n_tar), 1)
        L_up = np.hstack((L_ss, L_st))
        L_down = np.hstack((L_ts, L_tt))
        L = np.vstack((L_up, L_down))
        return L

    def get_kernel(self, kerneltype, kernelparam, x1, x2=None):
        '''
        Calculate kernel for TCA (inline func)
        :param kerneltype: 'rbf' | 'linear' | 'poly'
        :param kernelparam: param
        :param x1: x1 matrix (n1,d)
        :param x2: x2 matrix (n2,d)
        :return: Kernel K
        '''
        n1, dim = x1.shape
        K = None
        if x2 is not None:
            n2 = x2.shape[0]
        if kerneltype == 'linear':
            if x2 is not None:
                K = np.dot(x2, x1.T)
            else:
                K = np.dot(x1, x1.T)
        elif kerneltype == 'poly':
            if x2 is not None:
                K = np.power(np.dot(x1, x2.T), kernelparam)
            else:
                K = np.power(np.dot(x1, x1.T), kernelparam)
        elif kerneltype == 'rbf':
            if x2 is not None:
                sum_x2 = np.sum(np.multiply(x2, x2), axis=1)
                sum_x2 = sum_x2.reshape((len(sum_x2), 1))
                K = np.exp(-1 * (
                    np.tile(np.sum(np.multiply(x1, x1), axis=1).T, (n2, 1)) + np.tile(sum_x2, (1, n1)) - 2 * np.dot(x2,
                                                                                                                    x1.T)) / (
                               dim * 2 * kernelparam))
            else:
                P = np.sum(np.multiply(x1, x1), axis=1)
                P = P.reshape((len(P), 1))
                K = np.exp(
                    -1 * (np.tile(P.T, (n1, 1)) + np.tile(P, (1, n1)) - 2 * np.dot(x1, x1.T)) / (dim * 2 * kernelparam))
        # more kernels can be added
        return K

    def fit_transform(self, x_src, x_tar, x_tar_o=None):
        '''
        TCA main method. Wrapped from Sinno J. Pan and Qiang Yang's "Domain adaptation via transfer component ayalysis. IEEE TNN 2011" 
        :param x_src: Source domain data feature matrix. Shape is (n_src,d)
        :param x_tar: Target domain data feature matrix. Shape is (n_tar,d)
        :param x_tar_o: Out-of-sample target data feature matrix. Shape is (n_tar_o,d)
        :return: tranformed x_src_tca,x_tar_tca,x_tar_o_tca
        '''
        n_src = x_src.shape[0]
        n_tar = x_tar.shape[0]
        X = np.vstack((x_src, x_tar))
        L = self.get_L(n_src, n_tar)
        L[np.isnan(L)] = 0
        K = self.get_kernel(self.kerneltype, self.kernelparam, X)
        K[np.isnan(K)] = 0
        if x_tar_o is not None:
            K_tar_o = self.get_kernel(self.kerneltype, self.kernelparam, X, x_tar_o)

        H = np.identity(n_src + n_tar) - 1. / (n_src + n_tar) * np.ones(shape=(n_src + n_tar, 1)) * np.ones(
            shape=(n_src + n_tar, 1)).T
        forPinv = self.mu * np.identity(n_src + n_tar) + np.dot(np.dot(K, L), K)
        forPinv[np.isnan(forPinv)] = 0
        Kc = np.dot(np.dot(np.dot(np.linalg.pinv(forPinv), K), H), K)
        Kc[np.isnan(Kc)] = 0

        D, V = np.linalg.eig(Kc)
        eig_values = D.reshape(len(D), 1)
        eig_values_sorted = np.sort(eig_values[::-1], axis=0)
        index_sorted = np.argsort(-eig_values, axis=0)
        V = V[:, index_sorted]
        V = V.reshape((V.shape[0], V.shape[1]))
        x_src_tca = np.dot(K[:n_src, :], V)
        x_tar_tca = np.dot(K[n_src:, :], V)
        if x_tar_o is not None:
            x_tar_o_tca = np.dot(K_tar_o, V)
        else:
            x_tar_o_tca = None

        x_src_tca = np.asarray(x_src_tca[:, :self.dim], dtype=float)
        x_tar_tca = np.asarray(x_tar_tca[:, :self.dim], dtype=float)
        if x_tar_o is not None:
            x_tar_o_tca = x_tar_o_tca[:, :self.dim]
        return x_src_tca, x_tar_tca, x_tar_o_tca


if __name__ == '__main__':
    file_path = 'data/test_tca_data.csv'
    data = np.loadtxt(file_path, delimiter=',')
    x_src = data[:, :81]
    x_tar = data[:, 81:]

    # example usage
    my_tca = TCA(dim=30)
    x_src_tca, x_tar_tca, x_tar_o_tca = my_tca.fit_transform(x_src, x_tar)
    np.savetxt('x_src1.csv', x_src_tca, delimiter=',', fmt='%.6f')
    np.savetxt('x_tar1.csv', x_tar_tca, delimiter=',', fmt='%.6f')
