# encoding=utf-8
"""
    Created on 21:29 2018/11/12 
    @author: Jindong Wang
"""
import numpy as np
import scipy.io
import scipy.linalg
import sklearn.metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


def kernel(ker, X1, X2, gamma):
    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2 is not None:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2 is not None:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, None, gamma)
    return K


class TCA:
    def __init__(self, kernel_type='primal', dim=30, lamb=1, gamma=1):
        '''
        Init func
        :param kernel_type: kernel, values: 'primal' | 'linear' | 'rbf'
        :param dim: dimension after transfer
        :param lamb: lambda value in equation
        :param gamma: kernel bandwidth for rbf kernel
        '''
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma
        self.W_proj = None
        self.X_new_src  = None
        self.clf = None

    def fit(self, Xs, Xt1):
        '''
        Transform Xs and Xt1
        :param Xs: ns * n_feature, source feature
        :param Xt1: nt * n_feature, target feature
        :return: Xs_new and Xt_new after TCA
        '''
        X = np.hstack((Xs.T, Xt1.T))
        X /= np.linalg.norm(X, axis=0)
        m, n = X.shape
        ns, nt = len(Xs), len(Xt1)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        M = e * e.T
        M = M / np.linalg.norm(M, 'fro')
        H = np.eye(n) - 1 / n * np.ones((n, n))
        K = kernel(self.kernel_type, X, None, gamma=self.gamma)
        n_eye = m if self.kernel_type == 'primal' else n
        a, b = np.linalg.multi_dot([K, M, K.T]) + self.lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
        w, V = scipy.linalg.eig(a, b)
        ind = np.argsort(w)
        
        # Projecting to latent space
        A = V[:, ind[:self.dim]]
        Z = np.dot(A.T, K)
        Z /= np.linalg.norm(Z, axis=0)
        
        Xs_new, Xt1_new = Z[:, :ns].T, Z[:, ns:].T
        
        self.W_proj = A
        self.X_new_src = X
        
        return Xs_new, Xt1_new

    def fit_predict(self, Xs, Ys, Xt1, Yt1):
        '''
        Transform Xs and Xt, then make predictions on target using 1NN
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt: nt * n_feature, target feature
        :param Yt: nt * 1, target label
        :return: Accuracy and predicted_labels on the target domain
        '''
        Xs_new, Xt1_new = self.fit(Xs, Xt1)
        clf = KNeighborsClassifier(n_neighbors=3)
        
        clf.fit(Xs_new, Ys.ravel())
        y_pred = clf.predict(Xt1_new)
        acc = sklearn.metrics.accuracy_score(Yt1, y_pred)
        
        self.clf = clf
        return acc, y_pred

    def fit_target(self, Xt2):
        '''
        Map new Xt to the latent space create from self.fit using existing projection matrix
        This Xt2 is different from Xt1 in the .fit
        :param Xt : n_s, n_feature, new target feature
        '''
        if not np.all(self.W_proj):
            raise AssertionError('No projection matrix found, use .fit to find new feature of source and target')
        
        # Reshape to make it consistent with other methods
        Xt2 = Xt2.T
        
        # Compute kernel with respect to self.X_new_src
        K = kernel(self.kernel_type, X1 = Xt2, X2 = self.X_new_src, gamma=self.gamma)
        
        # New target features
        Xt2_new = K @ self.W_proj
        
        return Xt2_new
    
    def fit_predict_target(self, Xt2, Yt2):
        '''
        Predict new target from self.fit_target using existing classifier built in .fit_predict
        Transform Xt2, then make predictions on target using 1NN
        :param Xt2: nt * n_feature, target feature
        :param Yt2: nt * 1, target label
        :return: Accuracy and predicted_labels on the target domain
        '''
        if not np.all(self.W_proj) or not np.all(self.clf) :
            raise AssertionError('No classifier found, trained classifier first using .fit_predict')
        
        Xt2_new = self.fit_target(Xt2)
        
        y_pred = self.clf.predict(Xt2_new)
        acc = sklearn.metrics.accuracy_score(Yt2, y_pred)
        
        return acc, y_pred
    
    

if __name__ == '__main__':
    domains = ['caltech.mat', 'amazon.mat', 'webcam.mat', 'dslr.mat']
    for i in [1]:
        for j in [2]:
            if i != j:
                src, tar = 'data/' + domains[i], 'data/' + domains[j]
                src_domain, tar_domain = scipy.io.loadmat(src), scipy.io.loadmat(tar)
                Xs, Ys, Xt, Yt = src_domain['feas'], src_domain['labels'], tar_domain['feas'], tar_domain['labels']
                
                # Split target data
                Xt1, Xt2, Yt1, Yt2  = train_test_split(Xt, Yt, train_size=50, stratify=Yt, random_state=42)
                
                # Create latent space and evaluate using Xs and Xt1
                tca = TCA(kernel_type='linear', dim=30, lamb=1, gamma=1)
                acc1, ypre1 = tca.fit_predict(Xs, Ys, Xt1, Yt1)
                
                # Project and evaluate Xt2 existing projection matrix and classifier
                acc2, ypre2 = tca.fit_predict_target(Xt2, Yt2)
                
    print(f'Accuracy of mapped source and target1 data : {acc1:.3f}') #0.800
    print(f'Accuracy of mapped target2 data            : {acc2:.3f}') #0.714