# coding: utf-8
"""
    ZhiyaZuoGmm.py.

    adapted from:
        Expectation Maximization
        https://zhiyzuo.github.io/EM/

        or

        https://github.com/sohaib730/Gaussian-Mixture-Model

"""
import numpy as np
import numpy.random as rd
import scipy as sp
import seaborn as sns
from matplotlib import pyplot as plt
try:
    # for numba 1.49 or later
    from numba.core.decorators import jit
except:
    from numba.decorators import jit

class GMM(object):
    def __init__(self, X, k=2, max_iter=1000):
        # dimension
        X = np.asarray(X)
        print('X.shape=', X.shape)
        self.m, self.n = X.shape
        self.data = X.copy()
        # number of mixtures
        self.k = k
        self.max_iter = max_iter

    def _init(self):
        # init mixture means/sigmas
        if False:
            self.mean_arr = np.asmatrix(np.random.random((self.k, self.n)))
            self.sigma_arr = np.array([np.asmatrix(np.identity(self.n)) for i in range(self.k)])
        else:
            # improved procedure from MatsukenGmm.py.
            mu_list = []
            lim_list = []
            for d in range(self.n):
                max_x, min_x = np.max(self.data[:,d]), np.min(self.data[:,d])
                mu_list.append(rd.uniform(low=min_x, high=max_x, size=self.k))
            self.mean_arr = np.asmatrix(np.array(mu_list).T)
            self.sigma_arr = np.array([np.asmatrix(np.identity(self.n)*50) for i in range(self.k)])
        self.phi = np.ones(self.k)/self.k
        self.w = np.asmatrix(np.empty((self.m, self.k), dtype=float))
        print("init mean_arr=", self.mean_arr)
        print("init sigma_arr=", self.sigma_arr)
    
    def fit(self, tol=1e-4):
        self._init()
        num_iters = 0
        ll = 1
        previous_ll = 0
        print('self.max_iter=', self.max_iter)
        while(ll-previous_ll > tol):
            previous_ll = self.loglikelihood()
            self._fit()
            num_iters += 1
            if num_iters > self.max_iter:
                break
            ll = self.loglikelihood()
            print('Iteration %d: log-likelihood is %.6f'%(num_iters, ll))
        print('Terminate at %d-th iteration:log-likelihood is %.6f'%(num_iters, ll))
    
    def loglikelihood(self):
        ll = 0
        for i in range(self.m):
            tmp = 0
            for j in range(self.k):
                #print(self.sigma_arr[j])
                tmp += sp.stats.multivariate_normal.pdf(self.data[i, :], 
                                                        self.mean_arr[j, :].A1, 
                                                        self.sigma_arr[j, :]) *\
                       self.phi[j]
            ll += np.log(tmp) 
        return ll
    
    def _fit(self):
        self.e_step()
        self.m_step()

    # @jit(nopython=True)
    def e_step(self):
        # calculate w_j^{(i)}
        for i in range(self.m):
            den = 0
            for j in range(self.k):
                num = sp.stats.multivariate_normal.pdf(self.data[i, :], 
                                                       self.mean_arr[j].A1, 
                                                       self.sigma_arr[j]) *\
                      self.phi[j]
                den += num
                self.w[i, j] = num
            self.w[i, :] /= den
            wi_err = self.w[i, :].sum() - 1
            # print('wi_err=', wi_err)
            assert wi_err < 1e-4

    def m_step(self):
        for j in range(self.k):
            const = self.w[:, j].sum()
            self.phi[j] = 1/self.m * const
            _mu_j = np.zeros(self.n)
            _sigma_j = np.zeros((self.n, self.n))
            for i in range(self.m):
                _mu_j += (self.data[i, :] * self.w[i, j])
                _sigma_j += self.w[i, j] * ((self.data[i, :] - self.mean_arr[j, :]).T * (self.data[i, :] - self.mean_arr[j, :]))
                #print((self.data[i, :] - self.mean_arr[j, :]).T * (self.data[i, :] - self.mean_arr[j, :]))
            self.mean_arr[j] = _mu_j / const
            self.sigma_arr[j] = _sigma_j / const
        #print(self.sigma_arr)

def blog_demo():
    X = np.random.multivariate_normal([0, 3], [[0.5, 0], [0, 0.8]], 20)
    X = np.vstack((X, np.random.multivariate_normal([20, 10], np.identity(2), 50)))
    X.shape
    gmm = GMM(X)
    gmm.fit()

    print('mean', gmm.mean_arr)
    print('sigma', gmm.sigma_arr)
    print('phi', gmm.phi)

class GmmAdaptor:
    def __init__(self, k, max_iter=1000):
        self.k = k
        self.max_iter= max_iter

    def fit(self, X):

        while True:
            try:
                gmm = GMM(X, k=self.k, max_iter=self.max_iter)
                gmm.fit()
                break
            except AssertionError:
                # bad init. try until it succeeds.
                # this error can be avoided by improving init with kmeans.
                print('retry on AssertionError')
                continue

        print('mean', gmm.mean_arr)
        print('sigma', gmm.sigma_arr)
        print('phi', gmm.phi)

        self.weights_ = gmm.phi
        self.means_ = np.asarray(gmm.mean_arr).flatten()
        self.covariances_ = gmm.sigma_arr.flatten()
