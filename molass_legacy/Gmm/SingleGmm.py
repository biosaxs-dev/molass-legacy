# coding: utf-8
"""
    SingleGmm.py.

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
USE_SKLEARN_GMM = False


class SingleGmm:
    def __init__(self, k, max_iter=1000):
        if USE_SKLEARN_GMM:
            from sklearn.mixture import GaussianMixture
            self.gmm = GaussianMixture(k, max_iter=max_iter, covariance_type='spherical')
        else:
            from .FoleyGmm import GMM
            self.gmm = GMM(k, 40)

    def fit(self, X=None):
        """
        exected to get the following attributes
            weights_
            means_
            covariances_
        """

        gmm = self.gmm

        if USE_SKLEARN_GMM:
            gmm.fit(X=X)
            print('means_=', gmm.means_)
            print('covariances_=', gmm.covariances_)
            print('weights_=', gmm.weights_)

            self.weights_ = gmm.weights_
            self.means_ = gmm.means_.flatten()
            self.covariances_ = gmm.covariances_
            self.gamma = None   # ?
        else:
            gmm.fit(X)
            print('mu=', gmm.mu)
            print('sigma=', gmm.sigma)
            print('pi=', gmm.pi)

            self.weights_ = gmm.pi.flatten()
            self.means_ = gmm.mu.flatten()
            self.covariances_ = gmm.sigma.flatten()
            self.gamma = gmm.gamma
