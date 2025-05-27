# coding: utf-8
"""
    GssMixture.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import numpy as np
import scipy.stats as stats
from scipy.optimize import fsolve
from sklearn.cluster import KMeans
import molass_legacy.KekLib.DebugPlot as plt
from .GaussianMixture import gaussian_pdf

USE_MODE_AS_MU_INIT = False
USE_MODERATE_SIGMAS = False
MODERATE_SIGMAS_RATIO = 1.5

class GssMixture:
    def __init__(self, K, max_iter=100, anim_data=False):
        self.K = K
        self.max_iter = max_iter
        self.anim_data = anim_data

    def fit(self, X, bins=None):
        assert X.shape[1] == 1
        self.N = X.shape[0]
        self.X = X
        self.X_  = X.flatten()
        self.bins = bins
        self.initilize()

        for step in range(self.max_iter):
            self._e_step()
            self._m_step(step)

        print('result pi=', self.pi)
        print('result mu=', self.mu)
        print('result sigma=', self.sigma)

    def initilize(self):
        # self._initialise_parameters(self.X)
        self.guess_initial_params()

    def calculate_mean_covariance(self, X, prediction):
        """Calculate means and covariance of different
            clusters from k-means prediction

        Parameters:
        ------------
        prediction: cluster labels from k-means
        X: N*d numpy array data points 

        Returns:
        -------------
        intial_means: for E-step of EM algorithm
        intial_cov: for E-step of EM algorithm

        """
        d = X.shape[1]
        labels = np.unique(prediction)
        initial_means = np.zeros((self.K, d))
        initial_cov = np.zeros((self.K, d, d))
        initial_pi = np.zeros(self.K)
        
        counter=0
        for label in labels:
            ids = np.where(prediction == label) # returns indices
            initial_pi[counter] = len(ids[0]) / X.shape[0]
            initial_means[counter,:] = np.mean(X[ids], axis = 0)
            de_meaned = X[ids] - initial_means[counter,:]
            Nk = X[ids].shape[0] # number of data points in current gaussian
            initial_cov[counter,:, :] = np.dot(initial_pi[counter] * de_meaned.T, de_meaned) / Nk
            counter+=1

        # assert np.sum(initial_pi) == 1
        assert abs(np.sum(initial_pi) - 1) < 1e-10
        return (initial_means, initial_cov, initial_pi)

    def _initialise_parameters(self, X):
        """Implement k-means to find starting
            parameter values.
            https://datascience.stackexchange.com/questions/11487/how-do-i-obtain-the-weight-and-variance-of-a-k-means-cluster

        Parameters:
        ------------
        X: numpy array of data points

        Returns:
        ----------
        tuple containing initial means and covariance
        _initial_means: numpy array: (C*d)
        _initial_cov: numpy array: (C,d*d)

        """
        n_clusters = self.K
        kmeans = KMeans(n_clusters= n_clusters, init="k-means++", max_iter=500, algorithm = 'auto')
        fitted = kmeans.fit(X)
        prediction = kmeans.predict(X)
        _initial_means, _initial_cov, _initial_pi = self.calculate_mean_covariance(X, prediction)

        self.pi = _initial_pi
        self.mu = _initial_means.flatten()
        self.sigma = np.sqrt(_initial_cov.flatten())
        self.tau = np.zeros(self.K)
        print('init pi=', self.pi)
        print('init mu=', self.mu)
        print('init sigma=', self.sigma)
        print('init tau=', self.tau)

    def guess_initial_params(self):
        X = self.X
        num_entire_points = X.shape[0]
        kmeans = KMeans(n_clusters= self.K)
        kmeans.fit(X)
        predicted = kmeans.predict(X)
        labels = np.unique(predicted)
        init_pi = []
        init_mu = []
        init_sigma = []

        for label in labels:
            sy = X[predicted == label]

            M1 = np.mean(sy)
            if USE_MODE_AS_MU_INIT:
                # seems risky
                m, c = stats.mode(sy)
                mu = m[0,0]
            else:
                mu = M1

            init_mu.append(mu)
            num_points = sy.shape[0]
            pi = num_points/num_entire_points
            init_pi.append(pi)
            # M2 = pi*np.sum((sy - mu)**2)/num_points     # why pi* in Foley's implementation?
            M2 = np.sum((sy - M1)**2)/num_points
            sigma = np.sqrt(M2)
            # print([label], m, c, M2, np.sqrt(M2), sigma)
            init_sigma.append(sigma)

        self.pi = np.array(init_pi)
        self.mu = np.array(init_mu)
        self.sigma = np.array(init_sigma)
        print('init pi=', self.pi)
        print('init mu=', self.mu)
        print('init sigma=', self.sigma)

        # assert np.sum(initial_pi) == 1
        assert abs(np.sum(self.pi) - 1) < 1e-10

        if False:
            plt.push()
            fig = plt.figure()
            ax = fig.gca()
            ax.hist(X.flatten(), bins=self.bins)
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin, ymax)
            for mu in self.mu:
                ax.plot([mu, mu], [ymin, ymax], ':', color='red')
            plt.show()
            plt.pop()

        if self.anim_data:
            array_shape = (self.max_iter+1, self.K)
            self.pi_array = np.zeros(array_shape)
            self.mu_array = np.zeros(array_shape)
            self.sigma_array = np.zeros(array_shape)
            self.set_anim_params(0)

    def set_anim_params(self, n):
        self.pi_array[n,:] = self.pi
        self.mu_array[n,:] = self.mu
        self.sigma_array[n,:] = self.sigma

    def _e_step(self):
        self.gamma = np.zeros((self.N, self.K))

        for k in range(self.K):
            # Posterior Distribution using Bayes Rule
            self.gamma[:,k] = self.pi[k] * gaussian_pdf(self.X_, self.mu[k], self.sigma[k])

        # normalize across columns to make a valid probability
        # gamma_norm = np.sum(self.gamma, axis=1)[:,np.newaxis]
        gamma_norm = np.sum(self.gamma, axis=1)

        if False:
            print('pi=', self.pi)
            print('gamma_norm=', gamma_norm)
            plt.push()
            fig = plt.figure()
            ax = fig.gca()
            ax.plot(gamma_norm.flatten())
            plt.show()
            plt.pop()

        positive = gamma_norm > 0
        for k in range(self.K):
            self.gamma[positive,k] /= gamma_norm[positive]      # normalize only where positive

    def _m_step(self, step):

        # responsibilities for each gaussian
        self.pi = np.mean(self.gamma, axis=0)

        W_ = np.sum(self.gamma, axis=0)[:,np.newaxis].T
        # print('GT.shape=', GT.shape)
        # print('GT[:,0:5]=', GT[:,0:5])
        # print('W_=', W_)

        """
        2.1.1 Central Moments and their Distribution Functions
        https://www.iue.tuwien.ac.at/phd/puchner/node17.html#SECTION00811000000000000000
        2.1.2 Probability Weighted Moments and their Distribution Functions
        https://www.iue.tuwien.ac.at/phd/puchner/node18.html

        Central moment - Wikipedia
        https://en.wikipedia.org/wiki/Central_moment
        """

        debug = False

        # M1 = (np.dot(self.gamma.T, self.X) / W_.T).T
        M1 = np.sum(self.gamma * self.X, axis=0) / W_
        M2 = np.sum(self.gamma * (self.X-M1)**2, axis=0) / W_

        if debug:
            print('self.gamma[:5,:]=', self.gamma[:5,:])
            print('M1=', M1)
            print('M2=', M2)

        params = []
        for k in range(self.K):
            params.append(self.solve_params(k, M1[0,k], M2[0,k]))

        params_ = np.array(params)
        if debug:
            print('params_=', params_)
        self.mu = params_[:,0]
        if USE_MODERATE_SIGMAS:
            self.sigma = self.moderate_sigmas(params_[:,1])
        else:
            self.sigma = params_[:,1]

        if self.anim_data:
            self.set_anim_params(step+1)

    def moderate_sigmas(self, sigmas):
        if self.K > 1:
            L = list(range(self.K))
            k = np.argmax(sigmas)
            del L[k]
            sigmas_ = sigmas.copy()
            sigmas_[k] = min(sigmas[k], np.average(sigmas[L]) * MODERATE_SIGMAS_RATIO)
            return sigmas_
        else:
            return sigmas

    def solve_params(self, k, m1, m2):
        last_mu = self.mu[k]
        last_sigma = self.sigma[k]

        mu = m1
        sigma = np.sqrt(m2)

        debug = False

        if debug:
            print([k], (last_mu, '->', mu))
            print([k], (last_sigma, '->', sigma))

        return mu, sigma
