# coding: utf-8
"""
    EghMixture.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import numpy as np
import scipy.stats as stats
from scipy.optimize import fsolve
from sklearn.cluster import KMeans
import molass_legacy.KekLib.DebugPlot as plt

USE_MODE_AS_TR_INIT = True

def egh(x, H=1, tR=0, sigma=1.0, tau=0.0):
    x_  = x - tR
    s2  = 2 * sigma**2
    z   = s2 + tau*x_
    z_neg   = z <= 0
    z_pos   = z > 0

    zero_part = np.zeros( len(x) )[z_neg]
    posi_part = H * np.exp( -x_[z_pos]**2/( s2 + tau*x_[z_pos] ) )

    if tau >= 0:
        parts = [ zero_part, posi_part ]
    else:
        parts = [ posi_part, zero_part ]

    return np.hstack( parts )

"""
Kevin Lan, James W. Jorgenson, Journal of Chromatography A, 915 (2001) 1?13
"""
A0i = np.array([4, -6.293724, 9.232834, -11.342910, 9.123978, -4.173753, 0.827797])
e0 = np.poly1d(A0i[::-1])
"""
e0(th) = a0 + a1*th + a2*th**2 + ... + am*th**m
"""
A1i = np.array([0.75, 0.033807, -0.301080, 1.200371, -1.813317, 1.279318, -0.326582])
e1 = np.poly1d(A1i[::-1])

A2i = np.array([1, -0.982254, 0.568593, 0.512587, -1.184361, 0.939222, -0.240814])
e2 = np.poly1d(A2i[::-1])

A3i = np.array([0.5, -0.664611, 0.706192, -0.293500, -0.083980, 0.200306, -0.064264])
e3 = np.poly1d(A3i[::-1])

SQRT_PI_8 = np.sqrt(np.pi/8)
def egh_pdf(x, tR=0, sigma=1.0, tau=0.0, scale=1):
    tau_ = abs(tau)
    th = np.arctan2(tau_, sigma)
    M0 = (sigma*SQRT_PI_8 + abs(tau_))*e0(th)
    return egh(x, scale, tR, sigma, tau)/M0

class EghMixture:
    def __init__(self, K, max_iter=100):
        self.K = K
        self.max_iter = max_iter

    def fit(self, X, bins=None):
        assert X.shape[1] == 1
        self.N = X.shape[0]
        self.X = X
        self.X_  = X.flatten()
        self.bins = bins
        self.initilize()

        for step in range(self.max_iter):
            self._e_step()
            self._m_step()

        print('result pi=', self.pi)
        print('result tR=', self.tR)
        print('result sigma=', self.sigma)
        print('result tau=', self.tau)

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
        self.tR = _initial_means.flatten()
        self.sigma = np.sqrt(_initial_cov.flatten())
        self.tau = np.zeros(self.K)
        print('init pi=', self.pi)
        print('init tR=', self.tR)
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
        init_tR = []
        init_sigma = []

        for label in labels:
            sy = X[predicted == label]

            M1 = np.mean(sy)
            if USE_MODE_AS_TR_INIT:
                # seems risky
                m, c = stats.mode(sy)
                tR = m[0,0]
            else:
                tR = M1

            init_tR.append(tR)
            num_points = sy.shape[0]
            pi = num_points/num_entire_points
            init_pi.append(pi)
            # M2 = pi*np.sum((sy - tR)**2)/num_points     # why pi*?
            M2 = np.sum((sy - M1)**2)/num_points     # why pi*?
            sigma = np.sqrt(M2)
            # print([label], m, c, M2, np.sqrt(M2), sigma)
            init_sigma.append(sigma)

        self.pi = np.array(init_pi)
        self.tR = np.array(init_tR)
        self.sigma = np.array(init_sigma)
        self.tau = np.zeros(self.K)
        print('init pi=', self.pi)
        print('init tR=', self.tR)
        print('init sigma=', self.sigma)
        print('init tau=', self.tau)

        # assert np.sum(initial_pi) == 1
        assert abs(np.sum(self.pi) - 1) < 1e-10

        if False:
            plt.push()
            fig = plt.figure()
            ax = fig.gca()
            ax.hist(X.flatten(), bins=self.bins)
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin, ymax)
            for tR in self.tR:
                ax.plot([tR, tR], [ymin, ymax], ':', color='red')
            plt.show()
            plt.pop()

    def _e_step(self):
        self.gamma = np.zeros((self.N, self.K))

        for k in range(self.K):
            # Posterior Distribution using Bayes Rule
            self.gamma[:,k] = self.pi[k] * egh_pdf(self.X_, self.tR[k], self.sigma[k], self.tau[k])

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

    def _m_step(self):

        # responsibilities for each gaussian
        self.pi = np.mean(self.gamma, axis=0)
        self.m1 = np.dot(self.gamma.T, self.X) / np.sum(self.gamma, axis=0)

        W_ = np.sum(self.gamma, axis=0)[:,np.newaxis].T
        # print('GT.shape=', GT.shape)
        # print('GT[:,0:5]=', GT[:,0:5])

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
        GX = self.gamma * self.X
        M1 = np.sum(GX, axis=0) / W_
        M2 = np.sum(self.gamma * (self.X-M1)**2, axis=0) / W_
        # M3 = np.sum(self.gamma * (self.X-M1)**3, axis=0) / W_

        if debug:
            print('M1=', M1)
            print('M2=', M2)
            print('M3=', M3)

        params = []
        for k in range(self.K):
            params.append(self.solve_params(k, GX[:,k], M1[0,k], M2[0,k]))

        params_ = np.array(params)
        if debug:
            print('params_=', params_)
        self.tR = params_[:,0]
        self.sigma = params_[:,1]
        self.tau = params_[:,2]

    def solve_params(self, k, gx, m1, m2):

        gx_ = gx[gx > 10]

        if False:
            plt.push()
            fig = plt.figure()
            ax = fig.gca()
            ax.hist(gx_, bins=self.bins)
            plt.show()
            plt.pop()

        tR_, _ = stats.mode(gx_)
        tR = tR_[0]
        print([k], ('tR=', tR))

        # last_tR = self.tR[k]
        last_sigma = self.sigma[k]
        last_tau = self.tau[k]

        def equations(p):
            sigma, tau = p
            tau_ = abs(tau)         # better for 2019118_3
            # tau_ = tau            # seems better for some cases such as 20181203, although abs(tau) is mentioned in the paper.
            th = np.arctan2(tau_, sigma)
            return (
                        tR + tau*e1(th) - m1,
                        (sigma**2 + sigma*tau_ + tau**2)*e2(th) - m2,
                        # tau*(3*sigma**2 + 4*sigma*tau_ + 4*tau**2)*e3(th) - m3,
                   )

        sigma, tau = fsolve(equations, (last_sigma, last_tau))

        debug = False

        if debug:
            print([k], (last_tR, '->', tR))
            print([k], (last_sigma, '->', sigma))
            print([k], (last_tau, '->', tau))

        return tR, sigma, tau
