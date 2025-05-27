# coding: utf-8
"""
    EmgMixture.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import sys
import logging
import numpy as np
import scipy.stats as stats
from scipy.optimize import fsolve
from sklearn.cluster import KMeans
import molass_legacy.KekLib.DebugPlot as plt
from lmfit import minimize, Parameters
from molass_legacy.Models.ElutionCurveModels import emg

SQRT_2PI_INV = 1/np.sqrt(2*np.pi)

def emg_pdf(x, mu, sigma, tau=0, scale=1):
    return 1/sigma * SQRT_2PI_INV * emg(x, scale, mu, sigma, tau)

USE_MODE_AS_MU_INIT = True
USE_MODERATE_SIGMAS = False
MODERATE_SIGMAS_RATIO = 1.5
APPLY_PARAM_CONSTRAINTS = True

class EmgMixture:
    model_name = "EMG"

    def __init__(self, K, max_iter=100, random_state=None, anim_data=False):
        self.logger = logging.getLogger(__name__)
        self.K = K
        self.max_iter = max_iter
        self.random_state = random_state
        self.anim_data = anim_data

    def fit(self, X, bins=None):
        assert X.shape[1] == 1
        self.initilize(X, bins)

        for step in range(self.max_iter):
            self._e_step()
            self.update_params(step, self._m_step(step))

        print('result pi=', self.pi)
        print('result mu=', self.mu)
        print('result sigma=', self.sigma)
        print('result tau=', self.tau)

    def initilize(self, X, bins):
        self.N = X.shape[0]
        self.X = X
        self.X_  = X.flatten()
        self.bins = bins
        self.guess_initial_params()

    def guess_initial_params(self):
        X = self.X
        num_entire_points = X.shape[0]
        kmeans = KMeans(n_clusters= self.K, random_state=self.random_state)
        kmeans.fit(X)
        # predicted = kmeans.predict(X)
        predicted = kmeans.labels_
        labels = np.unique(predicted)
        init_params = []

        for label in labels:
            sy = X[predicted == label]

            M1 = np.mean(sy)
            if USE_MODE_AS_MU_INIT:
                # seems risky
                m, c = stats.mode(sy)
                mu = m[0,0]
            else:
                mu = M1

            num_points = sy.shape[0]
            pi = num_points/num_entire_points
            # M2 = pi*np.sum((sy - mu)**2)/num_points     # why pi* in Foley's implementation?
            M2 = np.sum((sy - M1)**2)/num_points
            sigma = np.sqrt(M2)
            # print([label], m, c, M2, np.sqrt(M2), sigma)
            init_params.append((pi, mu, sigma))

        sorted_params = np.array(sorted(init_params, key=lambda x:x[1]))

        self.pi = sorted_params[:,0]
        self.mu = sorted_params[:,1]
        self.sigma = sorted_params[:,2]
        self.tau = np.zeros(self.K)
        if False:
            print('init pi=', self.pi)
            print('init mu=', self.mu)
            print('init sigma=', self.sigma)
            print('init tau=', self.tau)
        if APPLY_PARAM_CONSTRAINTS:
            self.min_sigma = self.sigma*0.5
            self.max_sigma = self.sigma*2
            self.min_mu = np.max([np.zeros(self.K), self.mu - self.sigma], axis=0)
            self.max_mu = self.mu + self.sigma

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
            self.tau_array = np.zeros(array_shape)
            self.gamma_array = np.zeros((self.max_iter+1, self.N, self.K))
            self.set_anim_params(0)

    def set_anim_params(self, n):
        self.pi_array[n,:] = self.pi
        self.mu_array[n,:] = self.mu
        self.sigma_array[n,:] = self.sigma
        self.tau_array[n,:] = self.tau
        if n > 0:
            self.gamma_array[n,:,:] = self.gamma

    def _e_step(self):
        self.gamma = np.zeros((self.N, self.K))

        for k in range(self.K):
            # Posterior Distribution using Bayes Rule
            self.gamma[:,k] = self.pi[k] * emg_pdf(self.X_, self.mu[k], self.sigma[k], self.tau[k])

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

        if False:
            from .GammaVisualizer import GammaVisualizer
            print(sys.getsizeof(self.gamma))
            vis = GammaVisualizer()
            vis.show(self.gamma)

    def _m_step(self, step, pi_only=False):

        # responsibilities for each gaussian
        self.pi = np.mean(self.gamma, axis=0)

        if pi_only:
            return

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
        M3 = np.sum(self.gamma * (self.X-M1)**3, axis=0) / W_

        if debug:
            print('M1=', M1)
            print('M2=', M2)
            print('M3=', M3)

        params = []
        for k in range(self.K):
            params.append(self.solve_params(k, M1[0,k], M2[0,k], M3[0,k]))

        params_ = np.array(params)
        if debug:
            print('params_=', params_)

        return params_

    def update_params(self, step, params_):
        self.mu = params_[:,0]
        if USE_MODERATE_SIGMAS:
            self.sigma = self.moderate_sigmas(params_[:,1])
        else:
            self.sigma = params_[:,1]
        self.tau = params_[:,2]

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

    def solve_params(self, k, m1, m2, m3):
        last_mu = self.mu[k]
        last_sigma = self.sigma[k]
        last_tau = self.tau[k]

        """
            See 1998, N. Dyson, 
            for moment calculation
        """

        def equations(p):
            mu, sigma, tau = p
            # print('equations: ', mu, sigma, tau)
            return (
                        mu + tau - m1,
                        sigma**2 + tau**2 - m2,
                        2*tau**3 - m3,
                   )

        x, infodict, ier, mesg = fsolve(equations, (last_mu, last_sigma, last_tau), full_output=True)
        if ier in [1, 4, 5]:
            """
            5 : The iteration is not making good progress, as measured by the
                improvement from the last ten iterations.
            4 : The iteration is not making good progress, as measured by the
                improvement from the last five Jacobian evaluations.
            """
            mu, sigma, tau = x
        else:
            print('ier=', ier)
            self.logger.warning(mesg)
            self.fsolve_error = True
            mu, sigma, tau = last_mu, last_sigma, last_tau

        if APPLY_PARAM_CONSTRAINTS:
            """
            better constraints desired
            """
            mu = max(self.min_mu[k], min(self.max_mu[k], mu))
            sigma = max(self.min_sigma[k], min(self.max_sigma[k], sigma))
            ns = sigma*3
            tau = max(-ns, min(ns, tau))

        debug = False

        if debug:
            print([k], (last_mu, '->', mu))
            print([k], (last_sigma, '->', sigma))
            print([k], (last_tau, '->', tau))

        return mu, sigma, tau

    def get_anim_components(self, x, y, n):
        gy_list = []
        ty = np.zeros(len(x))
        for k in range(len(self.pi)):
            w = self.pi_array[n,k]
            m = self.mu_array[n,k]
            s = self.sigma_array[n,k]
            t = self.tau_array[n,k]
            # print([n,k], w, m, s, t)
            gy = emg_pdf(x, m, s, t, w)
            gy_list.append([gy, m])
            ty += gy

        sorted_gy_list = sorted(gy_list, key=lambda x:x[1])

        def obj_func(params):
            S   = params['S']
            return ty*S - y

        params = Parameters()
        S_init = np.max(y)/np.max(ty)
        params.add('S', value=S_init, min=0, max=S_init*100 )
        result = minimize(obj_func, params, args=())

        scale = result.params['S'].value
        return [ ty*scale ] + [ scale*rec[0] for rec in sorted_gy_list ]

    def get_anim_C(self, x, y, n, total=False):
        gy_list = self.get_anim_components(x, y, n)
        if total:
            return np.array(gy_list[1:]), gy_list[0]
        else:
            return np.array(gy_list[1:])

    def get_peak_mean_x(self):
        return self.mu

def get_sorted_params(eghmm):
    # sort results in the order of the means
    params = zip(eghmm.mu, eghmm.sigma, eghmm.tau, eghmm.pi)
    sorted_params = sorted([ (m, s, t, w) for m, s, t, w in params], key=lambda p:p[0])
    return sorted_params

def get_curves(eghmm, x):
    sorted_params = get_sorted_params(eghmm)

    # plot the results
    gy_list = []
    ty = np.zeros(len(x))
    for m, s, t, w in sorted_params:
        gy = emg_pdf(x, m, s, t, w)
        gy_list.append(gy)
        ty += gy

    return ty, gy_list

