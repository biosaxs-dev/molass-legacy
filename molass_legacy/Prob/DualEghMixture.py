# coding: utf-8
"""
    DualEghMixture.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import copy
import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize
from sklearn.cluster import KMeans
import molass_legacy.KekLib.DebugPlot as plt
from .EghMixture import egh_pdf, e1, e2, e3, EghMixture

USE_MODE_AS_TR_INIT = True
USE_MODERATE_SIGMAS = True
MODERATE_SIGMAS_RATIO = 1.5
APPLY_PARAM_CONSTRAINTS = True

class DualEghMixture:
    def __init__(self, K, max_iter=100, random_states=None, anim_data=False):
        self.K = K
        self.max_iter = max_iter
        self.random_states = random_states
        self.anim_data = anim_data

    def fit(self, X, bins=None):
        self.num_subs = len(X)
        self.X_list = X
        self.bins_list = bins
        self.subs = []
        for n in range(self.num_subs):
            random_state = None if self.random_states is None else self.random_states[n]
            proxy = EghMixture(self.K, self.max_iter, random_state=random_state, anim_data=self.anim_data)
            self.subs.append(proxy)

    def get_bins(self, n):
        return None if self.bins_list is None else self.bins_list[n]

    def get_sub(self, n):
        sub = self.subs[n]
        bins = self.get_bins(n)
        sub.fit(X=self.X_list[n], bins=bins)
        return sub

    def unified_fit(self, X, bins=None):
        assert bins is not None and len(bins) == 2

        self.fit(X, bins)

        for n, (sub, X) in enumerate(zip(self.subs, self.X_list)):
            print([n], X.shape)
            bins = self.get_bins(n)
            sub.initilize(X, bins=bins)

        for step in range(self.max_iter):
            params_list = []
            for sub in self.subs:
                sub._e_step()
                params_list.append(sub._m_step(step))

            new_params_list = self._u_step(params_list)
            for sub, params in zip(self.subs, new_params_list):
                sub.update_params(step, params)

    def _u_step(self, params_list):
        params_array = np.array(params_list)
        """
            [
                [(tR00, sigma00, tau00), (tR01, sigma01, tau01), ...],  # for UV ar 280
                [(tR10, sigma10, tau10), (tR11, sigma11, tau11), ...],  # for Xray at Q1
                ...
            ]
        """
        # print('params_array=', params_array)

        def obj_function(p):
            A, B = p
            return ( np.sum((A*params_array[1,:,0] + B - params_array[0,:,0])**2)
                    + np.sum(A*(params_array[1,:,1] - params_array[0,:,1])**2)
                    + np.sum(A*(params_array[1,:,2] - params_array[0,:,2])**2)
                    )

        A_init = self.bins_list[0]/self.bins_list[1]
        # print('A_init=', A_init)
        result = minimize(obj_function, (A_init, 0), args=())
        A, B = result.x

        A_ = 1/A
        B_ = -B/A

        tR_UV    = np.average([params_array[0,:,0], A*params_array[1,:,0] + B], axis=0)
        sigma_UV = np.average([params_array[0,:,1], A*params_array[1,:,1]], axis=0)
        tau_UV = np.average([params_array[0,:,2], A*params_array[1,:,2]], axis=0)

        tR_XR    = np.average([params_array[1,:,0], params_array[0,:,0]*A_ + B_], axis=0)
        sigma_XR = np.average([params_array[1,:,1], params_array[0,:,1]*A_], axis=0)
        tau_XR = np.average([params_array[1,:,2], params_array[0,:,2]*A_], axis=0)

        new_params_array = np.array([
                            np.array([tR_UV, sigma_UV, tau_UV]).T,
                            np.array([tR_XR, sigma_XR, tau_XR]).T,
                            ])
        # print('new_params_array=', new_params_array)
        return new_params_array
