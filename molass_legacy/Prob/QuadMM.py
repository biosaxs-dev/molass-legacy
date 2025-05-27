# coding: utf-8
"""
    QuadMM.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import copy
import logging
import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize
from sklearn.cluster import KMeans
import molass_legacy.KekLib.DebugPlot as plt

USE_MODE_AS_TR_INIT = True
USE_MODERATE_SIGMAS = True
MODERATE_SIGMAS_RATIO = 1.5
APPLY_PARAM_CONSTRAINTS = True
WITHIN_THE_SAME_DATA_TYPE = True
UNIFY_TAU_BETWEEN_DATA_TYPES = True

class QuadMM:
    def __init__(self, mm_type, K, max_iter=100, random_states=None, anim_data=False):
        self.logger = logging.getLogger(__name__)
        self.mm_type = mm_type
        self.K = K
        self.max_iter = max_iter
        self.random_states = random_states
        self.anim_data = anim_data
        self.logger.info("QuadQMM constructed with random_states=%s", str(random_states))

    def get_model_name(self):
        return self.mm_type.model_name

    def set_subs(self, X, bins=None):
        self.num_subs = len(X)
        self.X_list = X
        self.bins_list = bins
        self.subs = []
        for n in range(self.num_subs):
            random_state = None if self.random_states is None else self.random_states[n]
            proxy = self.mm_type(self.K, self.max_iter, random_state=random_state, anim_data=self.anim_data)
            self.subs.append(proxy)

    def get_bins(self, n):
        return None if self.bins_list is None else self.bins_list[n]

    def get_sub(self, n):
        sub = self.subs[n]
        return sub

    def separate_fit(self, X, bins=None):
        assert bins is not None and len(bins) == 4

        self.set_subs(X, bins)
        for n, (sub, X, b) in enumerate(zip(self.subs, X, bins)):
            sub.fit(X=X, bins=b)

    def unified_fit(self, X, bins=None):
        assert bins is not None and len(bins) == 4

        self.set_subs(X, bins)

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
                [(tR10, sigma10, tau10), (tR11, sigma11, tau11), ...],  # for UV ar 260
                [(tR20, sigma20, tau20), (tR21, sigma21, tau21), ...],  # for Xray at Q1
                [(tR30, sigma30, tau30), (tR31, sigma31, tau31), ...],  # for Xray at Q2
                ...
            ]
        """
        debug = False
        if debug:
            print('params_array=', params_array)

        def obj_function(p):
            A, B = p
            ret = 0
            if UNIFY_TAU_BETWEEN_DATA_TYPES:
                for k in range(2):
                    ret += ( np.sum((A*params_array[k+2,:,0] + B - params_array[k,:,0])**2)
                            + np.sum(A*(params_array[k+2,:,1] - params_array[k,:,1])**2)
                            + np.sum(A*(params_array[k+2,:,2] - params_array[k,:,2])**2)
                            )
            else:
                for k in range(2):
                    ret += ( np.sum((A*params_array[k+2,:,0] + B - params_array[k,:,0])**2)
                            + np.sum(A*(params_array[k+2,:,1] - params_array[k,:,1])**2)
                            )
            return ret

        A_init = self.bins_list[0]/self.bins_list[1]
        # print('A_init=', A_init)
        result = minimize(obj_function, (A_init, 0), args=())
        A, B = result.x

        if WITHIN_THE_SAME_DATA_TYPE:
            tR_UV    = np.average([params_array[0,:,0], params_array[1,:,0]], axis=0)
            sigma_UV = np.average([params_array[0,:,1], params_array[1,:,1]], axis=0)
            tau_UV   = np.average([params_array[0,:,2], params_array[1,:,2]], axis=0)

            tR_XR    = np.average([params_array[2,:,0], params_array[3,:,0]], axis=0)
            sigma_XR = np.average([params_array[2,:,1], params_array[3,:,1]], axis=0)
            tau_XR   = np.average([params_array[2,:,2], params_array[3,:,2]], axis=0)
        else:
            A_ = 1/A
            B_ = -B/A

            tR_UV    = np.average([params_array[0,:,0], params_array[1,:,0], A*params_array[2,:,0] + B, A*params_array[3,:,0] + B], axis=0)
            sigma_UV = np.average([params_array[0,:,1], params_array[1,:,1], A*params_array[2,:,1], A*params_array[3,:,1]], axis=0)
            if UNIFY_TAU_BETWEEN_DATA_TYPES:
                tau_UV   = np.average([params_array[0,:,2], params_array[1,:,2], A*params_array[2,:,2], A*params_array[3,:,2]], axis=0)
            else:
                tau_UV   = np.average([params_array[0,:,2], params_array[1,:,2]], axis=0)

            tR_XR    = np.average([params_array[2,:,0], params_array[3,:,0], A_*params_array[0,:,0] + B_, A_*params_array[1,:,0] + B_], axis=0)
            sigma_XR = np.average([params_array[2,:,1], params_array[3,:,1], A_*params_array[0,:,1], A_*params_array[1,:,1]], axis=0)

            if UNIFY_TAU_BETWEEN_DATA_TYPES:
                tau_XR   = np.average([params_array[2,:,2], params_array[3,:,2], A_*params_array[0,:,2], A_*params_array[1,:,2]], axis=0)
            else:
                tau_XR   = np.average([params_array[2,:,2], params_array[3,:,2]], axis=0)

        UV_params = np.array([tR_UV, sigma_UV, tau_UV]).T
        XR_params = np.array([tR_XR, sigma_XR, tau_XR]).T
        new_params_array = np.array([
                            UV_params,
                            UV_params.copy(),
                            XR_params,
                            XR_params.copy(),
                            ])
        if debug:
            print('new_params_array=', new_params_array)
        return new_params_array
