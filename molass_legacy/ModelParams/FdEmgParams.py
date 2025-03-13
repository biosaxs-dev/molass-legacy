"""
    FdEmgParams.py

    Copyright (c) 2022-2023, SAXS Team, KEK-PF
"""
import numpy as np
from .LjEghParams import LjEghParams
from .SimpleSecParams import initial_guess
from SecTheory.SecEstimator import NUM_SEC_PARAMS
from SecTheory.FoleyDorseyEmg import convert_to_xr_params_fd, compute_tau

class FdEmgParams(LjEghParams):
    def __init__(self, n_components):
        LjEghParams.__init__(self, n_components)

    def __str__(self):
        return "FdEmgParams(nc=%d)" % (self.n_components - 1)

    def get_estimator(self, editor, debug=True):
        if debug:
            from importlib import reload
            import Estimators.FdEmgEstimator
            reload(Estimators.FdEmgEstimator)
        from Estimators.FdEmgEstimator import FdEmgEstimator
        return FdEmgEstimator(editor)

    def split_params_simple(self, params, convert=True):
        req_params = params[0:-NUM_SEC_PARAMS]
        sec_params = params[-NUM_SEC_PARAMS:]
        Npc, tI = sec_params[[0,2]]
        split_params = self.split_params(self.n_components, req_params)
        if convert:
            xr_params = convert_to_xr_params_fd(split_params[0], tI, Npc)
        else:
            xr_params = split_params[0]
        self.separate_params = [xr_params] + split_params[1:] + [sec_params]
        return self.separate_params

    def split_as_unified_params(self, params, **kwargs):
        return self.split_params_simple(params, **kwargs)

    def get_xr_param_bounds(self, temp_xr_params, tI, Npc):
        xr_h_max = np.max(temp_xr_params[:,0])
        m_ = np.average(temp_xr_params[:,1])
        s_ = (m_ - tI)/np.sqrt(Npc)
        m_allow = np.max(temp_xr_params[:,1])*0.2
        xr_bounds = []
        for h, mu, As in temp_xr_params:        # temp_xr_params is not converted
            xr_bounds.append((0, xr_h_max*2))                       # h
            xr_bounds.append((max(0, mu - m_allow), mu + m_allow))  # m
            xr_bounds.append((s_*0.5, s_*1.5))                      # s
        return xr_bounds

    def get_xr_parameter_names(self):
        nc = self.n_components - 1
        xr_names = []
        for k in range(nc):
            xr_names += ["$h_%d$" % k, r"$\mu_%d$" % k, r"$\sigma_%d$" % k]
        return xr_names

    def get_peak_pos_array_list(self, x_array):
        n = self.n_components - 1
        pos_array_list = []
        tI = x_array[:,-2]
        Npc = x_array[:,-1]
        for k in range(n):
            mu = x_array[:,3*k+1]
            sigma = x_array[:,3*k+2]
            tau = compute_tau(tI, Npc, mu, sigma)    # note that this is a vector computation case
            pos_array_list.append(mu + tau)
        return pos_array_list

    def get_params_sheet(self, parent, params, dsets, optimizer):
        from .FdEmgParamsSheet import FdEmgParamsSheet
        return FdEmgParamsSheet(parent, params, dsets, optimizer)

    def compute_xr_peak_position(self, i, xr_params):
        return xr_params[i,1] + xr_params[i,3]
