"""
    Models.Stochastic.DispersiveLimits.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
from molass_legacy.Models.Stochastic.ParamLimits import ParamsScaler, make_monopore_bounds

UNDEF_BOUND = np.nan, np.nan

class DispersiveParamsScaler(ParamsScaler):
    def __init__(self, egh_moments_list, num_scales, peak_rgs, unreliable_indeces,
                 poresize_bounds=None, fronting=False, N0_bound=None):
        self.num_scales = num_scales
        self.num_unreliables = len(unreliable_indeces)
        reliable_indeces = np.setdiff1d(np.arange(len(peak_rgs)), unreliable_indeces)
        if len(reliable_indeces) > 0:
            reliable_rgmax = np.max(peak_rgs[reliable_indeces])
        else:
            # as in 20161104/BL-6A/pH6
            reliable_rgmax = np.max(peak_rgs)
        if np.isnan(reliable_rgmax):
            # as in 20161104/BL-6A/pH6
            reliable_rgmax = 100

        eghM = egh_moments_list[0]
        t0_upper = eghM[0]

        temp_bounds = make_monopore_bounds(t0_upper, num_scales, self.num_unreliables, reliable_rgmax, 0)
        self.bounds = np.concatenate([temp_bounds[0:4], [UNDEF_BOUND], temp_bounds[4:]])
        self.bounds[0] =  np.array([200, 3000])     # N
        self.bounds[1] =  np.array([200, 2000])     # K = NT

        M = np.array(egh_moments_list)
        total_sigma = (M[-1][0] + np.sqrt(M[-1][1])) - ( M[0][0] - np.sqrt(M[0][1]) )
        print("total_sigma=", total_sigma)
        t0_lower = t0_upper - total_sigma
        if fronting:
            # as in 20230303/HasA
            t0_lower -= 2*total_sigma
            t0_upper -= 1*total_sigma

        self.bounds[2] = t0_lower, t0_upper
        if poresize_bounds is not None:
            self.bounds[3] = poresize_bounds
        tI_bounds = self.bounds[2] - total_sigma*np.array([2, 1])
        if tI_bounds[1] > 0:
            tI_bounds[1] = 0   # i.e., tI <= 0
        tI_bounds[0] = min(tI_bounds[0], tI_bounds[1] - 1)
        self.bounds[4] = tI_bounds

        print("self.bounds=", self.bounds )
        """
        bounds[0]   N
        bounds[1]   K = NT
        bounds[2]   x0+
        bounds[3]   poresize
        bounds[4]   tI
        bounds[5]   unreliable Rg1
        """
        self.set_scales()
        print("self.scales=", self.scales)