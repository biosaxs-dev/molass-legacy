"""
    Models.Stochastic.ParamLimits.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
USE_K = True

N_BOUND = 300, 3000         # N=372 for 20160227, N>2000 is harmful for OA01
if USE_K:
    KT_BOUND = 500, 2000
else:
    KT_BOUND = 0.01, 3
T0_BOUND = [-1000, +1000]   # these will be changed according to t0_upper in make_mnp_bounds

PORESIZE_INTEG_LIMIT = 600  # changing this value to 600 once seemed harmful in LognormalPoreFunc.py
MAX_PORESIZE = 500      

M1_WEIGHT = 9
M2_WEIGHT = 1

PORESIZE_BOUNDS = 30, MAX_PORESIZE
PORESIZE_MEAN_BOUND = 40, MAX_PORESIZE + 50      # to be changed later
PORESIZE_STDEV_BOUND = 1, 100
LN_MU_BOUND = 3, 7      # exp(3)=20, exp(7)=1097
LN_SIGMA_BOUND = 0.05, 3

MNP_BOUNDS = [N_BOUND, KT_BOUND, T0_BOUND, PORESIZE_BOUNDS]
SCALE_BOUND = 0, 10
RG_BOUND = 5, 100       # upper bound will be later replaced according to reliable_rgmax
TIMESCALE_BOUND = 0.1, 2

BASINHOPPING_SCALE = 10

def make_monopore_bounds(t0_upper, num_scales, num_unreliables, reliable_rgmax, num_timescales):
    MNP_BOUNDS[2] = [t0_upper - 300, t0_upper]
    return np.array(MNP_BOUNDS
                    + [SCALE_BOUND] *num_scales
                    + [(RG_BOUND[0], reliable_rgmax*1.5)]*num_unreliables
                    + [TIMESCALE_BOUND]*num_timescales
                    )

N0_INIT = 14400     # 48000*0.3 (30cm) or (t0/σ0)**2, see meeting document 20221104/index.html 
N0_BOUND = N0_INIT//2, N0_INIT*2
PROP_BOUND = 0, 1

def make_oligopore_bounds(t0_lower, t0_upper, num_pszs, num_scales, num_unreliables, reliable_rgmax, num_timescales, dispersive=False):
    if t0_lower is None:
        t0_lower = t0_upper - 1000
    MNP_BOUNDS[2] = [t0_lower, t0_upper]
    bounds_list = (MNP_BOUNDS[0:3]
                    # + [(20, 50), (50, 100), (100, 500)]
                    + [PORESIZE_BOUNDS]*num_pszs      # allowing the upper bound to reach MAX_PORESIZE can cause problems?
                    + [PROP_BOUND]*(num_pszs-1)
                    )
    bounds_list += [SCALE_BOUND] *num_scales + [(RG_BOUND[0], reliable_rgmax*1.5)]*num_unreliables + [TIMESCALE_BOUND]*num_timescales
    if dispersive:
        bounds_list += [N0_BOUND]
    return np.array(bounds_list)

"""
task: unify V2-scaling using this scaler 
"""
class ParamsScaler:
    def set_scales(self):
        self.scales = (self.bounds[:,1] - self.bounds[:,0])

    def get_bounds(self):
        return self.bounds

    def scale(self, p):
        return (p - self.bounds[:,0])/self.scales * BASINHOPPING_SCALE

    def scale_back(self, p):
        return p/BASINHOPPING_SCALE * self.scales + self.bounds[:,0]

class MonoporeParamsScaler(ParamsScaler):
    def __init__(self, egh_moments_list, num_scales, peak_rgs, unreliable_indeces, num_timescales=0, allow_near_t0=False):
        self.num_scales = num_scales
        self.num_unreliables = len(unreliable_indeces)
        eghM = egh_moments_list[0]
        if allow_near_t0:
            t0_upper = eghM[0]
        else:
            t0_upper = eghM[0] - 5*np.sqrt(eghM[1])
        reliable_indeces = np.setdiff1d(np.arange(len(peak_rgs)), unreliable_indeces)
        reliable_rgmax = np.max(peak_rgs[reliable_indeces])
        self.bounds = make_monopore_bounds(t0_upper, num_scales, self.num_unreliables, reliable_rgmax, num_timescales)
        self.set_scales()

class OligoporeParamsScaler(ParamsScaler):
    def __init__(self, num_pszs, egh_moments_list, num_scales, peak_rgs, unreliable_indeces=[], num_timescales=0, dispersive=False, allow_near_t0=False, t0_lower=None):
        self.num_scales = num_scales
        self.num_unreliables = len(unreliable_indeces)
        eghM = egh_moments_list[0]
        if allow_near_t0:
            t0_upper = eghM[0]
        else:
            t0_upper = eghM[0] - 5*np.sqrt(eghM[1])
        reliable_indeces = np.setdiff1d(np.arange(len(peak_rgs)), unreliable_indeces)
        reliable_rgmax = np.max(peak_rgs[reliable_indeces])
        self.bounds = make_oligopore_bounds(t0_lower, t0_upper, num_pszs, num_scales, self.num_unreliables, reliable_rgmax, num_timescales, dispersive=dispersive)
        self.set_scales()