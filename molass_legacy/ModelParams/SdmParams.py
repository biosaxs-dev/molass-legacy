"""
    ModelParams.SdmParams.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import logging
import numpy as np
from .BaselineParams import get_num_baseparams
from molass_legacy._MOLASS.SerialSettings import get_setting
from molass_legacy.Models.Stochastic.DispersiveUtils import NUM_SDMCOL_PARAMS
from molass_legacy.Optimizer.BasicOptimizer import AVOID_VANISHING_RATIO

NUM_COL_PARAMS = 6      # N, K, x0, poresize, N0, tI 

def get_common_parameter_names(nc):
    xr_names = ["$h_%d$" % k for k in range(nc)]
    rg_names = ["$R_{g%d}$" % k for k in range(nc)]
    mapping_names = ["$mp_a$", "$mp_b$"]
    uv_names = ["$uh_%d$" % k for k in range(nc)]
    mr_names = ["$mr_a$", "$mr_b$"]
    # seccol_names = ["$t_0$", "$r_p$", "N_0", "$N$", "$me$", "$T$", "mp"]
    seccol_names = ["$N$", "$K$", "$t_0$", "$poresize$", "$N_0$", "$t_I$"]

    return xr_names, rg_names, mapping_names, uv_names, mr_names, seccol_names

class SdmParams:
    def __init__(self, n_components):
        self.logger = logging.getLogger(__name__)
        self.n_components = n_components
        self.num_baseparams = get_num_baseparams()
        self.integral_baseline = self.num_baseparams == 3
        self.t0_upper_bound = get_setting("t0_upper_bound")
        self.use_K = False      # use_K was used for the deprecated stochastic model
        self.estimator = None

        nc = n_components - 1

        self.pos = []
        self.pos.append(0)      # [0] xr_params
        sep = nc
        self.pos.append(sep)    # [1] xr_baseparams
        sep += self.num_baseparams
        self.pos.append(sep)    # [2] rgs
        sep += nc
        self.pos.append(sep)    # [3] mapping
        sep = sep+2
        self.pos.append(sep)    # [4] uv_params
        sep += nc
        self.pos.append(sep)    # [5] uv_baseparams
        sep += 5 + self.num_baseparams
        self.pos.append(sep)    # [6] mappable_range
        sep += 2
        self.pos.append(sep)    # [7] end

        self.num_params = sep   # note that this does not include sec params

        self.logger.info("pos=%s", str(self.pos))

    def get_model_name(self):
        return 'SDM'

    def get_estimator(self, editor, debug=True):
        if debug:
            from importlib import reload
            import molass_legacy.Estimators.SdmEstimator
            reload(molass_legacy.Estimators.SdmEstimator)
        from molass_legacy.Estimators.SdmEstimator import SdmEstimator

        estimator = SdmEstimator(editor)
        self.t0_upper_bound = estimator.get_t0_upper_bound()
        self.estimator = estimator
        return estimator

    def set_estimator(self, estimator):
        """
        this is required for the on-the-fly debug in PeakEditor
        in order to avoid repeated computation of colparam_bounds.
        the cause of this neccessity is suspected to be a result of bad design.
        """
        self.logger.info("setting estimator for debugging purposes")
        self.estimator = estimator

    def get_colparam_bounds(self):
        return self.estimator.get_colparam_bounds()

    def split_params(self, n, params):
        if self.num_params != len(params):
            raise ValueError("len(params)=%d != %d, which is calculated from number of components %d." % (len(params), self.num_params, n))

        ret_params = []
        for p, q in zip(self.pos[:-1], self.pos[1:]):
            ret_params.append(params[p:q])

        return ret_params

    def split_params_simple(self, params):
        r = len(params) - self.num_params
        if r == 0:
            decomp_params = params
            seccol_params = None
        else:
            assert r == NUM_COL_PARAMS
            decomp_params = params[:-NUM_COL_PARAMS]
            seccol_params = params[-NUM_COL_PARAMS:]

        self.separate_params = self.split_params(self.n_components, decomp_params) + [seccol_params]
        return self.separate_params

    def split_as_unified_params(self, params):
        return self.split_params_simple(params) + [(None, None)]

    def get_param_bounds(self, params, real_bounds=None):
        init_xr_params, init_xr_baseparams, init_rgs, init_mapping, init_uv_params, init_uv_baseparams, init_mappable_range = self.split_params_simple(params)[0:7]

        xr_h_max = np.max(init_xr_params[:])
        xr_h_min = xr_h_max*AVOID_VANISHING_RATIO
        m_allow = 100

        xr_bounds = []
        for h in init_xr_params:
            xr_bounds.append((xr_h_min, xr_h_max*2))       # h

        for k, v in enumerate(init_xr_baseparams):
            v_allow = max(0.1, abs(v))*0.2          # note that v may be zero
            if self.integral_baseline and k == 2:
                bounds = (max(0, v - v_allow), v + v_allow)
            else:
                bounds = (v - v_allow, v + v_allow)
            xr_bounds.append(bounds)

        rg_bounds = [(rg*0.5, rg*2) for rg in init_rgs]
        a, b = init_mapping
        mapping_bounds = [(a*0.8, a*1.2), (-m_allow, m_allow)]

        uv_h_max = np.max(init_uv_params)
        uv_h_min = uv_h_max*AVOID_VANISHING_RATIO
        uv_bounds = [(uv_h_min, uv_h_max*2) for h in init_uv_params]
        for k, v in enumerate(init_uv_baseparams):
            v_allow = max(0.1, abs(v))*0.2          # note that v may be zero
            if self.integral_baseline and k == 7:
                bounds = (max(0, v - v_allow), v + v_allow)
            else:
                bounds = (v - v_allow, v + v_allow)
            uv_bounds.append(bounds)

        f, t = init_mappable_range
        dx = (t - f)*0.1
        range_bounds = [(f-dx, f+dx), (t-dx, t+dx)]

        # colparam_bounds
        # N, K, x0, poresize, tI
        if real_bounds is None:
            colparam_bounds = self.get_colparam_bounds()
            self.logger.info("got colparam_bounds=%s from the estimator", str(colparam_bounds))
        else:
            # self.estimator may be None in this case
            colparam_bounds = list(real_bounds[-NUM_COL_PARAMS:])
            self.logger.info("got colparam_bounds=%s from real_bounds", str(colparam_bounds))

        self.bounds_lengths = [len(b) for b in [xr_bounds, rg_bounds, mapping_bounds, uv_bounds, range_bounds, colparam_bounds]]
        return xr_bounds + rg_bounds + mapping_bounds + uv_bounds + range_bounds + colparam_bounds

    def make_bounds_mask(self):
        bounds_mask = np.zeros(self.num_params + NUM_COL_PARAMS, dtype=bool)
        n = self.n_components
        nc = n - 1
        bounds_mask[0:nc] = True            # xr_params
        sep = nc + self.num_baseparams

        if self.integral_baseline:
            bounds_mask[sep-1] = True           # xr baseline fouling
            bounds_mask[self.pos[6]-1] = True   # uv aseline fouling

        bounds_mask[sep:sep+nc] = True      # rgs

        return bounds_mask

    def get_rg_start_index(self):
        return self.pos[2]

    def get_mr_start_index(self):
        return self.pos[6]

    def get_parameter_names(self):
        xr_names, rg_names, mapping_names, uv_names, mr_names, seccol_names = get_common_parameter_names(self.n_components - 1)
        xr_basenames = ["$xb_a$", "$xb_b$"]
        if self.num_baseparams == 3:
            xr_basenames += ["$xb_r$"]
        uv_basenames = ["$L$", "$x_0$", "$k$", "$b$", "$s_1$", "$s_2$", "$diffratio$"]
        if self.num_baseparams == 3:
            uv_basenames += ["$ub_r$"]
        return np.array(xr_names + xr_basenames + rg_names + mapping_names + uv_names + uv_basenames + mr_names + seccol_names)

    def get_trans_indeces(self):
        return -4, -3, -2, -1

    def split_get_unified_sec_params(self, params):
        tI, t0, rp, N0, N, me, T, mp = self.split_params_simple(params)[-1]
        # K = N*T
        # m = me + mp
        return tI, t0, rp, N0, N*T, me+mp, N, T, me, mp, None, None, None

    def compute_comformance(self, xr_params, rg_params, seccol_params, poresize_bounds=None):
        # task:
        return 0

    def get_peak_pos_array_list(self, x_array):
        nc = self.n_components - 1
        gr_start = self.get_rg_start_index()
        me = 1.5
        mp = 1.5

        pos_array_list = []
        for params in x_array:
            rg_params = params[gr_start:gr_start+nc]
            N, K, x0, poresize, N0, tI  = params[-NUM_COL_PARAMS:]
            T = K/N
            rho = rg_params/poresize
            rho[rho > 1] = 1
            model_trs = x0 + N*T*(1 - rho)**(me + mp)
            pos_array_list.append(model_trs)

        return np.array(pos_array_list).T

    def get_params_sheet(self, parent, params, dsets, optimizer, debug=True):
        if debug:
            from importlib import reload
            import molass_legacy.ModelParams.SdmParamsSheet
            reload(molass_legacy.ModelParams.SdmParamsSheet)
        from .SdmParamsSheet import SdmParamsSheet
        return SdmParamsSheet(parent, params, dsets, optimizer)
    
    def get_paramslider_info(self, devel=True):
        if devel:
            from importlib import reload
            import molass_legacy.ModelParams.SdmSliderInfo
            reload(molass_legacy.ModelParams.SdmSliderInfo)
        from .SdmSliderInfo import SdmSliderInfo
        nc = self.n_components - 1
        return SdmSliderInfo(nc=nc)