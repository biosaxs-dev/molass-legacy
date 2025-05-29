"""
    EdmParams.py

    Copyright (c) 2023-2025, SAXS Team, KEK-PF
"""
import logging
import numpy as np
from .BaselineParams import get_num_baseparams
from molass_legacy._MOLASS.SerialSettings import get_setting
from molass_legacy.Models.RateTheory.EDM import edm_impl, MIN_CINJ, MAX_CINJ
from molass_legacy.Optimizer.BasicOptimizer import AVOID_VANISHING_RATIO

NUM_COL_PARAMS = 1
NUM_ELEMENT_PARAMS = 7

def get_common_parameter_names(nc):
    xr_names = []
    for k in range(nc):
        xr_names += ["$t0_%d$" % k, "$u_%d$" % k, "$a_%d$" % k, r"$\b_%d$" % k, r"$\e_%d$" % k, r"$\Dz_%d$" % k, r"cinj_%d" % k]

    rg_names = ["$R_{g%d}$" % k for k in range(nc)]
    mapping_names = ["$mp_a$", "$mp_b$"]
    uv_names = ["$uh_%d$" % k for k in range(nc)]
    mr_names = ["$mr_a$", "$mr_b$"]
    edmcol_names = ["$Tz$"]

    return xr_names, rg_names, mapping_names, uv_names, mr_names, edmcol_names

class EdmParams:
    def __init__(self, n_components):
        self.logger = logging.getLogger(__name__)
        self.n_components = n_components
        self.num_baseparams = get_num_baseparams()
        self.integral_baseline = self.num_baseparams == 3
        self.t0_upper_bound = get_setting("t0_upper_bound")
        self.use_K = False

        nc = n_components - 1

        self.pos = []
        self.pos.append(0)      # [0] xr_params
        sep = nc*NUM_ELEMENT_PARAMS
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

    def __str__(self):
        return "EdmParams(nc=%d)" % (self.n_components)

    def get_model_name(self):
        return 'EDM'

    def set_x(self, x):
        self.x = x

    def get_estimator(self, editor, debug=True):
        if debug:
            from importlib import reload
            import molass_legacy.Estimators.EdmEstimator
            reload(molass_legacy.Estimators.EdmEstimator)
        from molass_legacy.Estimators.EdmEstimator import EdmEstimator

        estimator = EdmEstimator(editor, self.n_components)
        self.t0_upper_bound = estimator.get_t0_upper_bound()
        return estimator

    def compute_init_guess(self, *args):
        assert False

    def compute_comformance(self, *args, **kwargs):
        assert False

    def estimate_conformance_params(self, *args):
        assert False

    def get_xr_param_bounds(self, xr_params):

        xr_h_max = np.max(xr_params[:,-1])      # cinj
        xr_h_min = xr_h_max*AVOID_VANISHING_RATIO
        xr_bounds = []
        for t0, u, a, b, e, Dz, cinj in xr_params:
            xr_bounds.append((-500, 1000))      # t0
            xr_bounds.append((0.00001, 50.0))   # u
            xr_bounds.append((0.0001, 1.0))     # a
            xr_bounds.append((-20.0, +20.0))    # b
            xr_bounds.append((0.001,  1.0))     # e
            xr_bounds.append((0.001, 40.0))     # Dz
            xr_bounds.append((max(MIN_CINJ, xr_h_min), min(MAX_CINJ, xr_h_max*2)))      # cinj

        return xr_bounds

    def get_param_bounds(self, params):

        init_xr_params, init_xr_baseparams, init_rgs, init_mapping, init_uv_params, init_uv_baseparams, init_mappable_range = self.split_params_simple(params)[0:7]

        m_allow = 100

        xr_bounds = self.get_xr_param_bounds(init_xr_params)

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

        # u
        edm_colparams = [(0.45, 0.55)]

        self.bounds_lengths = [len(b) for b in [xr_bounds, rg_bounds, mapping_bounds, uv_bounds, range_bounds, edm_colparams]]
        self.logger.info("bounds_lengths=%s", str(self.bounds_lengths))
        return xr_bounds + rg_bounds + mapping_bounds + uv_bounds + range_bounds + edm_colparams

    def split_bounds(self, bounds):
        separate_bounds = []
        offset = 0
        for k, length in enumerate(self.bounds_lengths):
            next_offset = offset + length
            sep_bounds = bounds[offset:next_offset]
            if k == 0:
                sep_bounds = self.reshape_xr_bounds(sep_bounds)
            separate_bounds.append(sep_bounds)
            offset = next_offset
        return separate_bounds

    def update_bounds_hook(self, masked_init_params):
        # nothing to do
        pass

    def get_peak_pos_array_list(self, x_array):
        nc = self.n_components - 1
        gr_start = self.get_rg_start_index()
        x = self.x

        pos_array_list = []
        for params in x_array:

            # these must be unified with split_params_simple
            xr_params = params[0:nc*NUM_ELEMENT_PARAMS].reshape((nc,NUM_ELEMENT_PARAMS))

            pos = []
            for t0, u, a, b, e, Dz, cinj in xr_params:
                xr_cy = edm_impl(x, t0, u, a, b, e, Dz, cinj)
                j = np.argmax(xr_cy)
                pos.append(x[j])
            pos_array_list.append(pos)

        return np.array(pos_array_list).T

    def get_params_sheet(self, parent, params, dsets, optimizer, debug=True):
        if debug:
            from importlib import reload
            import molass_legacy.ModelParams.EdmParamsSheet
            reload(molass_legacy.ModelParams.EdmParamsSheet)
        from .EdmParamsSheet import EdmParamsSheet
        return EdmParamsSheet(parent, params, dsets, optimizer)

    def get_adjuster(self, debug=True):
        if debug:
            from importlib import reload
            import molass_legacy.ModelParams.StcAdjuster
            reload(molass_legacy.ModelParams.StcAdjuster)
        from .StcAdjuster import StcAdjuster
        return StcAdjuster()

    def split_params(self, n, params):
        if self.num_params != len(params):
            raise ValueError("len(params)=%d != %d, which is calculated from number of components %d." % (len(params), self.num_params, n))

        nc = self.n_components - 1
        ret_params = []
        k = 0
        for p, q in zip(self.pos[:-1], self.pos[1:]):
            if k == 0:
                params_ = params[p:q].reshape((nc, NUM_ELEMENT_PARAMS))
            else:
                params_ = params[p:q]
            ret_params.append(params_)
            k += 1

        return ret_params

    def split_params_simple(self, params):
        decomp_params = params[:-NUM_COL_PARAMS]
        seccol_params = params[-NUM_COL_PARAMS:]
        self.separate_params = self.split_params(self.n_components, decomp_params) + [seccol_params]
        return self.separate_params

    def split_as_unified_params(self, params, **kwargs):
        class NotSupportedError(Exception): pass
        raise NotSupportedError()

    def make_bounds_mask(self):
        bounds_mask = np.zeros(self.num_params + NUM_COL_PARAMS, dtype=bool)
        n = self.n_components
        nc = n - 1
        bounds_mask[0:nc*5] = True              # xr_params
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
        # never called
        assert False

    def split_get_unified_sec_params(self, params):
        # should not be called
        assert False

    def compute_comformance(self, xr_params, rg_params, edm_colparams, **kwargs):
        # this is made in accordance to .SimpleSecParams.sec_comformance

        from molass_legacy.SecTheory.ColumnConstants import SECCONF_LOWER_BOUND, BAD_CONFORMANCE_REDUCE

        stdev = np.std(np.concatenate([xr_params[:,0], edm_colparams]))     # edm_colparams == [Tz]
        log_conformance = np.log10(stdev)

        if log_conformance > 0:
            log_conformance *= BAD_CONFORMANCE_REDUCE   # large conformance at early stages can be misleading

        return max(SECCONF_LOWER_BOUND, log_conformance)

    def get_paramslider_info(self, devel=True):
        if devel:
            from importlib import reload
            import molass_legacy.ModelParams.EdmSliderInfo
            reload(molass_legacy.ModelParams.EdmSliderInfo)
        from .EdmSliderInfo import EdmSliderInfo
        nc = self.n_components - 1
        return EdmSliderInfo(nc=nc)