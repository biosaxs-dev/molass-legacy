"""
    StcParams.py

    Copyright (c) 2022-2023, SAXS Team, KEK-PF
"""
import numpy as np
from .MonoporeSecParams import MonoporeSecParams, NUM_SEC_PARAMS, sec_comformance_impl
from .BaselineParams import get_num_baseparams
from molass_legacy._MOLASS.SerialSettings import get_setting

def get_common_parameter_names(nc):
    xr_names = ["$h_%d$" % k for k in range(nc)]
    rg_names = ["$R_{g%d}$" % k for k in range(nc)]
    mapping_names = ["$mp_a$", "$mp_b$"]
    uv_names = ["$uh_%d$" % k for k in range(nc)]
    mr_names = ["$mr_a$", "$mr_b$"]
    seccol_names = ["$t_0$", "$r_p$", "$N$", "$me$", "$T$", "mp"]

    return xr_names, rg_names, mapping_names, uv_names, mr_names, seccol_names

class StcParamsBase:
    def __init__(self, n_components, poresize, poreexponent, use_K=False):
        self.n_components = n_components
        self.num_baseparams = get_num_baseparams()
        self.integral_baseline = self.num_baseparams == 3
        self.poresize = poresize
        if poreexponent == 0:
            # workaround for set_setting("poreexponent", None) not working
            poreexponent = None
        self.poreexponent = poreexponent
        sectype = MonoporeSecParams(poresize, poreexponent)

        """
        t0_upper_bound will be set as follows
            1.  initialzied from get_setting("t0_upper_bound")
                which is None as first in the monitor process,
                but has already been given a proper value in the optimizer process

            2.  in the the monitor process, it will be given a proper value later
                at MonoporeComponentParams.get_estimator,
                and then, it will be passed to the optimizer process
        """
        self.t0_upper_bound = sectype.t0_upper_bound

        self.init_method = sectype.init_method
        # self.conf_method = sectype.conf_method
        self.estm_method = sectype.estm_method
        self.nump_adjust = sectype.nump_adjust
        self.use_K = use_K

    def compute_init_guess(self, *args):
        return self.init_method(*args)

    def compute_comformance(self, *args, **kwargs):
        # return self.conf_method(*args, **kwargs)      # caused a bug in EGH model in Python 3.10.7
        return sec_comformance_impl(self.t0_upper_bound, *args, **kwargs)   # may be safer

    def estimate_conformance_params(self, *args):
        return self.estm_method(*args)

    def get_param_bounds(self, params):

        init_xr_params, init_xr_baseparams, init_rgs, init_mapping, init_uv_params, init_uv_baseparams, init_mappable_range = self.split_params_simple(params)[0:7]

        xr_h_max = np.max(init_xr_params[:])
        m_allow = 100

        xr_bounds = []
        for h in init_xr_params:
            xr_bounds.append((0, xr_h_max*2))       # h

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
        uv_bounds = [(0, uv_h_max*2) for h in init_uv_params]
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

        # t0, rp, N, me, T, mp
        if self.use_K:
            # K = N*T
            # m = me + mp
            # p = log(T)/log(K)
            # q = mp/m
            sec_bounds = [(-100, 100), (30, 150), (200, 3000), (1, 3), (-0.1, 1.1), (-0.2, 1.2)]
        else:
            sec_bounds = [(-100, 100), (30, 150), (100, 2000), (1, 3), (0.1, 10), (1, 3)]
        self.bounds_lengths = [len(b) for b in [xr_bounds, rg_bounds, mapping_bounds, uv_bounds, range_bounds, sec_bounds]]
        return xr_bounds + rg_bounds + mapping_bounds + uv_bounds + range_bounds + sec_bounds

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

        pos_array_list = []
        for params in x_array:
            rg_params = params[gr_start:gr_start+nc]
            t0, rp, N, me, T, mp = params[-NUM_SEC_PARAMS:]
            rho = rg_params/rp
            rho[rho > 1] = 1
            model_trs = t0 + N*T*(1 - rho)**(me + mp)
            pos_array_list.append(model_trs)

        return np.array(pos_array_list).T

    def get_params_sheet(self, parent, params, dsets, optimizer, debug=True):
        if debug:
            from importlib import reload
            import ModelParams.StcParamsSheet
            reload(ModelParams.StcParamsSheet)
        from .StcParamsSheet import StcParamsSheet
        return StcParamsSheet(parent, params, dsets, optimizer)

    def get_adjuster(self, debug=True):
        if debug:
            import ModelParams.StcAdjuster
            from importlib import reload
            reload(ModelParams.StcAdjuster)
        from .StcAdjuster import StcAdjuster
        return StcAdjuster()

class MonoporeComponentParams(StcParamsBase):
    def __init__(self, n_components, poresize=None, poreexponent=None, use_K=False):
        StcParamsBase.__init__(self, n_components, poresize, poreexponent, use_K=use_K)
        self.advanced = True
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


    def __str__(self):
        ps, px = [str(v) if v is None else "%g" % v for v in [self.poresize, self.poreexponent]]
        return "MonoporeComponentParams(nc=%d, ps=%s, px=%s)" % (self.n_components, ps, px)

    def get_estimator(self, editor, debug=True):
        if debug:
            from importlib import reload
            import Estimators.StcEstimator
            reload(Estimators.StcEstimator)
        from Estimators.StcEstimator import StcEstimator

        estimator = StcEstimator(editor)
        self.t0_upper_bound = estimator.get_t0_upper_bound()
        return estimator

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
            assert r == NUM_SEC_PARAMS
            decomp_params = params[:-NUM_SEC_PARAMS]
            seccol_params = params[-NUM_SEC_PARAMS:]

        self.separate_params = self.split_params(self.n_components, decomp_params) + [seccol_params]
        return self.separate_params

    def split_as_unified_params(self, params):
        return self.split_params_simple(params) + [(None, None)]

    def make_bounds_mask(self):
        bounds_mask = np.zeros(self.num_params + NUM_SEC_PARAMS, dtype=bool)
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
        t0, rp, N, me, T, mp = self.split_params_simple(params)[-1]
        # K = N*T
        # m = me + mp
        return t0, N*T, rp, me+mp, N, T, me, mp, None, None, None

def construct_stochastic_params_type(n_components, use_K=False):
    poresize = get_setting("poresize")
    poreexponent = get_setting("poreexponent")
    uv_basemodel = get_setting("uv_basemodel")
    if uv_basemodel == 1:
        params_type = MonoporeComponentParams(n_components, poresize, poreexponent, use_K=use_K)
    else:
        assert False
    return params_type
