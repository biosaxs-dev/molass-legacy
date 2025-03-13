"""
    RtEmgParams.py

    Copyright (c) 2022-2023, SAXS Team, KEK-PF
"""
import numpy as np
from .EghParams import EghParamsBase, get_common_parameter_names
from molass_legacy._MOLASS.SerialSettings import get_setting
from SecTheory.RetensionTime import compute_retention_time
from SecTheory.HermansEmg import convert_to_xr_params_hermans
from SecTheory.ColumnConstants import SECCONF_LOWER_BOUND, BAD_CONFORMANCE_REDUCE
from .SimpleSecParams import SEC_PENALTY_SCALE

SEC_START = -5
SEC_STOP = -1

class RtEmgParams(EghParamsBase):
    def __init__(self, n_components):
        EghParamsBase.__init__(self, n_components, None, None)
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
        return "RtEmgParams(nc=%d)" % (self.n_components - 1)

    def get_estimator(self, editor, debug=True):
        if debug:
            from importlib import reload
            import Estimators.RtEmgEstimator
            reload(Estimators.RtEmgEstimator)
        from Estimators.RtEmgEstimator import RtEmgEstimator
        return RtEmgEstimator(editor)

    def split_params(self, n, params):
        if self.num_params != len(params):
            raise ValueError("len(params)=%d != %d, which is calculated from number of components %d." % (len(params), self.num_params, n))

        ret_params = []
        for p, q in zip(self.pos[:-1], self.pos[1:]):
            ret_params.append(params[p:q])

        return ret_params

    def split_params_simple(self, params, convert=True):
        req_params = params[0:SEC_START]
        sec_params = params[SEC_START:SEC_STOP]
        R = params[SEC_STOP]
        split_params = self.split_params(self.n_components, req_params)
        if convert:
            xr_params = convert_to_xr_params_hermans(split_params[0], split_params[2], sec_params, R)
        else:
            xr_params = split_params[0]
        self.separate_params = [xr_params] + split_params[1:] + [sec_params, R]
        return self.separate_params

    def split_as_unified_params(self, params, **kwargs):
        class NotSupportedError(Exception): pass
        raise NotSupportedError()

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

        """
        task: unify bounds setting
        """

        Np = get_setting("numplates_pc")
        RATE_R_UPPER_BOUND = get_setting("RATE_R_UPPER_BOUND")

        #               t0,         P,          rp,         m,      R
        sec_bounds = [(-200, 200), (200, 3000), (30, 150), (1, 6), (0, RATE_R_UPPER_BOUND)]

        self.bounds_lengths = [len(b) for b in [xr_bounds, rg_bounds, mapping_bounds, uv_bounds, range_bounds, sec_bounds]]
        return xr_bounds + rg_bounds + mapping_bounds + uv_bounds + range_bounds + sec_bounds

    def make_bounds_mask(self):
        bounds_mask = np.zeros(self.num_params + 5, dtype=bool)
        nc = self.n_components - 1
        xr_base_start = nc
        bounds_mask[0:xr_base_start] = True

        rg_start = xr_base_start + self.num_baseparams

        if self.integral_baseline:
            bounds_mask[rg_start-1] = True      # xr baseline fouling
            bounds_mask[self.pos[6]-1] = True   # uv baseline fouling

        bounds_mask[rg_start:rg_start+nc] = True
        return bounds_mask

    def update_bounds_hook(self, masked_init_params):
        # nothing to do
        pass

    def get_rg_start_index(self):
        return self.pos[2]

    def get_mr_start_index(self):
        return self.pos[6]

    def get_xr_parameter_names(self):
        nc = self.n_components - 1
        xr_names = []
        for k in range(nc):
            xr_names += ["$h_%d$" % k]
        return xr_names

    def get_parameter_names(self):
        n = self.n_components
        _, rg_names, mapping_names, uv_names, mr_names, seccol_names = get_common_parameter_names(n, nc_rg=n-1)
        xr_names = self.get_xr_parameter_names()
        xr_basenames = ["$xb_a$", "$xb_b$"]
        if self.num_baseparams == 3:
            xr_basenames += ["$xb_r$"]
        uv_basenames = ["$L$", "$x_0$", "$k$", "$b$", "$s_1$", "$s_2$", "$diffratio$"]
        if self.num_baseparams == 3:
            uv_basenames += ["$ub_r$"]
        return np.array(xr_names + xr_basenames + rg_names + mapping_names + uv_names + uv_basenames + mr_names + seccol_names + ["$R$"])

    def get_peak_pos_array_list(self, x_array):
        ni = x_array.shape[0]
        nc = self.n_components - 1
        ret_pos_array = np.zeros((nc, ni))
        xr_start = 0
        xr_stop = nc
        rg_start = nc + 2
        rg_stop = nc + 2 + nc
        for i, p in enumerate(x_array):
            xr_params = convert_to_xr_params_hermans(p[xr_start:xr_stop], p[rg_start:rg_stop], p[SEC_START:SEC_STOP], p[SEC_STOP])
            for k in range(nc):
                ret_pos_array[k,i] = xr_params[k,1] + xr_params[k,2]    # i.e., mu + tau
        return ret_pos_array

    def split_get_unified_sec_params(self, params):
        separate_params = self.split_params_simple(params)
        t0, K, rp, m = separate_params[-2]
        R = separate_params[-1]
        return t0, K, rp, m, None, None, None,None, None, None, R

    def compute_xr_peak_position(self, i, xr_params):
        return xr_params[i,1] + xr_params[i,3]
