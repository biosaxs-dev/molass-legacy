"""
    LjEghParams.py

    Copyright (c) 2022-2023, SAXS Team, KEK-PF
"""
import numpy as np
from molass_legacy.KekLib.BasicUtils import Struct
from .EghParams import EghParamsBase
from .SimpleSecParams import initial_guess
from .BaselineParams import get_num_baseparams
from molass_legacy._MOLASS.SerialSettings import get_setting
from SecTheory.SecEstimator import NUM_SEC_PARAMS
from SecTheory.LanJorgensonEgh import convert_to_xr_params_lj

RG_UPPER_BOUND = get_setting("RG_UPPER_BOUND")

class LjEghParams(EghParamsBase):
    def __init__(self, n_components):
        self.n_components = n_components
        self.num_baseparams = get_num_baseparams()
        self.integral_baseline = self.num_baseparams == 3
        self.use_K = False
        self.advanced = True
        self.poresize = None        # to be removed
        self.poreexponent = None    # to be removed

        nc = n_components - 1
        self.pos = []
        self.pos.append(0)      # [0] xr_params
        sep = 3*nc
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
        return "LjEghParams(nc=%d)" % (self.n_components - 1)

    def get_estimator(self, editor, debug=True):
        if debug:
            from importlib import reload
            import Estimators.LjEghEstimator
            reload(Estimators.LjEghEstimator)
        from Estimators.LjEghEstimator import LjEghEstimator
        return LjEghEstimator(editor)

    def split_params(self, n, params):
        if self.num_params != len(params):
            raise ValueError("len(params)=%d != %d, which is calculated from number of components %d." % (len(params), self.num_params, n))

        ret_params = []
        for p, q in zip(self.pos[:-1], self.pos[1:]):
            ret_params.append(params[p:q])

        ret_params[0] = ret_params[0].reshape((n-1,3))  # xr_params

        return ret_params

    def split_params_simple(self, params, convert=True):
        req_params = params[0:-NUM_SEC_PARAMS]
        sec_params = params[-NUM_SEC_PARAMS:]
        Npc, tI = sec_params[[0,2]]
        split_params = self.split_params(self.n_components, req_params)
        if convert:
            xr_params = convert_to_xr_params_lj(split_params[0], tI, Npc)
        else:
            xr_params = split_params[0]

        self.separate_params = [xr_params] + split_params[1:] + [sec_params]
        return self.separate_params

    def split_as_unified_params(self, params, **kwargs):
        return self.split_params_simple(params, **kwargs)

    def compute_init_guess(self, xr_params):
        temp_guess = initial_guess(xr_params)     # uses only xr_params[:,1]
        Np = get_setting("numplates_pc")
        return Struct(  params=temp_guess.params,
                                                    #     Ti         Np
                        bounds=temp_guess.bounds + ((-2000, 0), (Np/1.5, Np*1.5)),  # (9600, 21600) for Np=14400
                        )

    def get_xr_param_bounds(self, temp_xr_params, Ti, Np):
        xr_h_max = np.max(temp_xr_params[:,0])
        m_ = np.average(temp_xr_params[:,1])
        s_ = (m_ - Ti)/np.sqrt(Np)
        m_allow = np.max(temp_xr_params[:,1])*0.2
        xr_bounds = []
        for h, m, t in temp_xr_params:      # temp_xr_params is not converted
            xr_bounds.append((0, xr_h_max*1.1))                     # h
            xr_bounds.append((max(0, m - m_allow), m + m_allow))    # m
            xr_bounds.append((s_*0.5, s_*1.5))                      # s
        return xr_bounds

    def get_extended_bounds(self, init_xr_params):
        bounds = super().get_extended_bounds(init_xr_params)
        return bounds

    def make_bounds_mask(self):
        bounds_mask = np.zeros(self.num_params + NUM_SEC_PARAMS, dtype=bool)
        nc = self.n_components - 1
        xr_base_start = 3*nc
        bounds_mask[0:xr_base_start] = np.array([True, True, False]*nc)      # 

        rg_start = xr_base_start + self.num_baseparams
        self.rg_start = rg_start

        if self.integral_baseline:
            bounds_mask[rg_start-1] = True      # xr baseline fouling
            bounds_mask[self.pos[6]-1] = True   # uv baseline fouling

        bounds_mask[rg_start:rg_start+nc] = True
        return bounds_mask

    def update_bounds_hook(self, masked_init_params):
        # this overrides  EghParamsBase.update_bounds_hook which modifies sigma bounds unnecessary for this class
        nc = self.n_components - 1
        rg_start = nc*2             # note that params here are masked
        for i in range(nc):
            j = rg_start + i
            masked_init_params[j] = min(RG_UPPER_BOUND, masked_init_params[j])

    def get_xr_parameter_names(self):
        nc = self.n_components - 1
        xr_names = []
        for k in range(nc):
            xr_names += ["$h_%d$" % k, r"$\mu_%d$" % k, r"$\sigma_%d$" % k]
        return xr_names

    def get_parameter_names(self):
        nc = self.n_components - 1

        xr_names = self.get_xr_parameter_names()
        xr_basenames = ["$xb_a$", "$xb_b$"]
        if self.num_baseparams == 3:
            xr_basenames += ["$xb_r$"]
        rg_names = ["$R_{g%d}$" % k for k in range(nc)]
        mapping_names = ["$mp_a$", "$mp_b$"]
        uv_names = ["$uh_%d$" % k for k in range(nc)]
        uv_basenames = ["$L$", "$x_0$", "$k$", "$b$", "$s_1$", "$s_2$", "$diffratio$"]
        if self.num_baseparams == 3:
            uv_basenames += ["$ub_r$"]
        mr_names = ["$mr_a$", "$mr_b$"]
        seccol_names = ["$N_{pc}$", "$r_p$", "$t_I$", "$t_0$", "$P$", "$m$"]    # task: unify with EghParams.get_common_parameter_names()

        return np.array(xr_names + xr_basenames + rg_names + mapping_names + uv_names + uv_basenames + mr_names + seccol_names)

    def get_rg_start_index(self):
        return self.pos[2]

    def get_mr_start_index(self):
        return self.pos[6]

    def get_peak_pos_array_list(self, x_array):
        n = self.n_components - 1
        pos_array_list = []
        for k in range(n):
            pos_array_list.append(x_array[:,3*k+1])     # mu in (h, mu, sigma)
        return pos_array_list

    def get_params_sheet(self, parent, params, dsets, optimizer):
        from .LjEghParamsSheet import LjEghParamsSheet
        return LjEghParamsSheet(parent, params, dsets, optimizer)

    def split_get_unified_sec_params(self, params):
        separate_parame = self.split_params_simple(params)
        t0, K, rp, m = separate_parame[-2]
        Ti, Np = separate_parame[-1]
        return t0, K, rp, m, None, None, None,None, Ti, Np, None
