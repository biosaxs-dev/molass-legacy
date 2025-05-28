"""
    EghParams.py

    Copyright (c) 2022-2024, SAXS Team, KEK-PF
"""
import numpy as np
from molass_legacy._MOLASS.SerialSettings import get_setting
# from SecTheory.RetensionTime import make_initial_guess
from molass_legacy.SecTheory.SecEstimator import guess_initial_secparams, NUM_SEC_PARAMS
from .SimpleSecParams import SimpleSecParams, sec_comformance
from .BaselineParams import get_num_baseparams
from molass_legacy.Optimizer.BasicOptimizer import AVOID_VANISHING_RATIO

def get_common_parameter_names(n, nc_rg=None, seccol=True):
    if nc_rg is None:
        nc_rg = n
    else:
        # used in RtEmgParams
        pass

    nc = n - 1

    xr_names = []
    for k in range(nc):
        xr_names += ["$h_%d$" % k, r"$\mu_%d$" % k, r"$\sigma_%d$" % k, r"$\tau_%d$" % k]

    rg_names = ["$R_{g%d}$" % k for k in range(nc_rg)]

    mapping_names = ["$mp_a$", "$mp_b$"]

    uv_names = ["$uh_%d$" % k for k in range(nc)]

    mr_names = ["$mr_a$", "$mr_b$"]

    if seccol:
        seccol_names = ["$N_{pc}$", "$r_p$", "$t_I$", "$t_0$", "$P$", "$m$"]
    else:
        seccol_names = []

    return xr_names, rg_names, mapping_names, uv_names, mr_names, seccol_names

class EghParamsBase:
    def __init__(self, n_components, poresize, poreexponent, sec_class=None):
        self.n_components = n_components
        self.num_baseparams = get_num_baseparams()
        self.integral_baseline = self.num_baseparams == 3
        self.poresize = poresize
        if poreexponent == 0:
            # workaround for set_setting("poreexponent", None) not working
            poreexponent = None
        self.poreexponent = poreexponent
        if sec_class is None:
            sectype = SimpleSecParams(poresize, poreexponent)
        else:
            sectype = sec_class(poresize, poreexponent)
        self.init_method = sectype.init_method
        # self.conf_method = sectype.conf_method
        self.estm_method = sectype.estm_method
        self.nump_adjust = sectype.nump_adjust
        self.use_K = False

    def __str__(self):
        name = "Egh%sParams" % ("Advansed" if self.advanced else "")
        ps, px = [str(v) if v is None else "%g" % v for v in [self.poresize, self.poreexponent]]
        return "%s(nc=%d, ps=%s, px=%s)" % (name, self.n_components, ps, px)

    def get_model_name(self):
        return 'EGH'

    def compute_init_guess(self, *args):
        return self.init_method(*args)

    def compute_comformance(self, *args, **kwargs):
        # return self.conf_method(*args, **kwargs)       # bug in Python 3.10.7?
        return sec_comformance(*args, **kwargs)          # bug fix

    def get_seccol_params_for_disp(self, seccol_params):
        # recosider whether this is required
        return seccol_params

    def estimate_conformance_params(self, *args):
        return self.estm_method(*args)

    def get_xr_param_bounds(self, temp_xr_params, Ti, Np):      # Ti, Np not used
        xr_h_max = np.max(temp_xr_params[:,0])
        xr_h_min = xr_h_max*AVOID_VANISHING_RATIO
        m_allow = np.max(temp_xr_params[:,1])*0.2
        s_allow = np.average(np.abs(temp_xr_params[:,2]))*1.0
        xr_bounds = []
        for h, m, s, t in temp_xr_params:
            s = abs(s)
            xr_bounds.append((xr_h_min, xr_h_max*2))                # h
            xr_bounds.append((max(0, m - m_allow), m + m_allow))    # m
            xr_bounds.append((max(3, s - s_allow), s + s_allow))    # s
            xr_bounds.append((0, s + s_allow))                      # t
        return xr_bounds

    def get_extended_bounds(self, init_sec_params):
        Npc, rp, tI, t0, P, m = init_sec_params
        t0_upper_bound = get_setting("t0_upper_bound")
        if t0_upper_bound is None:
            t0_upper_bound = t0+500

        bounds = [(Npc-1, Npc+1), (rp-1, rp+1), (tI-500, tI+200), (t0-500, t0_upper_bound), (1000, 10000), (1, 3)]
        return bounds

    def get_param_bounds(self, params):
        split_params = self.split_params_simple(params, convert=False)      # note that init_xr_params will not be converted
        init_xr_params, init_xr_baseparams, init_rgs, init_mapping, init_uv_params, init_uv_baseparams, init_mappable_range, init_sec_params = split_params
        Npc, tI = init_sec_params[[0,2]]

        if self.advanced:
            temp_xr_params = init_xr_params
        else:
            temp_xr_params = np.vstack([init_xr_params, init_xr_baseparams])
        m_allow = np.max(temp_xr_params[:,1])*0.2

        xr_bounds = self.get_xr_param_bounds(temp_xr_params, tI, Npc)   # note that temp_xr_params is not converted

        if self.advanced:
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
        if self.advanced:
            temp_uv_params = init_uv_params
        else:
            temp_uv_params = np.concatenate([init_uv_params, init_uv_baseparams])
        uv_h_max = np.max(temp_uv_params)
        uv_h_min = uv_h_max*AVOID_VANISHING_RATIO
        uv_bounds = [(uv_h_min, uv_h_max*2) for h in temp_uv_params]
        if self.advanced:
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

        extended_bounds = self.get_extended_bounds(init_sec_params)

        self.bounds_lengths = [len(b) for b in [xr_bounds, rg_bounds, mapping_bounds, uv_bounds, range_bounds, extended_bounds]]
        return xr_bounds + rg_bounds + mapping_bounds + uv_bounds + range_bounds + extended_bounds

    def get_sigma_bounds(self, x, debug=True):
        # task: is this really necessary?
        xr_params = self.split_params_simple(x)[0]
        sigma = xr_params[:,2]
        tau = xr_params[:,3]
        max_sigma = np.max(np.sqrt(sigma**2 + tau**2))
        min_sigma = max_sigma/4
        return min_sigma, max_sigma*1.5

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
        for i in range(self.n_components - 1):
            j = 3*i + 2     # sigma index
            masked_init_params[j] = max(1, masked_init_params[j])   # sigma should not be less than 1

    def get_peak_pos_array_list(self, x_array):
        n = self.n_components - 1
        pos_array_list = []
        for k in range(n):
            pos_array_list.append(x_array[:,4*k+1])     # mu in (h, mu, sigma, tau)
        return pos_array_list

    def get_params_sheet(self, parent, params, dsets, optimizer, debug=True):
        if debug:
            from importlib import reload
            import molass_legacy.ModelParams.EghParamsSheet
            reload(molass_legacy.ModelParams.EghParamsSheet)
        from .EghParamsSheet import EghParamsSheet
        return EghParamsSheet(parent, params, dsets, optimizer)

    def get_adjuster(self, debug=True):
        if debug:
            import molass_legacy.ModelParams.EghAdjuster
            from importlib import reload
            reload(molass_legacy.ModelParams.EghAdjuster)
        from .EghAdjuster import EghAdjuster
        return EghAdjuster()

    def compute_xr_peak_position(self, i, xr_params):
        return xr_params[i,1]

class EghParams(EghParamsBase):
    def __init__(self, n_components, poresize=None, poreexponent=None, sec_class=None):
        EghParamsBase.__init__(self, n_components, poresize, poreexponent, sec_class=sec_class)
        self.num_params = 6*n_components + 4
        self.advanced = False

    def split_params(self, n, p):
        if self.num_params != len(p):
            raise ValueError("len(params)=%d != %d, which is calculated from number of components %d." % (len(p), self.num_params, n))

        sep = 4*(n-1)
        xr_params = p[0:sep].reshape((n-1,4))
        xr_baseparams = p[sep:sep+4]
        sep += 4
        rgs = p[sep:sep+n]
        sep += n
        mapping = p[sep:sep+2]
        sep = sep+2
        uv_params = p[sep:sep+n-1]
        sep += n-1
        uv_baseparams = p[sep:sep+1]    # i.e., size one vector
        sep += 1
        mappable_range = p[sep:]

        return xr_params, xr_baseparams, rgs, mapping, uv_params, uv_baseparams, mappable_range

    def split_params_simple(self, params, convert=False):       # convert is not relevant in free EGH
        n = self.n_components
        r = len(params) - self.num_params
        if r > 0:
            decomp_params = params[:-r]
            seccol_params = params[-r:]
        else:
            decomp_params = params
            seccol_params = None
        self.separate_params = self.split_params(n, decomp_params) + [seccol_params]
        return self.separate_params

    def make_bounds_mask(self):
        bounds_mask = np.zeros(self.num_params, dtype=bool)
        n = self.n_components
        xr_baseparams_mask = [False]*4      # no bounds for baseline params
        bounds_mask[0:n*5] = np.array([True, True, True, False]*(n-1) + xr_baseparams_mask
            + [True]*(n-1) + [False])       # no bounds for the last rg of baseline
        return bounds_mask

    def get_parameter_names(self):
        xr_names, rg_names, mapping_names, uv_names, mr_names, seccol_names = get_common_parameter_names(self.n_components)
        xr_basenames = ["$xb_h$", "$xb_a$", "$xb_b$", "$\tau_u$"]
        uv_basenames = ["$uh_b$"]
        return np.array(xr_names + xr_basenames + rg_names + mapping_names + uv_names + uv_basenames + mr_names + seccol_names)

    def get_rg_start_index(self):
        return 4*self.n_components

    def get_mr_start_index(self):
        return 6*self.n_components + 2

class EghAdvansedParams(EghParamsBase):
    def __init__(self, n_components, poresize=None, poreexponent=None, sec_class=None, baseline_rg=True):
        EghParamsBase.__init__(self, n_components, poresize, poreexponent, sec_class=sec_class)
        self.advanced = True
        self.baseline_rg = baseline_rg

        nc = n_components - 1

        self.pos = []
        self.pos.append(0)      # [0] xr_params
        sep = 4*nc
        self.pos.append(sep)    # [1] xr_baseparams
        sep += self.num_baseparams
        self.pos.append(sep)    # [2] rgs
        if self.baseline_rg:
            # for backward compatibility toward _MOLASS 2.0.0*
            sep += nc + 1                   # note that, due to historical reasons, len(rgs) = n_components,
                                            # i.e., the extra rg from molass_legacy.Baseline is included, which is not used
        else:
            # for _MOLASS 2.1.0* or later
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

    def get_estimator(self, editor, debug=True):
        if debug:
            from importlib import reload
            import molass_legacy.Estimators.EghEstimator
            reload(molass_legacy.Estimators.EghEstimator)
        from molass_legacy.Estimators.EghEstimator import EghEstimator
        return EghEstimator(editor)

    def split_params(self, n, params):
        if self.num_params != len(params):
            raise ValueError("len(params)=%d != %d, which is calculated from number of components %d." % (len(params), self.num_params, n))

        ret_params = []
        for p, q in zip(self.pos[:-1], self.pos[1:]):
            ret_params.append(params[p:q])

        ret_params[0] = ret_params[0].reshape((n-1,4))  # xr_params

        return ret_params

    def split_params_simple(self, params, convert=False):       # convert is not relevant in free EGH
        n = self.n_components
        r = len(params) - self.num_params
        if r > 0:
            decomp_params = params[:-r]
            seccol_params = params[-r:]
        else:
            decomp_params = params
            seccol_params = None
        return self.split_params(n, decomp_params) + [seccol_params]

    def split_as_unified_params(self, params, **kwargs):
        return self.split_params_simple(params, **kwargs) + [(None, None)]

    def make_bounds_mask(self):
        bounds_mask = np.zeros(self.num_params + NUM_SEC_PARAMS, dtype=bool)
        n = self.n_components
        xr_base_start = 4*(n-1)
        bounds_mask[0:xr_base_start] = np.array([True, True, True, False]*(n-1))

        rg_start = xr_base_start + self.num_baseparams

        if self.integral_baseline:
            bounds_mask[rg_start-1] = True      # xr baseline fouling
            bounds_mask[self.pos[6]-1] = True   # uv baseline fouling

        bounds_mask[rg_start:rg_start+n] = np.array([True]*(n-1) + [False])    # no bounds for the last rg of baseline
        return bounds_mask

    def get_parameter_names(self):
        xr_names, rg_names, mapping_names, uv_names, mr_names, seccol_names = get_common_parameter_names(self.n_components)
        xr_basenames = ["$xb_a$", "$xb_b$"]
        if self.num_baseparams == 3:
            xr_basenames += ["$xb_r$"]
        uv_basenames = ["$L$", "$x_0$", "$k$", "$b$", "$s_1$", "$s_2$", "$diffratio$"]
        if self.num_baseparams == 3:
            uv_basenames += ["$ub_r$"]
        return np.array(xr_names + xr_basenames + rg_names + mapping_names + uv_names + uv_basenames + mr_names + seccol_names)

    def get_rg_start_index(self):
        return self.pos[2]

    def get_mr_start_index(self):
        return self.pos[6]

    def split_get_unified_sec_params(self, params):
        t0, K, rp, m = self.split_params_simple(params)[-1]
        return (t0, K, rp, m) + (None,)*7

def construct_egh_params_type(n_components, sec_class=None, baseline_rg=True):
    from molass_legacy._MOLASS.SerialSettings import get_setting

    poresize = get_setting("poresize")
    poreexponent = get_setting("poreexponent")
    uv_basemodel = get_setting("uv_basemodel")
    if uv_basemodel == 1:
        params_type = EghAdvansedParams(n_components, poresize, poreexponent, sec_class=sec_class, baseline_rg=baseline_rg)
    else:
        params_type = EghParams(n_components, poresize, poreexponent, sec_class=sec_class)
    return params_type
