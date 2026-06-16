"""
    ModelParams.GrmParams.py

    Parameter layout for the GRM (General Rate Model) objective function.

    Column-params layout:
      [Pe, t0, R_p, D_eff, R_0, k_ext_0, R_1, k_ext_1, ..., R_{nc-1}, k_ext_{nc-1}]
    num_col_params = 4 + 2*nc   where nc = n_components - 1 (excluding baseline)

    Copyright (c) 2026, SAXS Team, KEK-PF
"""
import logging
import numpy as np
from .BaselineParams import get_num_baseparams
from molass_legacy._MOLASS.SerialSettings import get_setting
from molass_legacy.Optimizer.BasicOptimizer import AVOID_VANISHING_RATIO
from .ParamsTypeBase import ParamsTypeBase

# Reasonable search ranges for BH / NS sampling
PE_LO_FACTOR   = 0.05    # Pe lower bound = init * PE_LO_FACTOR
PE_HI_FACTOR   = 20.0    # Pe upper bound = init * PE_HI_FACTOR
T0_LO_FACTOR   = 0.50    # t0 lower bound = init * T0_LO_FACTOR
T0_HI_FACTOR   = 1.20    # t0 upper bound = init * T0_HI_FACTOR
RP_LO_FACTOR   = 0.50    # R_p lower bound = init * RP_LO_FACTOR
RP_HI_FACTOR   = 2.0     # R_p upper bound = init * RP_HI_FACTOR
DEFF_LO_FACTOR = 0.01    # D_eff lower bound = init * DEFF_LO_FACTOR
DEFF_HI_FACTOR = 100.0   # D_eff upper bound = init * DEFF_HI_FACTOR
R_LO_FACTOR    = 0.30    # R lower bound  = init * R_LO_FACTOR (min 1.001)
R_HI_FACTOR    = 5.0     # R upper bound  = init * R_HI_FACTOR
KEXT_LO_FACTOR = 0.001   # k_ext lower bound = init * KEXT_LO_FACTOR
KEXT_HI_FACTOR = 1000.0  # k_ext upper bound = init * KEXT_HI_FACTOR


def get_common_parameter_names(nc):
    xr_names      = ["$h_%d$" % k for k in range(nc)]
    rg_names      = ["$R_{g%d}$" % k for k in range(nc)]
    mapping_names = ["$mp_a$", "$mp_b$"]
    uv_names      = ["$uh_%d$" % k for k in range(nc)]
    mr_names      = ["$mr_a$", "$mr_b$"]
    grmcol_names  = ["$Pe$", "$t_0$", "$R_p$", "$D_{eff}$"] + [
        f"$R_{k}$" if j == 0 else f"$k_{{ext,{k}}}$"
        for k in range(nc) for j in range(2)
    ]
    return xr_names, rg_names, mapping_names, uv_names, mr_names, grmcol_names


class GrmParams(ParamsTypeBase):
    """Parameter layout manager for GRM rigorous optimization (G1500)."""

    def __init__(self, n_components):
        self.logger         = logging.getLogger(__name__)
        self.n_components   = n_components
        nc                  = n_components - 1           # pure components (excl. baseline)
        self.num_col_params = 4 + 2 * nc
        self.num_baseparams = get_num_baseparams()
        self.integral_baseline = (self.num_baseparams == 3)
        self.t0_upper_bound = get_setting("t0_upper_bound")
        self.estimator      = None                       # set by get_estimator (optional)

        self.pos = []
        self.pos.append(0)                               # [0] xr_params start
        sep = nc
        self.pos.append(sep)                             # [1] xr_baseparams start
        sep += self.num_baseparams
        self.pos.append(sep)                             # [2] rgs start
        sep += nc
        self.pos.append(sep)                             # [3] mapping (a,b) start
        sep += 2
        self.pos.append(sep)                             # [4] uv_params start
        sep += nc
        self.pos.append(sep)                             # [5] uv_baseparams start
        sep += 5 + self.num_baseparams
        self.pos.append(sep)                             # [6] mappable_range start
        sep += 2
        self.pos.append(sep)                             # [7] end (excl. col params)

        self.num_params = sep
        self.logger.info("GrmParams pos=%s", str(self.pos))

    # ── Model identification ──────────────────────────────────────────────────

    def get_model_name(self):
        return 'GRM'

    # ── Estimator ─────────────────────────────────────────────────────────────

    def get_estimator(self, editor, **kwargs):
        """Return a GrmEstimator for this model."""
        if True:
            from importlib import reload
            import molass_legacy.Estimators.GrmEstimator
            reload(molass_legacy.Estimators.GrmEstimator)
        from molass_legacy.Estimators.GrmEstimator import GrmEstimator
        self.estimator = GrmEstimator(editor, self.n_components)
        return self.estimator

    def set_estimator(self, estimator):
        """Store an externally-created estimator (used by the Result Viewer)."""
        self.logger.info("setting estimator for result viewer")
        self.estimator = estimator

    def get_colparam_bounds(self):
        """Return None; bounds are computed in get_param_bounds from init values."""
        return None

    # ── Parameter splitting ───────────────────────────────────────────────────

    def split_params(self, n_components, decomp_params):
        """Decode the flat decomp_params vector into named sections."""
        p = self.pos
        xr_params      = decomp_params[p[0]:p[1]]
        xr_baseparams  = decomp_params[p[1]:p[2]]
        rg_params      = decomp_params[p[2]:p[3]]
        mapping        = decomp_params[p[3]:p[4]]
        uv_params      = decomp_params[p[4]:p[5]]
        uv_baseparams  = decomp_params[p[5]:p[6]]
        mappable_range = decomp_params[p[6]:p[7]]
        return [xr_params, xr_baseparams, rg_params, mapping,
                uv_params, uv_baseparams, mappable_range]

    def split_params_simple(self, params):
        r = len(params) - self.num_params
        if r == 0:
            decomp_params = params
            seccol_params = None
        else:
            assert r == self.num_col_params, (
                f"GrmParams: expected extra {self.num_col_params} col params, got {r}")
            decomp_params = params[:-self.num_col_params]
            seccol_params = params[-self.num_col_params:]

        self.separate_params = self.split_params(self.n_components, decomp_params) + [seccol_params]
        return self.separate_params

    def split_as_unified_params(self, params):
        return self.split_params_simple(params) + [(None, None)]

    # ── Parameter bounds ──────────────────────────────────────────────────────

    def get_param_bounds(self, params, real_bounds=None):
        init_xr_params, init_xr_baseparams, init_rgs, init_mapping, \
            init_uv_params, init_uv_baseparams, init_mappable_range = \
            self.split_params_simple(params)[0:7]

        xr_h_max = np.max(init_xr_params)
        xr_h_min = xr_h_max * AVOID_VANISHING_RATIO
        m_allow  = 100

        xr_bounds = []
        for h in init_xr_params:
            xr_bounds.append((xr_h_min, xr_h_max * 2))
        for k, v in enumerate(init_xr_baseparams):
            v_allow = max(0.1, abs(v)) * 0.2
            if self.integral_baseline and k == 2:
                bounds = (max(0, v - v_allow), v + v_allow)
            else:
                bounds = (v - v_allow, v + v_allow)
            xr_bounds.append(bounds)

        rg_bounds = [(rg * 0.5, rg * 2) for rg in init_rgs]

        a, b = init_mapping
        mapping_bounds = [(a * 0.8, a * 1.2), (-m_allow, m_allow)]

        uv_h_max = np.max(init_uv_params)
        uv_h_min = uv_h_max * AVOID_VANISHING_RATIO
        uv_bounds = [(uv_h_min, uv_h_max * 2) for _ in init_uv_params]
        for k, v in enumerate(init_uv_baseparams):
            v_allow = max(0.1, abs(v)) * 0.2
            if self.integral_baseline and k == 7:
                bounds = (max(0, v - v_allow), v + v_allow)
            else:
                bounds = (v - v_allow, v + v_allow)
            uv_bounds.append(bounds)

        f, t = init_mappable_range
        dx = (t - f) * 0.1
        range_bounds = [(f - dx, f + dx), (t - dx, t + dx)]

        # ── Column-params bounds ───────────────────────────────────────────────
        if real_bounds is None:
            grmcol = self.split_params_simple(params)[-1]
            Pe, t0, R_p, D_eff = grmcol[0], grmcol[1], grmcol[2], grmcol[3]
            nc = (len(grmcol) - 4) // 2
            colparam_bounds = [
                (Pe   * PE_LO_FACTOR,   Pe   * PE_HI_FACTOR),
                (t0   * T0_LO_FACTOR,   t0   * T0_HI_FACTOR),
                (R_p  * RP_LO_FACTOR,   R_p  * RP_HI_FACTOR),
                (D_eff * DEFF_LO_FACTOR, D_eff * DEFF_HI_FACTOR),
            ]
            for i in range(nc):
                R     = grmcol[4 + 2 * i]
                k_ext = grmcol[4 + 2 * i + 1]
                colparam_bounds.append((max(1.001, R * R_LO_FACTOR), R * R_HI_FACTOR))
                colparam_bounds.append((k_ext * KEXT_LO_FACTOR, k_ext * KEXT_HI_FACTOR))
        else:
            colparam_bounds = list(real_bounds[-self.num_col_params:])

        self.bounds_lengths = [len(b) for b in
                               [xr_bounds, rg_bounds, mapping_bounds,
                                uv_bounds, range_bounds, colparam_bounds]]
        return xr_bounds + rg_bounds + mapping_bounds + uv_bounds + range_bounds + colparam_bounds

    def make_bounds_mask(self):
        bounds_mask = np.zeros(self.num_params + self.num_col_params, dtype=bool)
        nc          = self.n_components - 1
        bounds_mask[0:nc] = True                  # xr_params
        sep = nc + self.num_baseparams
        if self.integral_baseline:
            bounds_mask[sep - 1] = True           # xr baseline fouling
            bounds_mask[self.pos[6] - 1] = True   # uv baseline fouling
        bounds_mask[sep:sep + nc] = True          # rgs
        return bounds_mask

    def get_rg_start_index(self):
        return self.pos[2]

    def get_mr_start_index(self):
        return self.pos[6]

    def get_parameter_names(self):
        nc = self.n_components - 1
        xr_names, rg_names, mapping_names, uv_names, mr_names, grmcol_names = \
            get_common_parameter_names(nc)
        xr_basenames = ["$xb_a$", "$xb_b$"]
        if self.num_baseparams == 3:
            xr_basenames += ["$xb_r$"]
        uv_basenames = ["$L$", "$x_0$", "$k$", "$b$", "$s_1$", "$s_2$", "$diffratio$"]
        if self.num_baseparams == 3:
            uv_basenames += ["$ub_r$"]
        return np.array(
            xr_names + xr_basenames + rg_names + mapping_names +
            uv_names + uv_basenames + mr_names + grmcol_names
        )

    def get_trans_indeces(self):
        # Return indices of the last 4 shared col params + per-comp params
        # Used for "normalised" parameter display (trans = Pe, t0, R_p, D_eff are shared)
        return -4, -3, -2, -1

    def compute_comformance(self, xr_params, rg_params, grmcol_params,
                            poresize_bounds=None):
        """GRM has no pore-size constraint separate from D_eff; always returns 0."""
        return 0

    def split_get_unified_sec_params(self, params):
        """Returns placeholder tuple for compatibility with display utilities."""
        grmcol = self.split_params_simple(params)[-1]
        Pe, t0, R_p, D_eff = grmcol[0], grmcol[1], grmcol[2], grmcol[3]
        return t0, t0, None, None, Pe, None, R_p, D_eff, None, None, None, None, None

    def get_peak_pos_array_list(self, x_array):
        """Return estimated peak positions for dashboard rendering (mean = t0 * R_i)."""
        nc = self.n_components - 1
        pos_array_list = []
        for params in x_array:
            grmcol = params[-self.num_col_params:]
            t0     = grmcol[1]
            trs    = np.array([t0 * grmcol[4 + 2 * i] for i in range(nc)])
            pos_array_list.append(trs)
        return np.array(pos_array_list).T

    # ── GUI inspection helpers ────────────────────────────────────────────────

    def get_params_sheet(self, parent, params, dsets, optimizer, debug=True):
        """Return the parameter inspection sheet for the GUI 'Show Parameters' button."""
        if debug:
            from importlib import reload
            import molass_legacy.ModelParams.GrmParamsSheet
            reload(molass_legacy.ModelParams.GrmParamsSheet)
        from .GrmParamsSheet import GrmParamsSheet
        return GrmParamsSheet(parent, params, dsets, optimizer)
