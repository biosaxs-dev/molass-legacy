"""
    CedmParams.py — parameter class for Constrained-EDM (G2020)

    Parameter layout (all but colparams inside split_params):
        xr_params       : (nc × 3)  — [a_k, b_k, cinj_k]  per component
        xr_baseparams   : num_baseparams
        rg_params       : nc
        mapping         : (a_mp, b_mp)
        uv_params       : nc
        uv_baseparams   : 5 + num_baseparams
        mappable_range  : (c, d)
    Appended at the end (outside split_params):
        cedm_colparams  : [t0_sh, u_sh, e_sh, Dz_sh]   (4 values)

    Copyright (c) 2024-2025, SAXS Team, KEK-PF
"""
import logging
import numpy as np
from .BaselineParams import get_num_baseparams
from molass_legacy._MOLASS.SerialSettings import get_setting
from molass_legacy.Models.RateTheory.EDM import edm_impl, MIN_CINJ, MAX_CINJ
from molass_legacy.Optimizer.BasicOptimizer import AVOID_VANISHING_RATIO

NUM_COL_PARAMS = 4       # t0_sh, u_sh, e_sh, Dz_sh
NUM_ELEMENT_PARAMS = 3   # a, b, cinj  per component


class CedmParams:
    """Parameter type for Constrained-EDM rigorous optimisation (G2020).

    The four column parameters (t0, u, e, Dz) are shared across all
    components; only the per-component physical parameters (a = K_SEC, b, cinj)
    are free per component.
    """

    def __init__(self, n_components):
        self.logger = logging.getLogger(__name__)
        self.n_components = n_components
        self.num_baseparams = get_num_baseparams()
        self.integral_baseline = self.num_baseparams == 3
        self.use_K = False

        nc = n_components - 1

        self.pos = []
        self.pos.append(0)                          # [0] xr_params start
        sep = nc * NUM_ELEMENT_PARAMS
        self.pos.append(sep)                        # [1] xr_baseparams start
        sep += self.num_baseparams
        self.pos.append(sep)                        # [2] rg_params start
        sep += nc
        self.pos.append(sep)                        # [3] mapping start
        sep += 2
        self.pos.append(sep)                        # [4] uv_params start
        sep += nc
        self.pos.append(sep)                        # [5] uv_baseparams start
        sep += 5 + self.num_baseparams
        self.pos.append(sep)                        # [6] mappable_range start
        sep += 2
        self.pos.append(sep)                        # [7] end (excl. colparams)

        self.num_params = sep
        self.logger.info("CedmParams pos=%s", str(self.pos))

    def __str__(self):
        return "CedmParams(nc=%d)" % (self.n_components)

    def get_model_name(self):
        return 'CEDM'

    def set_x(self, x):
        self.x = x

    # ------------------------------------------------------------------
    # parameter splitting
    # ------------------------------------------------------------------

    def split_params(self, n, params):
        if self.num_params != len(params):
            raise ValueError(
                "len(params)=%d != %d (expected for %d components)"
                % (len(params), self.num_params, n)
            )
        nc = self.n_components - 1
        ret_params = []
        for k, (p, q) in enumerate(zip(self.pos[:-1], self.pos[1:])):
            if k == 0:
                chunk = params[p:q].reshape((nc, NUM_ELEMENT_PARAMS))
            else:
                chunk = params[p:q]
            ret_params.append(chunk)
        return ret_params

    def split_params_simple(self, params):
        """Split full parameter vector.

        Returns
        -------
        list of 8 elements:
            [xr_params (nc×3), xr_baseparams, rg_params, mapping,
             uv_params, uv_baseparams, mappable_range, cedm_colparams]
        """
        decomp_params = params[:-NUM_COL_PARAMS]
        cedm_colparams = params[-NUM_COL_PARAMS:]
        self.separate_params = (
            self.split_params(self.n_components, decomp_params) + [cedm_colparams]
        )
        return self.separate_params

    def split_as_unified_params(self, params, **kwargs):
        raise NotImplementedError("CedmParams.split_as_unified_params is not supported")

    # ------------------------------------------------------------------
    # bounds
    # ------------------------------------------------------------------

    def get_xr_per_comp_bounds(self, xr_params_abc):
        """Return per-component bounds: [(a_lo, a_hi), (b_lo, b_hi), (cinj_lo, cinj_hi)] × nc.

        Bounds mirror EdmOptimizer.py conventions, but must be finite (BasicOptimizer
        converts them to a numpy array for parameter scaling):
        - a  (K_SEC): lower-bounded at near-zero; upper cap 5.0 is well above the
          physically expected range [0, 1] and above observed values (e.g. a ≈ 1.63).
        - b: wide symmetric range that comfortably covers extreme values (e.g. b ≈ -30)
          that arise when L-BFGS-B finds flexible asymmetric curve shapes.
        """
        cinj_max = np.max(xr_params_abc[:, 2])       # cinj is column 2
        cinj_min = cinj_max * AVOID_VANISHING_RATIO
        bounds = []
        for a_k, b_k, cinj_k in xr_params_abc:
            bounds.append((0.0001, 5.0))                        # a  (K_SEC; finite upper for BasicOptimizer)
            bounds.append((-50.0, 50.0))                        # b  (wide; covers fitted ≈ ±31)
            bounds.append((max(MIN_CINJ, cinj_min), min(MAX_CINJ, cinj_max * 2)))  # cinj
        return bounds

    def get_cedm_col_bounds(self, cedm_colparams):
        """Return bounds for the 4 shared column parameters."""
        t0, u, e, Dz = cedm_colparams
        return [
            (-500.0, 1000.0),      # t0_sh
            (0.00001, 50.0),       # u_sh
            (0.001, 1.0),          # e_sh
            (0.001, 40.0),         # Dz_sh
        ]

    def get_param_bounds(self, params):
        """Return one (lo, hi) pair for every element of params."""
        (init_xr_params_abc, init_xr_baseparams, init_rgs, init_mapping,
         init_uv_params, init_uv_baseparams, init_mappable_range,
         init_cedm_colparams) = self.split_params_simple(params)

        m_allow = 100

        xr_bounds = self.get_xr_per_comp_bounds(init_xr_params_abc)

        # XR baseline
        for k, v in enumerate(init_xr_baseparams):
            v_allow = max(0.1, abs(v)) * 0.2
            if self.integral_baseline and k == 2:
                bounds = (max(0, v - v_allow), v + v_allow)
            else:
                bounds = (v - v_allow, v + v_allow)
            xr_bounds.append(bounds)

        rg_bounds = [(rg * 0.5, rg * 2) for rg in init_rgs]

        a_mp, b_mp = init_mapping
        mapping_bounds = [(a_mp * 0.8, a_mp * 1.2), (-m_allow, m_allow)]

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

        col_bounds = self.get_cedm_col_bounds(init_cedm_colparams)

        all_bounds = (
            xr_bounds + rg_bounds + mapping_bounds + uv_bounds
            + range_bounds + col_bounds
        )
        self.bounds_lengths = [
            len(b) for b in [xr_bounds, rg_bounds, mapping_bounds,
                              uv_bounds, range_bounds, col_bounds]
        ]
        self.logger.info("bounds_lengths=%s", str(self.bounds_lengths))
        return all_bounds

    def split_bounds(self, bounds):
        separate_bounds = []
        offset = 0
        for k, length in enumerate(self.bounds_lengths):
            next_offset = offset + length
            sep_bounds = bounds[offset:next_offset]
            if k == 0:
                # reshape xr_bounds to (nc, 3) for clarity
                nc = self.n_components - 1
                sep_bounds = [
                    sep_bounds[i * NUM_ELEMENT_PARAMS: (i + 1) * NUM_ELEMENT_PARAMS]
                    for i in range(nc)
                ]
            separate_bounds.append(sep_bounds)
            offset = next_offset
        return separate_bounds

    def update_bounds_hook(self, masked_init_params):
        pass   # nothing to do

    def make_bounds_mask(self):
        total = self.num_params + NUM_COL_PARAMS
        bounds_mask = np.zeros(total, dtype=bool)
        nc = self.n_components - 1
        bounds_mask[0: nc * NUM_ELEMENT_PARAMS] = True   # xr_params (a, b, cinj)
        if self.integral_baseline:
            sep = nc * NUM_ELEMENT_PARAMS + self.num_baseparams
            bounds_mask[sep - 1] = True                  # xr baseline fouling
            bounds_mask[self.pos[6] - 1] = True          # uv baseline fouling
        # mask the 4 shared column params at the very end
        bounds_mask[-NUM_COL_PARAMS:] = True
        return bounds_mask

    # ------------------------------------------------------------------
    # SEC conformance
    # ------------------------------------------------------------------

    def compute_comformance(self, xr_params_abc, rg_params, cedm_colparams, **kwargs):
        """SEC conformance for CEDM.

        Since the column parameters are constrained to be shared, the
        conformance is inherently enforced by the model structure.  We
        return the best possible (lowest) conformance value.
        """
        from molass_legacy.SecTheory.ColumnConstants import SECCONF_LOWER_BOUND
        return SECCONF_LOWER_BOUND

    # ------------------------------------------------------------------
    # index helpers
    # ------------------------------------------------------------------

    def get_rg_start_index(self):
        return self.pos[2]

    def get_mr_start_index(self):
        return self.pos[6]

    # ------------------------------------------------------------------
    # parameter names (for GUI / logging)
    # ------------------------------------------------------------------

    def get_parameter_names(self):
        nc = self.n_components - 1
        xr_names = []
        for k in range(nc):
            xr_names += [f"$a_{k}$", f"$b_{k}$", f"cinj_{k}"]

        rg_names = [f"$R_{{g{k}}}$" for k in range(nc)]
        mapping_names = ["$mp_a$", "$mp_b$"]
        uv_names = [f"$uh_{k}$" for k in range(nc)]
        uv_base = ["$L$", "$x_0$", "$k$", "$b$", "$s_1$", "$s_2$", "$diffratio$"]
        if self.num_baseparams == 3:
            uv_base += ["$ub_r$"]
        xr_base = ["$xb_a$", "$xb_b$"]
        if self.num_baseparams == 3:
            xr_base += ["$xb_r$"]
        mr_names = ["$mr_a$", "$mr_b$"]
        col_names = ["$t0_{sh}$", "$u_{sh}$", "$e_{sh}$", "$Dz_{sh}$"]
        return np.array(
            xr_names + xr_base + rg_names + mapping_names
            + uv_names + uv_base + mr_names + col_names
        )

    # ------------------------------------------------------------------
    # peak positions (used by SEC conformance checks)
    # ------------------------------------------------------------------

    def get_peak_pos_array_list(self, x_array):
        nc = self.n_components - 1
        x = self.x
        pos_array_list = []
        for params in x_array:
            xr_params_abc = params[: nc * NUM_ELEMENT_PARAMS].reshape((nc, NUM_ELEMENT_PARAMS))
            cedm_colparams = params[-NUM_COL_PARAMS:]
            t0_sh, u_sh, e_sh, Dz_sh = cedm_colparams
            pos = []
            for a_k, b_k, cinj_k in xr_params_abc:
                cy = edm_impl(x, t0_sh, u_sh, a_k, b_k, e_sh, Dz_sh, cinj_k)
                cy = np.nan_to_num(cy, nan=0.0, posinf=0.0, neginf=0.0)
                j = np.argmax(cy)
                pos.append(x[j])
            pos_array_list.append(pos)
        return np.array(pos_array_list).T

    # ------------------------------------------------------------------
    # GUI stubs (not needed for in-process optimisation)
    # ------------------------------------------------------------------

    def get_estimator(self, editor, debug=False):
        """Stub — CEDM uses RigorousCedmParams for init, not a legacy estimator."""
        return None

    def get_adjuster(self, debug=True):
        from .StcAdjuster import StcAdjuster
        return StcAdjuster()

    def get_params_sheet(self, parent, params, dsets, optimizer, debug=True):
        raise NotImplementedError("CedmParams GUI sheet not yet implemented")

    def get_paramslider_info(self, devel=True):
        raise NotImplementedError("CedmParams slider info not yet implemented")

    def get_trans_indeces(self):
        raise NotImplementedError("CedmParams.get_trans_indeces is not supported (use_K=False)")
