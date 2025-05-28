"""
    BasicOptimizer.py

    Copyright (c) 2021-2025, SAXS Team, KEK-PF
"""
import logging
import numpy as np
from bisect import bisect_right
from time import time
import numpy as np
from bisect import bisect_right
from molass_legacy.KekLib.ExceptionTracebacker import log_exception
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.Baseline.Constants import SLOPE_SCALE
from molass_legacy.DataStructure.SvdDenoise import get_denoised_data
from molass_legacy._MOLASS.SerialSettings import get_setting
from .StateSequence import save_opt_params
from .RgDiscreteness import compute_rg_discreteness
from .MwIntegrity import compute_mw_integrity_impl
from .FvSynthesizer import synthesize
from .OptLrfInfo import OptLrfInfo
from molass_legacy._MOLASS.Version import is_developing_version
from .ValidComponents import ValidComponents

USE_COLUMN_INTERP = True

from molass_legacy.GuinierTools.GuinierDeviation import USE_NORMALIZED_RMSD_FOR_RGCURVES

USE_NORMALIZED_RMSD = True
USE_FROBENIUS_XDIFFMAX = False
USE_FEATURE_DEVIATION = False
USE_JSD = False
if USE_NORMALIZED_RMSD:
    from molass_legacy.Distance.NormalizedRmsd import normalized_rmsd
elif USE_FROBENIUS_XDIFFMAX:
    from molass_legacy.Distance.FrobeniusXdiffmax import frobenius_xdiffmax
elif USE_FEATURE_DEVIATION:
    from molass_legacy.Distance.DeviationMeasure import feature_deviation
elif USE_JSD:
    from molass_legacy.Distance.JensenShannon import deformed_jsd

USE_BOUNDS = True
AVOID_VANISHING_RATIO = 0.02    # minimum scale ratio against the max scale

PENALTY_SCALE = 1e3
UV_XR_RATIO_ALLOW = 0.5
UV_XR_RATIO_SCALE = 100
WEAK_PENALTY_SCALE = 0.01
SUPERIOR_2D_LRF_ALLOW = 0.1
SLOPE_ALLOWANCE = 0.05
INTERCEPT_ALLOWANCE = 0.05
SCALE_MAX_RATIO = 1.5
PARAMS_SCALE = 10
IMMEDIATE_KNOWN_BEST = False
BAD_PARAMS_RETURN = 1e8
SCORE_DEV_ALLOW = 0.25
SCORE_DEV_LIMIT = 0.5
MEAN_WEIGHT = 0.2
MAX_WEIGHT = 0.6
DEV_WEIGHT = 0.2
EVAL_PEAK_DEVIATION = False
PRIMARY_PEAK_DEV_ALLOW = 0.01
KRATKY_SMOOTHNESS_BAD_VALUE = 1.0
USE_COMPOSED_XR_COMPONENTS = False
USE_COMPOSED_UV_COMPONENTS = False
COERCE_BOUNDED_BQ = True
USE_RGCURVE_DEVIATION = True
ADJUST_2D_TARGET = 1

class BasicOptimizer:
    def __init__(self, dsets, n_components, params_type, kwargs):
        self.NUM_MAJOR_SCORES = get_setting("NUM_MAJOR_SCORES")

        self.devel = is_developing_version()
        self.logger = logging.getLogger(__name__)

        self.dsets = dsets
        self.n_components = n_components
        self.params_type = params_type
        self.use_K = params_type.use_K
        if self.use_K:
            self.nj, self.mej, self.tj, self.mpj = params_type.get_trans_indeces()

        for_split_only = kwargs.pop("for_split_only", False)
        if for_split_only:
            # as used in test_6690_BasinHopping.py
            return

        self.uv_base_curve = kwargs.pop("uv_base_curve", None)
        self.xr_base_curve = kwargs.pop("xr_base_curve", None)
        self.qvector = kwargs.pop("qvector", None)
        self.qzeros = np.zeros(len(self.qvector))
        self.wvector = kwargs.pop("wvector", None)  # lvector in sd
        self.uv_index = bisect_right(self.wvector, get_setting("absorbance_picking"))
        self.xr_index = bisect_right(self.qvector, get_setting("intensity_picking"))
        self.shm = kwargs.pop("shared_memory", None)
        if False:
            import inspect
            for frm in inspect.stack()[1:]:
                self.logger.info("optimizer: %s %s (%d)", frm.filename, frm.function, frm.lineno)

        self.logger.info("optimizer has been created with params_type=%s", str(self.params_type))

        self.separate_eoii = get_setting("separate_eoii")
        self.separate_eoii_type = get_setting("separate_eoii_type")
        self.separate_eoii_flags = get_setting("separate_eoii_flags")
        self.num_pure_components = self.n_components - (2 if self.separate_eoii else 1)

        composite = kwargs.pop("composite", None)
        if composite is None:
            from .CompositeInfo import CompositeInfo
            composite = CompositeInfo(n_components)
        self.composite = composite

        extra_num_components = self.composite.get_extra_num_components()
        self.num_pure_components = self.n_components - 1 - extra_num_components

        (xr_curve, xrD), rg_curve, (uv_curve, uvD) = dsets

        if USE_JSD:
            # negetive values are not allowed in scipy.spatial.distance.jensenshannon
            for curve in xr_curve, uv_curve:
                curve.y[curve.y < 0] = 0

        self.xr_curve = xr_curve
        self.xrD = xrD
        self.xrE = dsets.E
        self.xr_zero_scatter = np.zeros(xrD.shape[0])
        xr_rank  = self.n_components
        if self.separate_eoii_type == 1:
            xr_rank += 1
        elif self.separate_eoii_type == 2:
            # not yet supported
            xr_rank += np.sum(self.separate_eoii_flags, dtype=int)
        self.xrD_ = get_denoised_data(xrD, rank=xr_rank)
        self.xr_norm1 = np.linalg.norm(xr_curve.y)
        self.xr_norm2 = np.linalg.norm(xrD)
        self.rg_curve = rg_curve
        self.uv_curve = uv_curve
        self.uvD = uvD
        self.uvD_ = get_denoised_data(uvD, rank=self.n_components)

        if False:
            from MatrixData import simple_plot_3d
            plt.push()
            fig = plt.figure(figsize=(14,7))
            ax1 = fig.add_subplot(121, projection='3d')
            ax2 = fig.add_subplot(122, projection='3d')
            simple_plot_3d(ax1, uvD)
            simple_plot_3d(ax2, xrD)
            fig.tight_layout()
            plt.show()
            plt.pop()

        uv_scale = uv_curve.max_y / xr_curve.max_y
        self.uv_norm1 = self.xr_norm1 * uv_scale
        self.uv_norm2 = self.xr_norm2 * uv_scale
        self.uv_i = np.arange(uvD.shape[0])
        self.uv_j = uv_curve.x

        if USE_NORMALIZED_RMSD or USE_FROBENIUS_XDIFFMAX:
            self.uv2d_adjust = ADJUST_2D_TARGET - np.log10(self.uv_norm1)
            self.xr2d_adjust = ADJUST_2D_TARGET - np.log10(self.xr_norm1)

        if USE_NORMALIZED_RMSD_FOR_RGCURVES:
            self.rgcurve_adjust = -1.2
        else:
            self.rgcurve_adjust = -2.7

        if USE_COLUMN_INTERP:
            if False:
                import Optimizer.ColumnInterp
                reload(Optimizer.ColumnInterp)
            from .ColumnInterp import ColumnInterp
            self.uv_interp = ColumnInterp(self.uvD_)
        else:
            from scipy.interpolate import RectBivariateSpline
            self.uv_interp = RectBivariateSpline(self.uv_i, self.uv_j, self.uvD_, s=0)
        self.uv_shape = (uvD.shape[0], xrD.shape[1])
        self.given_seeds = None
        self.real_bounds = None
        self.eval_counter = 0
        self.callback_counter = 0
        self.svd_error_count = 0
        self.minima_props = None
        self.valid_components = np.ones(self.num_pure_components, dtype=bool)
        self.ratio_interpretation = get_setting("ratio_interpretation")
        self.poresize_bounds = get_setting("poresize_bounds")
        self.apply_rg_discreteness = get_setting("apply_rg_discreteness")
        self.apply_mw_integrity = get_setting("apply_mw_integrity")
        self.apply_discreteness = self.apply_rg_discreteness or self.apply_mw_integrity
        self.mw_integer_ratios = get_setting("mw_integer_ratios")
        self.c_indeces = None
        self.rg_discreteness_unit = get_setting("rg_discreteness_unit")
        self.kratky_smoothness = get_setting("kratky_smoothness")
        self.avoid_peak_fronting = get_setting("avoid_peak_fronting")
        self.apply_sf_bounds = get_setting("apply_sf_bounds")
        self.logger.info("ratio_interpretation=%d, poresize_bounds=%s, separate_eoii_type=%d, apply_rg_discreteness=%d, rg_discreteness_unit=%g",
            self.ratio_interpretation, str(self.poresize_bounds), self.separate_eoii_type, self.apply_rg_discreteness, self.rg_discreteness_unit)

        if self.apply_mw_integrity:
            self.logger.info("applying mw discreteness with ratios %s", str(self.mw_integer_ratios))

        self.exception_count = 0
        self.isnan_logged = False
        self.debug_fv = False

        # note that these weights are independent of composites variation
        self.uw, self.vw, self.W_ = dsets.get_opt_weight_info()
        self.xrDw = self.W_ * self.xrD_
        # self.rg_weights = self.rg_curve.get_weights()
        self.exports_bounds = False
        self.vc = ValidComponents(self.num_pure_components)

    def get_model_name(self):
        return self.params_type.get_model_name()

    def get_num_components(self):
        return self.num_pure_components

    def get_paramslider_info(self):
        return self.params_type.get_paramslider_info()

    def set_composite(self, composite):
        self.logger.info("this composite has been renewed to %s", str(composite))
        self.composite = composite

    def get_required_params(self, params):
        from molass_legacy._MOLASS.SerialSettings import set_setting
        num_params = self.params_type.num_params
        set_setting("init_sec_params", params[num_params:])
        return params[0:num_params], params[num_params:]

    def split_params_simple(self, p):
        self.separate_params = self.params_type.split_params_simple(p)
        return self.separate_params

    def split_as_unified_params(self, p, **kwargs):
        return self.params_type.split_as_unified_params(p, **kwargs)

    def get_parameter_names(self):
        return self.params_type.get_parameter_names()

    def solve(self, init_params, real_bounds=None, niter=100, seed=None, callback=True, method=None, debug=False, show_history=False):
        t0 = time()
        self.logger.info("solve started with niter=%d, seed=%s", niter, str(seed))
        self.logger.info("init_params=%s", str(init_params))

        self.prepare_for_optimization(init_params, real_bounds=real_bounds)

        self.fv_array = np.zeros(niter + 3)
        self.fv_array_size = 0
        self.min_fv = None
        self.min_i = None

        self.debug_fv = debug
        if callback:
            self.cb_fh = open("callback.txt", "w")
            self.write_init_callback_txt(init_params)

        if method is None:
            from .OptimizerUtils import get_impl_method_name
            method = get_impl_method_name()

        bounds = np.array([(0, 10)]*len(init_params))
        norm_params = self.to_norm_params(init_params)

        if method == "bh":
            from importlib import reload
            import molass_legacy.Solvers.BH.SolverBH
            reload(molass_legacy.Solvers.BH.SolverBH)
            from molass_legacy.Solvers.BH.SolverBH import SolverBH
            bh = SolverBH(self)
            result = bh.minimize(self.objective_func_wrapper, norm_params, niter=niter, seed=seed, bounds=bounds, show_history=show_history)

        elif method == "ultranest":
            from importlib import reload
            import molass_legacy.Solvers.UltraNest.SolverUltraNest
            reload(molass_legacy.Solvers.UltraNest.SolverUltraNest)
            from molass_legacy.Solvers.UltraNest.SolverUltraNest import SolverUltraNest
            ultranest = SolverUltraNest(self)
            result = ultranest.minimize(self.objective_func_wrapper, norm_params, niter=niter, seed=seed, bounds=bounds)
 
        elif method == "emcee":
            from importlib import reload
            import molass_legacy.Solvers.MCMC.SolverEmcee
            reload(molass_legacy.Solvers.MCMC.SolverEmcee)
            from molass_legacy.Solvers.MCMC.SolverEmcee import SolverEmcee
            mcmc = SolverEmcee(self)
            result = mcmc.minimize(self.objective_func_wrapper, norm_params, niter=niter, seed=seed, bounds=bounds)
 
        elif method == "pyabc":
            from importlib import reload
            import molass_legacy.Solvers.ABC.SolverPyABC
            reload(molass_legacy.Solvers.ABC.SolverPyABC)
            from molass_legacy.Solvers.ABC.SolverPyABC import SolverPyABC
            abc = SolverPyABC(self)
            result = abc.minimize(self.objective_func_wrapper, norm_params, niter=niter, seed=seed, bounds=bounds)

        elif method == "pymc":
            from importlib import reload
            import molass_legacy.Solvers.SMC.SolverPyMC
            reload(molass_legacy.Solvers.SMC.SolverPyMC)
            from molass_legacy.Solvers.SMC.SolverPyMC import SolverPyMC, get_picklable_func
            smc = SolverPyMC(self)
            func = get_picklable_func(self)
            result = smc.minimize(func, norm_params, niter=niter, seed=seed, bounds=bounds)
 
        else:
            raise ValueError("Unknown method: %s" % method)

        if callback:
            self.cb_fh.close()

        self.logger.info("solve finished in %.3g minutes %.3g seconds with %d iterations and %d evaluations.",
                            *divmod(time()-t0, 60), result.nit, result.nfev)

        if self.svd_error_count > 0:
            self.logger.warning("There were %d SVD errors.", self.svd_error_count)

        return result

    def prepare_for_optimization(self, init_params, real_bounds=None, minimally=False):
        self.eval_counter = 0
        self.init_params = init_params
        self.init_separate_params = self.split_params_simple(init_params)
        self.separate_params = self.init_separate_params    # to ensure that self.separate_params always exists
        xr_params, xr_baseparams, rgs, mapping, uv_params, uv_baseparams, mappable_range = self.init_separate_params[0:7]
        self.init_rgs = rgs
        self.init_uv_params = uv_params
        self.init_uv_baseparams = uv_baseparams
        self.init_mapping = mapping
        self.init_mappable_range = mappable_range
        if real_bounds is None:
            self.real_bounds = np.array(self.get_param_bounds(init_params))
            from_arg = False
        else:
            # for SDM
            self.real_bounds = real_bounds
            from_arg= True
        self.logger.info("from_arg=%s, real bounds=%s", from_arg, str(self.real_bounds))
        self.set_params_scale(self.real_bounds)
        self.bounds_mask = self.params_type.make_bounds_mask()
        self.update_bounds(self.init_params)
        masked_init_params = self.init_params[self.bounds_mask]
        self.zero_bounds = np.zeros(masked_init_params.shape)
        self.sf_bounds = None   # referenced in objective_func, but is this ok?

        self.update_minima_props(init_params)

        self.ones_nc = np.ones(xr_params.shape[0])
        self.init_xr_params = xr_params
        self.init_xr_baseparams = xr_baseparams
        init_slope, init_intercept = xr_baseparams[0:2]     # how about r?
        self.init_slope = init_slope
        self.init_intercept = init_intercept
        self.slope_allowance = (init_slope*SLOPE_ALLOWANCE)**2
        self.intercept_allowance = (init_intercept*INTERCEPT_ALLOWANCE)**2
        self.slope_penalty_scale = 1/max(1e-16, self.slope_allowance)
        self.intercept_penalt_scale = 1/max(1e-16, self.intercept_allowance)

        self.zeros_rg = np.zeros(len(self.init_separate_params[2]) - 1)
        if EVAL_PEAK_DEVIATION:
            self.update_primary_peak()          # must be called before update_guinier_region()

        # estimate approximate peak_points used in the optimization for SEC conformance evaluation
        lrf_info = self.objective_func(init_params, return_lrf_info=True)
        self.peak_points = lrf_info.estimate_xr_peak_points()
        self.update_guinier_region()

        if minimally:
            return

        self.xm, self.ym, self.rg = self.rg_curve.get_valid_curves()
        self.mask = self.rg_curve.get_mask()
        self.ones_rg = np.ones(len(self.rg))

        if self.separate_eoii_type > 0 and self.apply_sf_bounds:
            if True:
                from importlib import reload
                import molass_legacy.Optimizer.StructureFactorBounds
                reload(molass_legacy.Optimizer.StructureFactorBounds)
            from .StructureFactorBounds import StructureFactorBounds
            from molass_legacy.Kratky.GuinierKratkyInfo import GuinierKratkyInfo
            # consier using GuinierDeviation as an aternative
            gk_info = GuinierKratkyInfo(self, init_params, lrf_info)
            self.sf_bounds = StructureFactorBounds(self.qvector, lrf_info, gk_info)

    def update_minima_props(self, params):
        lrf_info = self.objective_func(params, return_lrf_info=True)
        self.minima_props = lrf_info.get_xr_proportions()
        self.valid_components = self.vc.get_valid_vector()
        self.logger.info("valid_components=%s", self.valid_components)
        num_valid_components = len(np.where(self.valid_components)[0])
        if num_valid_components > 1:
            self.zeros_valid_rg = np.zeros(num_valid_components - 1)
        else:
            # self.zeros_valid_rg will not be used in the optimization
            pass

    def write_init_callback_txt(self, init_params):
        self.eval_counter = -1
        fv  = self.objective_func(init_params)
        x = self.to_norm_params(init_params)
        self.minima_callback(x, fv, False)

    def compute_LRF_matrices(self, x, y, xr_cy_list, xr_ty, uv_x, uv_y, uv_cy_list, uv_ty, debug=False):
        Cxr = self.composite.compute_C_matrix(y, xr_cy_list, eoii=True, ratio_interpretation=self.ratio_interpretation)
        Pxr = (self.xrDw @ np.linalg.pinv(Cxr*self.vw))/self.uw
        Cuv = self.composite.compute_C_matrix(uv_y, uv_cy_list, ratio_interpretation=self.ratio_interpretation)
        if USE_COLUMN_INTERP:
            mapped_UvD = self.uv_interp(uv_x)
        else:
            mapped_UvD = self.uv_interp(self.uv_i, uv_x).reshape(self.uv_shape)
        Puv = mapped_UvD @ np.linalg.pinv(Cuv)

        if debug:
            from molass_legacy.DataStructure.MatrixData import simple_plot_3d

            with plt.Dp():
                fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5), subplot_kw=dict(projection="3d"))
                fig.suptitle("compute_LRF_matrices debug (1)")
                ax1.set_title("uvD")
                ax2.set_title("mapped_UvD")
                simple_plot_3d(ax1, self.uvD)
                simple_plot_3d(ax2, mapped_UvD)
                fig.tight_layout()
                plt.show()

        if USE_COMPOSED_XR_COMPONENTS:
            scaled_xr_cy_array = (Pxr[self.xr_index,:] * Cxr.T).T
            xr_ty = np.sum(scaled_xr_cy_array, axis=0)
        else:
            scaled_xr_cy_array = Cxr

        if USE_COMPOSED_UV_COMPONENTS:
            scaled_uv_cy_array = (Puv[self.uv_index,:] * Cuv.T).T
            uv_ty = np.sum(scaled_uv_cy_array, axis=0)
        else:
            scaled_uv_cy_array = Cuv

        if debug:
            print("self.uvD.shape=", self.uvD.shape)
            print("mapped_UvD.shape=", mapped_UvD.shape)

            def plot_componnts(ax, x, y, n, cy_list, ty=None, data_color=None):
                computing_ty = ty is None
                if computing_ty:
                    ty = np.zeros(len(x))
                ax.plot(x, y, label="data", color=data_color)
                for k, cy in enumerate(cy_list):
                    ax.plot(x, cy, ":", label="component-%d" % k)
                    if computing_ty and k < n:
                        ty += cy
                ax.plot(x, ty, ":", color="red", label="total")
                ax.legend()

            n = self.n_components
            with plt.Dp():
                fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(12,10))
                fig.suptitle("compute_LRF_matrices debug (2)")
                ax11 = axes[0,0]
                ax12 = axes[0,1]
                ax21 = axes[1,0]
                ax22 = axes[1,1]

                ax11.set_title("UV components")
                ax12.set_title("XR components")
                ax21.set_title("UV scaled components")
                ax22.set_title("XR scaled components")

                plot_componnts(ax11, uv_x, uv_y, n, uv_cy_list, data_color="blue")
                plot_componnts(ax12, x, y, n, xr_cy_list, data_color="orange")
                plot_componnts(ax21, uv_x, uv_y, n, scaled_uv_cy_array, uv_ty, data_color="blue")
                plot_componnts(ax22, x, y, n, scaled_xr_cy_array, xr_ty, data_color="orange")

                fig.tight_layout()
                plt.show()

        if COERCE_BOUNDED_BQ:
            if self.sf_bounds is not None:
                Pxr[:,-1] = self.sf_bounds.compute_bounded_bq(Pxr)

        return OptLrfInfo(Pxr, Cxr, Puv, Cuv, mapped_UvD,
                    self.qvector, self.xrD, self.xrE, x, y, xr_ty, scaled_xr_cy_array, uv_x, uv_y, uv_ty, scaled_uv_cy_array, self.composite)

    def create_lrf_info_for_debug(self, x, y, xr_ty, xr_cy_list, uv_x, uv_y, uv_ty, uv_cy_list):
        from importlib import reload
        import molass_legacy.Optimizer.OptLrfInfoDebug
        reload(molass_legacy.Optimizer.OptLrfInfoDebug)
        from molass_legacy.Optimizer.OptLrfInfoDebug import OptLrfInfoProxy
        return OptLrfInfoProxy(self.qvector, self.xrD, self.xrE, x, y, xr_ty, xr_cy_list, uv_x, uv_y, uv_ty, uv_cy_list, self.composite)

    def update_rg_params(self, lrf_info):
        lrf_info.update_optimizer(self)

    def compute_fv(self, lrf_info, xr_params, rg_params, seccol_params, penalties, p, debug=False):
        Pxr, Cxr, Puv, Cuv, mapped_UvD = lrf_info.matrices

        if USE_NORMALIZED_RMSD:
            XR_2D_fitting = normalized_rmsd(lrf_info.xr_ty, lrf_info.y, adjust=self.xr2d_adjust)
            UV_2D_fitting = normalized_rmsd(lrf_info.uv_ty, lrf_info.uv_y, adjust=self.uv2d_adjust)
        elif USE_FROBENIUS_XDIFFMAX:
            XR_2D_fitting = frobenius_xdiffmax(lrf_info.xr_ty, lrf_info.y, adjust=self.xr2d_adjust)
            UV_2D_fitting = frobenius_xdiffmax(lrf_info.uv_ty, lrf_info.uv_y, adjust=self.uv2d_adjust)            
        elif USE_FEATURE_DEVIATION:
            XR_2D_fitting = feature_deviation(lrf_info.xr_ty, lrf_info.y, self.xr_curve.max_y, self.xr_norm1)
            UV_2D_fitting = feature_deviation(lrf_info.uv_ty, lrf_info.uv_y, self.uv_curve.max_y, self.uv_norm1)
        elif USE_JSD:
            XR_2D_fitting = deformed_jsd(lrf_info.xr_ty, lrf_info.y)
            UV_2D_fitting = deformed_jsd(lrf_info.uv_ty, lrf_info.uv_y)
        else:
            assert False, "invalid fitting method"
 
        XR_LRF_residual = np.log10(np.linalg.norm(self.W_*(Pxr @ Cxr - self.xrD_))/self.xr_norm2)
        UV_LRF_residual = np.log10(np.linalg.norm(Puv @ Cuv - mapped_UvD)/self.uv_norm2)

        Guinier_deviation = self.get_guinier_deviation(Pxr, Cxr, rg_params)
        SEC_conformance = self.compute_comformance(xr_params, rg_params, seccol_params)

        qv = self.gdev.qv
        kratky_plot_smoothness = 0
        for k, p in enumerate(Pxr.T[0:self.num_pure_components]):
            rg = self.gdev.rgs[k]
            if np.isnan(rg):
                continue
            iz = self.gdev.izs[k]
            kratky_y = (qv*rg)**2*p/iz

            # negative penalty
            kratky_plot_smoothness += np.mean(np.min([self.xr_zero_scatter, kratky_y], axis=0)**2)

            if self.kratky_smoothness:
                # smoothness
                # https://stackoverflow.com/questions/68023055/how-to-evaluate-the-smoothness-flatness-of-a-curve-in-python
                kratky_plot_smoothness += np.std(np.diff(kratky_y))

        if np.isnan(kratky_plot_smoothness):
            kratky_plot_smoothness = KRATKY_SMOOTHNESS_BAD_VALUE
        # basic_mean = np.mean([XR_2D_fitting, UV_2D_fitting, XR_LRF_residual, UV_LRF_residual])
        Kratky_smoothness = np.log10(kratky_plot_smoothness) - 0.6
        Guinier_deviation = Guinier_deviation

        if debug:
            self.logger.info("Guinier_deviation=%.3g, kratky_plot_smoothness=%.3g", Guinier_deviation, Kratky_smoothness)

        score_list = [XR_2D_fitting, XR_LRF_residual, UV_2D_fitting, UV_LRF_residual, Guinier_deviation, Kratky_smoothness, SEC_conformance]
        assert len(score_list) == self.NUM_MAJOR_SCORES

        if not COERCE_BOUNDED_BQ:
            # B(q) out of bounds
            if self.sf_bounds is not None:
                penalties[3] += self.sf_bounds.compute_penalty(Pxr)

        # negetive_penalty for P's
        for P in [Pxr, Puv]:
            P_ = P[:,:-1]   # excluding baseline component
            penalties[1] += WEAK_PENALTY_SCALE * np.linalg.norm(P_[P_ < 0])

        # common order_penalty
        valid_rg_params = rg_params[self.valid_components]
        if len(valid_rg_params) > 1:
            # Rg order_penalty
            penalties[4] += PENALTY_SCALE * np.sum(np.min([self.zeros_valid_rg, valid_rg_params[:-1] - valid_rg_params[1:]], axis=0)**2)

        if EVAL_PEAK_DEVIATION:
            penalties[4] += PENALTY_SCALE * self.compute_peak_deviation(xr_params, debug=debug)

        # to avoid cases as in 20230303/HasA where UV_LRF_residual is lowered
        # without considering the bad effect that it seriously sacrifices UV_2D_fitting
        control_penalty = WEAK_PENALTY_SCALE * ( 
                              max(0, XR_2D_fitting - XR_LRF_residual - SUPERIOR_2D_LRF_ALLOW)**2
                            + max(0, UV_2D_fitting - UV_LRF_residual - SUPERIOR_2D_LRF_ALLOW)**2
                            )
        penalties.append(control_penalty)

        discreteness = 0
        if self.apply_rg_discreteness:
            discreteness += compute_rg_discreteness(rg_params, unit=self.rg_discreteness_unit)

        if self.apply_mw_integrity:
            discreteness += self.compute_mw_integrity(Pxr, Cuv, UV_2D_fitting)

        if self.apply_discreteness:
            score_list.append(discreteness)

        score_array = np.array(score_list)
        fv = synthesize(score_array, positive_elevate=3) + np.sum(penalties)

        if np.isnan(fv):
            # NaN is not allowed in scipy.optimize.basinhopping
            if not self.isnan_logged:
                self.logger.info("fv is NaN: score_array=%s", str(score_array))
                self.isnan_logged = True
            fv = np.inf
        return fv, score_array

    def objective_func(self, p, plot=False, debug=False, fig_info=None, axis_info=None, return_full=False):
        # override this
        assert False

    def debug_plot_params(self, norm_params, **kwargs):
        plot = kwargs.pop("plot", True)
        return self.objective_func(self.to_real_params(norm_params), plot=plot, **kwargs)

    def objective_func_wrapper(self, norm_params, **kwargs):
        return self.objective_func(self.to_real_params(norm_params), **kwargs)

    def get_score_names(self, major_only=False):
        names = ([  "XR_2D_fitting", "XR_LRF_residual", "UV_2D_fitting", "UV_LRF_residual",
                    "Guinier_deviation", "Kratky_smoothness", "SEC_conformance",
                    "mapping_penalty", "negative_penalty", "baseline_penalty",
                    "outofbounds_penalty", "order_penalty", "control_penalty",
                    ]
                )
        if major_only:
            return names[0:self.NUM_MAJOR_SCORES]

        if self.apply_rg_discreteness or self.apply_mw_integrity:
            count = 0
            if self.apply_rg_discreteness:
                plotname = "discreteness: "
                rg_name = "%g" % self.rg_discreteness_unit
                count += 1
            else:
                rg_name = ""
            if self.apply_mw_integrity:
                plotname = "MW ratios: "    # this overwrite the above
                mw_name = "%s" % self.mw_integer_ratios
                count += 1
            else:
                mw_name = ""
            comma = ", " if count == 2 else ""
            names.insert(7, "%s%s%s%s" % (plotname, rg_name, comma, mw_name))
        return names

    def get_num_scores(self, penalties):
        n = self.NUM_MAJOR_SCORES + len(penalties)
        if self.apply_rg_discreteness:
            n += 1
        return n

    def get_num_matrices(self):
        return 5

    def set_params_scale(self, real_bounds, debug=False):
        lower = real_bounds[:,0]
        upper = real_bounds[:,1]
        self.scale_shift = lower
        self.scale_slope = (upper - lower)/PARAMS_SCALE
        size = len(real_bounds)
        self.xlower = np.zeros(size)
        self.xupper = np.ones(size)*PARAMS_SCALE

        if debug:
            with plt.Dp():
                fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
                for ax, (start, stop) in [
                                            (ax1, (0, -6)),
                                            (ax2, (-6, len(lower)))
                                            ]:
                    slice_ = slice(start, stop)
                    ax.set_title("set_params_scale %s" % (str((start, stop))))
                    ax.plot(self.scale_shift[slice_], label="lower(or scale_shift)")
                    ax.plot(upper[slice_], label="upper")
                    axt = ax.twinx()
                    axt.grid(False)
                    axt.plot(self.scale_slope[slice_], color="C2", label="scale_slope")
                    ax.legend()
                    axt.legend(loc="center left")
                fig.tight_layout()
                plt.show()

    def to_norm_params(self, real_params):
        conv_params = real_params.copy()   # for not to change the argument
        if self.use_K:
            # K = N*T
            # log(K) = log(N) + log(T)
            # p = log(T)/log(K)
            # m = me + mp
            # q = mp/m
            N, me, T, mp = real_params[self.nj:]
            K =  N*T
            m = me + mp
            conv_params[self.nj] = K
            conv_params[self.mej] = m
            conv_params[self.tj] = np.log(T)/np.log(K)
            conv_params[self.mpj] = mp/m
        nx = (conv_params - self.scale_shift)/self.scale_slope
        return nx

    def to_real_params(self, norm_params):
        rx = self.scale_slope*norm_params + self.scale_shift
        if self.use_K:
            K, m, p, q = rx[self.nj:]
            # N = K**(1 - p)
            # T = K**p
            # me = m*(1 - q)
            # mp = m*q
            rx[self.nj] = K**(1 - p)
            rx[self.mej] = m*(1 - q)
            rx[self.tj] = K**p
            rx[self.mpj] = m*q
        return rx

    def accept_test(self, **kwargs):
        x = kwargs["x_new"]
        tmax = np.all(x <= self.xupper)
        tmin = np.all(x >= self.xlower)
        return tmax and tmin

    def minima_callback(self, x, f, accept):
        real_params = self.to_real_params(x)
        self.logger.info("minima_callback: f=%.3g, accept=%s", f, str(accept))
        save_opt_params(self.cb_fh, real_params, f, accept, self.eval_counter)
        self.callback_counter += 1
        if self.shm is not None:
            self.shm.array[0] = self.callback_counter

        self.update_at_minima(x, f, accept)

        if IMMEDIATE_KNOWN_BEST:
            if self.min_fv is None or f < self.min_fv:
                self.min_fv = f
                self.min_i = self.fv_array_size
                self.fv_array[self.min_i] = f
                self.fv_array_size += 1
                self.update_bounds(real_params)

        return False

    def update_at_minima(self, x, f, accept):
        self.update_minima_props(x)

    def update_bounds(self, x, debug=False):
        if True:
            if self.exports_bounds:
                bounds_array = np.array(self.params_type.get_param_bounds(x, real_bounds=self.real_bounds))
            else:
                bounds_array = np.array(self.params_type.get_param_bounds(x))
            # self.logger.info("update_bounds(exports_bounds=%s): bounds_array=%s", self.exports_bounds, str(bounds_array))
            if bounds_array.shape[0] == len(x):
                pass
            else:
                error_msg = "bounds_array.shape[0](%d) != len(x)(%d)" % (bounds_array.shape[0], len(x))
                self.logger.error(error_msg)
                raise AssertionError(error_msg)

            self.lower_bounds = bounds_array[self.bounds_mask,0]
            self.upper_bounds = bounds_array[self.bounds_mask,1]
        else:
            scale = 2
            masked_init_params = x[self.bounds_mask]
            self.params_type.update_bounds_hook(masked_init_params)
            self.lower_bounds = masked_init_params / scale
            self.upper_bounds = masked_init_params * scale
        if debug:
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.set_title("update_bounds")
                ax.plot(self.lower_bounds, label="lower_bounds")
                ax.plot(self.upper_bounds, label="upper_bounds")
                ax.legend()
                fig.tight_layout()
                plt.show()

    def update_bounds_hook(self, masked_init_params):
        # nothing to do
        pass

    def get_param_bounds(self, params):
        return self.params_type.get_param_bounds(params)

    def split_bounds(self, bounds):
        return self.params_type.split_bounds(bounds)

    def reshape_xr_bounds(self, xr_bounds):
        pass

    def xr_baseline(self, x, xr_baseparams, y_, cy_list):
        # note: update also ComplementaryView
        return self.xr_base_curve(x, xr_baseparams, y_, cy_list)

    def uv_baseline(self, x, uv_baseparams, y_, cy_list):
        # note: update also ComplementaryView
        return self.uv_base_curve(x, uv_baseparams, y_, cy_list)

    def get_name(self):
        import re
        name_re = re.compile(r"'([^']+)'")
        m = name_re.search(str(type(self)))
        name = m.group(1).split(".")[-1]
        return name

    def get_region_limit(self):
        # this should be properly overridden
        return 0

    def update_primary_peak(self, debug=False):
        xr_params = self.init_separate_params[0]
        if len(xr_params.shape) == 1:
            i = np.argmax(xr_params)
        else:
            i = np.argmax(xr_params[:,0])
        self.primary_component = i
        j = np.argmax(self.xr_curve.y)
        x = self.xr_curve.x
        self.xr_primary_x = x[j]
        self.xr_x_width = x[-1] - x[0]

        x = self.uv_curve.x
        y = self.uv_curve.y
        self.uv_x_width = x[-1] - x[0]
        a, b = self.init_separate_params[3]
        uv_x = a*self.xr_primary_x + b
        j = bisect_right(x, uv_x)
        start = max(0, j - 10)
        stop = min(len(x), j + 10)
        neighbor = slice(start, stop)
        m = np.argmax(y[neighbor])
        j_ = start + m
        self.uv_primary_x = x[j_]
        if debug:
            
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.plot(x, y)
                ax.plot(x[j_], y[j_], "o", color="red")
                fig.tight_layout()
                plt.show()

    def compute_peak_deviation(self, xr_params, debug=False):
        if self.use_K:
            # meaning if self.stochastic
            # currently there is no simple method to compute deviation for stochastic models
            return 0
        else:
            a, b = self.separate_params[3]
            xr_x = self.params_type.compute_xr_peak_position(self.primary_component, xr_params)
            xr_dev = (max(PRIMARY_PEAK_DEV_ALLOW, abs(xr_x - self.xr_primary_x)/self.xr_x_width) - PRIMARY_PEAK_DEV_ALLOW)**2
            uv_x = a*xr_x + b
            uv_dev = (max(PRIMARY_PEAK_DEV_ALLOW, abs(uv_x - self.uv_primary_x)/self.uv_x_width) - PRIMARY_PEAK_DEV_ALLOW)**2
            if debug:
                self.logger.info("xr_peak_deviation=%g, uv_peak_deviation=%g", xr_dev, uv_dev)
            return xr_dev + uv_dev

    def update_guinier_region(self, params=None, debug=False):
        if debug:
            from importlib import reload
            import molass_legacy.GuinierTools.GuinierDeviation
            reload(molass_legacy.GuinierTools.GuinierDeviation)
        from molass_legacy.GuinierTools.GuinierDeviation import GuinierDeviation

        if params is None:
            params = self.init_params
            rg_params = self.init_separate_params[2]
        else:
            rg_params = self.separate_params[2]     # this will be probably right, meaning that rg_params are consistant with params
        region_limit = self.get_region_limit()
        self.gdev = GuinierDeviation(self.qvector, self.rg_curve, rg_params, region_limit, composite=self.composite)
        
        lrf_info = self.objective_func(params, return_lrf_info=True)
        Pxr = lrf_info.matrices[0]
        Cuv = lrf_info.matrices[3]
        self.gdev.update_region(rg_params, region_limit, P=Pxr, M=self.xrD, E=self.xrE, debug=debug)

        if self.apply_mw_integrity:
            self.c_indeces = np.argmax(Cuv[:-1], axis=1)
            self.logger.info("updated c_indeces to %s", str(self.c_indeces))

    def get_guinier_deviation(self, P, Cxr, rg_params, debug=False):
        if USE_RGCURVE_DEVIATION:
            # use Optimizer.FvScoreConverter.compute_fv_adjustment to change the adjustment
            deviation = self.gdev.compute_rgcurve_deviation(Cxr, rg_params, adjust=self.rgcurve_adjust)
            return deviation 
        else:
            deviation = self.gdev.compute_deviation(P, Cxr, rg_params, valid_components=self.valid_components, debug=debug)
            if deviation is None:
                return
            return deviation*0.5 - 0.5

    def compute_comformance(self, xr_params, rg_params, seccol_params):
        conformance = self.params_type.compute_comformance(xr_params, rg_params, seccol_params, poresize_bounds=self.poresize_bounds)
        return conformance*0.5 - 0.1

    def compute_mw_integrity(self, P, C, preceder):
        if self.c_indeces is None:
            self.c_indeces = np.argmax(C[:-1], axis=1)
        cvector = np.array([C[i,j] for i,j in enumerate(self.c_indeces)])
        mw_ratios = self.gdev.izs / cvector
        if self.mw_integer_ratios is None:
            from .IntegerRatios import determine_integer_ratios
            self.mw_integer_ratios = determine_integer_ratios(mw_ratios)
        dist = compute_mw_integrity_impl(mw_ratios, self.mw_integer_ratios, preceder)
        return dist

    def compute_moments_list(self, params=None, debug=False):
        from molass_legacy.Peaks.MomentsUtils import compute_moments
        if params is None:
            params = self.init_params
        
        lrf_info = self.objective_func(params, return_lrf_info=True)
        x = lrf_info.x
        moments_list = []
        for cy in lrf_info.scaled_xr_cy_array[0:-1]:    # excluding baseline component
            M = compute_moments(x, cy)
            moments_list.append(M)
        if debug:
            y = lrf_info.y
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.plot(x, y, label="data")
                for k, (cy, M) in enumerate(zip(lrf_info.scaled_xr_cy_array, moments_list)):
                    ax.plot(x, cy, ":", label="component-%d" % k)
                    ax.axvline(M[1], color="green", label="M1=%g" % M[1])
                    s = np.sqrt(M[2])
                    ax.axvspan(M[1]-s, M[1]+s, color="green", alpha=0.2)
                ax.legend()
                fig.tight_layout()
                plt.show()
        return moments_list
    
    def get_strategy(self):
        from molass_legacy.Optimizer.Strategies.DefaultStrategy import DefaultStrategy
        return DefaultStrategy()
    
    def is_stochastic(self):
        return False