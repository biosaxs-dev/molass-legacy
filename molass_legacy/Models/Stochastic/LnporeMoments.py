"""
    Models.Stochastic.LnporeMoments.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import logging
import numpy as np
from scipy.optimize import minimize, basinhopping
from scipy.integrate import quad
from datetime import datetime
from importlib import reload
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.Models.Stochastic.MomentUtils import compute_egh_moments, to_moderate_props
from molass_legacy.Models.Stochastic.LognormalUtils import compute_mu_sigma
import Models.Stochastic.LognormalPoreFunc
reload(Models.Stochastic.LognormalPoreFunc)
from molass_legacy.Models.Stochastic.LognormalPoreFunc import distr_func
from molass_legacy.Models.Stochastic.LnporeUtils import plot_lognormal_fitting_state
from molass_legacy.Models.Stochastic.MomentsStudy import monopore_study
from molass_legacy.Models.Stochastic.ParamLimits import MAX_PORESIZE, M1_WEIGHT, M2_WEIGHT

class LnporeMomentsEstimater:
    def __init__(self, lrf_src, parent=None, debug=False):
        self.logger = logging.getLogger(__name__)
        self.lrf_src = lrf_src
        self.parent = parent
        self.x = lrf_src.xr_x
        self.y = lrf_src.xr_y
        all_peaks = lrf_src.xr_peaks
        rgs, trs, orig_props, peak_rgs, peak_trs, props, indeces, qualities = lrf_src.compute_rgs(keep_num_components=False, debug=False)
        self.peaks = all_peaks[indeces]
        self.peak_rgs = peak_rgs
        self.props = props
        self.indeces = indeces

    def estimate(self, debug=False):
        import Models.Stochastic.MonoporeMoments
        reload(Models.Stochastic.MonoporeMoments)
        from molass_legacy.Models.Stochastic.MonoporeMoments import study_monopore_moments_impl

        num_peaks = len(self.peaks)
        egh_moments_list = compute_egh_moments(self.peaks)
        m_props = to_moderate_props(self.props)

        x = self.x
        y = self.y
        if False:
            mnp_params = monopore_study(x, y, peaks, peak_rgs, props, egh_moments_list, logger=logger, debug=False)
        else:
            mnp_params, temp_rgs, unreliable_indeces, params_scaler = study_monopore_moments_impl(self.lrf_src, debug=debug)
            self.logger.info("mnp_params=%s, temp_rgs=%s, unreliable_indeces=%s", mnp_params, temp_rgs, unreliable_indeces)
        N, T, x0, me, mp, poresize = mnp_params[0:6]

        nur = len(unreliable_indeces)

        use_basinhopping = False
        param_scales = np.array([100, 0.1, 50, 1, 0.1] + [10]*nur)  # scales to normalize the parameters for basinhopping

        def lnp_column_flex_objective(p, debug=False):
            if use_basinhopping:
                p *= param_scales
            N_, T_, x0_, mu, sigma = p[0:5]
            if nur > 0:
                temp_rgs[unreliable_indeces] = p[5:]

            dev_list = []
            for k, (M, rg) in enumerate(zip(egh_moments_list, temp_rgs)):
                M1_ = x0_ + N_ * T_   *  quad(lambda r : distr_func(r, mu, sigma) * (1 - min(1, rg/r))**(me +   mp), rg, MAX_PORESIZE)[0]
                M2_ =   2 * N_ * T_**2 * quad(lambda r : distr_func(r, mu, sigma) * (1 - min(1, rg/r))**(me + 2*mp), rg, MAX_PORESIZE)[0]
                dev_list.append(M1_WEIGHT*(M1_ - M[0])**2 + M2_WEIGHT*(M2_ - M[1])**2)
                if debug:
                    print([k], "M1, M1_ = %.3g, %.3g" % (M[0], M1_))
                    print([k], "M2, M2_ = %.3g, %.3g" % (M[1], M2_))
            return np.sum(np.asarray(dev_list)*m_props)

        stdev = poresize*0.2
        init_mu, init_sigma = compute_mu_sigma(poresize, stdev)
        init_params = [N, T, x0, init_mu, init_sigma] + list(temp_rgs[unreliable_indeces])
        eghM = egh_moments_list[0]
        x0_upper = eghM[0] - 5*np.sqrt(eghM[1])
        bounds = [(500, 20000), (0.01, 2), (-1000, x0_upper), (1, 10), (0.02, 1.0)] + [(10, 100)]*nur
        print("init_params=", init_params)
        print("bounds=", bounds)
        if use_basinhopping:
            init_params_ = np.array(init_params)/param_scales
            bounds_ = [(lower/scale, upper/scale) for (lower, upper), scale in zip(bounds, param_scales)]
            self.minima_counter = 0
            res = basinhopping(lnp_column_flex_objective, init_params_, niter=2,
                            callback=self.minima_callback,
                            minimizer_kwargs=dict(method='Nelder-Mead', bounds=bounds_))
            res.x *= param_scales   # do not forget to scale back the result
        else:
            res = minimize(lnp_column_flex_objective, init_params, bounds=bounds, method='Nelder-Mead')
        print("res.x=", res.x)

        N, T, x0, mu, sigma = res.x[0:5]
        if nur > 0:
            temp_rgs[unreliable_indeces] = res.x[5:]
        scales = self.props

        lnp_params = np.concatenate([(N, T, x0, me, mp, mu, sigma), scales])
        title = "lnp_column_objective result"
        ret = plot_lognormal_fitting_state(x, y, lnp_params, temp_rgs, title=title)

    def minima_callback(self, x, f, accept):
        self.minima_counter += 1
        print("%s: minima_counter=%d" % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), self.minima_counter))