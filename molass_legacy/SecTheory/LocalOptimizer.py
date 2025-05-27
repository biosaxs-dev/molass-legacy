"""
    SecTheory.LocalOptimizer.py

    Copyright (c) 2022-2024, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize, basinhopping
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.Optimizer.NumericalUtils import safe_ratios
from molass_legacy.SerialAnalyzer.DataUtils import get_in_folder
from molass_legacy._MOLASS.SerialSettings import get_setting

PENALTY_SCALE = 1e5
NUM_SECCOL_PARAMS = 6
MEMP_LIMIT = 3.1
N_LIMIT = 3000

def split_params(p, nc):
        return (
            p[0:NUM_SECCOL_PARAMS],
            p[NUM_SECCOL_PARAMS:NUM_SECCOL_PARAMS+nc], 
            p[NUM_SECCOL_PARAMS+nc:NUM_SECCOL_PARAMS+nc*2]
            )

class RgInfo:
    def __init__(self, rg_curve, nc):
        trs, _, rgs = rg_curve.get_valid_curves()
        self.mask = rg_curve.get_mask()
        self.rg_weights = rg_curve.get_weights()

        mean = np.mean(rgs)
        sigma = np.std(rgs)

        # self.rg_lower = min(mean - 2*sigma, np.min(rgs))/2
        self.rg_lower = 3
        self.rg_upper = max(mean + 2*sigma, np.max(rgs))*2
        print("----------- self.rg_lower=", self.rg_lower)
        print("----------- self.rg_upper=", self.rg_upper)
        self.trs = trs
        self.rgs = rgs
        self.zeros_rg = np.zeros(nc-1)
        self.ones_rgs = np.ones(len(rgs))

    def get_order_penalty(self, rg):
        return np.sum(np.min([self.zeros_rg, rg[:-1] - rg[1:]], axis=0)**2)

    def get_bound_penalty(self, rg):
        return min(0, np.min(rg) - self.rg_lower)**2 + min(0, self.rg_upper - np.max(rg))**2

class SecInfo:
    def __init__(self, t0_upper_bound=None):
        self.d_init = get_setting("poresize")
        rp_lower, rp_upper = get_setting("poresize_bounds")
        self.d_lower = rp_lower
        self.d_upper = rp_upper
        self.t0_upper_bound = t0_upper_bound
        self.t0_init = 0 if t0_upper_bound is None else t0_upper_bound*0.7
        self.N = 500

    def get_sec_params(self, K):
        N = self.N
        return self.t0_init, self.d_init, N, 1, K/N, 1

class LocalOptimizer:
    def __init__(self, ecurve, rg_info, sec_info, nc, elutionmodel_func, logger=None):
        self.ecurve = ecurve
        self.rg_info = rg_info
        self.sec_info = sec_info
        self.nc = nc
        self.elutionmodel_func = elutionmodel_func
        if logger is None:
            import logging
            logger = logging.getLogger(__name__)
        self.logger = logger
        self.init_tr = None
        self.abort = False

    def objective_all(self, p, get_curves=False, debug=False, debug_title="all"):
        ecurve = self.ecurve
        x = ecurve.x
        y = ecurve.y
        max_y = ecurve.max_y

        rgs = self.rg_info.rgs

        d_init = self.sec_info.d_init
        d_lower = self.sec_info.d_lower
        d_upper = self.sec_info.d_upper
        t0_init = self.sec_info.t0_init

        (t0, rp, N, me, T, mp), rg, w = split_params(p, self.nc)
        if debug:
            print("(t0, rp, N, me, T, mp)=", (t0, rp, N, me, T, mp))
            print("rg=", rg)
            print("w=", w)

        rho = rg/rp
        rho[rho > 1] = 1
        ty = np.zeros(len(x))

        model_trs = t0 + N*T*(1 - rho)**(me + mp)

        if self.init_tr is None:
            tr_decency = 0
        else:
            tr_decency = np.sum((model_trs - self.init_tr)**2)

        penalty = min(0, N)**2 + min(0, T)**2 

        cy_list = []
        for w_, r_ in zip(w, rho):
            penalty += min(0, w_)**2
            np_ = N*(1 - r_)**me
            tp_ = T*(1 - r_)**mp
            cy = w_*self.elutionmodel_func(x - t0, np_, tp_)
            ty += cy
            cy_list.append(cy)

        if get_curves:
            return cy_list, ty

        penalty += self.rg_info.get_order_penalty(rg)
        penalty += self.rg_info.get_bound_penalty(rg)
        penalty += min(0, rp - d_lower)**2 + min(0, d_upper - rp)**2
        penalty += min(0, MEMP_LIMIT - (me+mp))**2 + min(0, me)**2 + min(0, mp)**2
        t0_upper_bound = self.sec_info.t0_upper_bound
        if t0_upper_bound is not None:
            penalty += max(0, t0 - t0_upper_bound)**2

        mask = self.rg_info.mask

        tym = ty[mask]
        t_rg = np.zeros(len(tym))

        for r, cy in zip(rg, cy_list):
            t_rg += r * safe_ratios(self.rg_info.ones_rgs, cy[mask], tym)

        xr_fitting = np.log10(np.sum((ty - y)**2))
        if not np.isfinite(xr_fitting):
            xr_fitting = PENALTY_SCALE
            penalty += 1
        rg_fitting = 0.1*np.log10(np.average((self.rg_info.rg_weights*(t_rg - rgs)/t_rg)**2))
        if not np.isfinite(rg_fitting):
            rg_fitting = PENALTY_SCALE
            penalty += 1
        penalty *= PENALTY_SCALE
        fv = xr_fitting + rg_fitting + penalty + tr_decency
        if np.isnan(fv):
            fv = penalty

        if debug:
            print("xr_fitting=", xr_fitting)
            print("rg_fitting=", rg_fitting)
            print("penalty=", penalty)
            print("fv=", fv)
            rg_info = self.rg_info
            trs = rg_info.trs
            rg_upper = rg_info.rg_upper

            in_folder = get_in_folder()

            with plt.Dp():
                fig, ax = plt.subplots()
                ax.set_title("%s: Decomposition %s" % (debug_title, in_folder), fontsize=16)
                axt = ax.twinx()
                axt.grid(False)
                ax.plot(x, y, label="data")
                axt.plot(trs, rgs, color="gray", alpha=0.5, label="observed rg")
                axt.plot(trs, t_rg, ":", color="black", label="reconstructed rg")
                axt.set_ylim(0, rg_upper)

                print("t0, rp, N, me, T, mp=", t0, rp, N, me, T, mp)
                print("rg=", rg)
                print("w=", w)

                for k, (cy, tr) in enumerate(zip(cy_list, model_trs)):
                    ax.plot(x, cy, ":", label="component-%d" % (k+1))
                    if np.isfinite(tr):
                        j = int(tr)
                        if j >= 0 and j < len(cy):
                            ax.plot(tr, cy[j], "o", color="yellow")
                ax.plot(x, ty, ":", color="red", label="component-total")

                ax.legend(fontsize=14)
                axt.legend(loc="upper left", fontsize=14)
                fig.tight_layout()
                ret = plt.show()
            if not ret:
                self.abort = True

        return fv

    def optimize(self, init_params, params_given=False, init_tr=None, global_opt=False, debug=False):
        self.init_tr = init_tr
        if global_opt:
            if debug:
                self.objective_all(init_params, debug=True)

            ret = basinhopping(self.objective_all, init_params)
        else:
            init_fv = self.objective_all(init_params, debug=debug)
            print("init_fv=", init_fv)

            if params_given:
                ret = minimize(self.objective_all, init_params)
                self.logger.info("estimate_monopore_params: minimize success=%s, status=%d, message=%s", str(ret.success), ret.status, ret.message)
            else:
                k = 0
                # (may be default) method="BFGS" seems best
                # , method="Nelder-Mead"
                ret = minimize(self.objective_all, init_params)
                print([k], "ret.fun=", ret.fun, "ret.success=", ret.success, "ret.message=", ret.message)

                if ret.success:
                    self.logger.info("estimate_monopore_params: minimize[%d] ok with fv=%g", k, ret.fun)
                    # break
                else:
                    self.logger.info("estimate_monopore_params: minimize[%d] failed with fv=%g, status=%d, message=%s", k, ret.fun, ret.status, ret.message)

        if debug:
            self.objective_all(ret.x, debug=True)

        return ret
