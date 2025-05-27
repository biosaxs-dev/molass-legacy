# coding: utf-8
"""
    SecTheory.SinglePoreNonCf.py

    Copyright (c) 2022, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import basinhopping
import molass_legacy.KekLib.DebugPlot as plt
from DataUtils import get_in_folder
from SecTheory.BasicModels import single_pore_elution

def single_pore_pdf(x, rg, rp, x0, nperm, tperm, me, mp):
    return single_pore_elution(x - x0, rg, rp, nperm, tperm, me, mp)

class SinglePoreSolver:
    def __init__(self, x, y, n_species, rg_curve=None):
        self.x = x
        self.y = y
        self.area = np.sum(y)
        self.n_species = n_species
        self.rg_curve = rg_curve
        if rg_curve is not None:
            self.mask = rg_curve.get_mask()
            self.xm, self.ym, self.valid_rg = rg_curve.get_valid_curves()
            self.rg_area = np.sum(self.valid_rg)
            print("area=", self.area, "rg_area=", self.rg_area)

    def split_params(self, params):
        sep1 = 4
        sep2 = sep1 + self.n_species
        rp_params = params[0:sep1]
        rg_params = params[sep1:sep2]
        wt_params = params[sep2:]
        return rp_params, rg_params, wt_params

    def objective_function(self, params, debug=False):
        x = self.x
        y = self.y
        rp_params, rg_params, wt_params = self.split_params(params)

        ty = np.zeros(len(y))
        penalty = 0
        tw = 0
        last_rg = None
        cy_list = []
        for k, rg in enumerate(rg_params):
            if k < len(wt_params):
                weight = wt_params[k]
            else:
                weight = 1 - tw
            tw += weight
            cy = weight*self.area*single_pore_pdf(x, rg, *rp_params, 1, 1)
            cy_list.append(cy)
            ty += cy
            penalty += min(0, weight - 1e-3)**2
            if last_rg is not None:
                penalty += min(0, last_rg - rg)**2
            last_rg = rg

        elution_dev = np.log10(np.sum((ty - y)**2)/self.area)/2

        valid_rg = self.valid_rg
        t_rg = np.zeros(len(valid_rg))
        tym = ty[self.mask]
        for r, xr_cy in zip(rg_params, cy_list):
            t_rg += r * xr_cy[self.mask]/tym

        rg_dev = np.log10(np.sum((t_rg -valid_rg)**2)/self.rg_area)/2   # equivalent to np.log10(np.sqrt(...

        fv = elution_dev + rg_dev + penalty*1e8
        if debug:
            print("penalty=", penalty)
            print("fv=", fv)
            in_folder = get_in_folder()
            rg_estimation = np.zeros(len(y))
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.set_title("Single Pore Model Fit for %s FV=%.3g" % (in_folder, fv), fontsize=20)
                ax.plot(x, y, color="orange", label="Data")
                tw = 0
                for k, cy in enumerate(cy_list):
                    if k < len(wt_params):
                        weight = wt_params[k]
                    else:
                        weight = 1 - tw
                    rg = rg_params[k]
                    rg_estimation += rg*cy/ty
                    print([k], "rg=%.3g, weight=%.3g" % (rg, weight))
                    tw += weight
                    ax.plot(x, cy, ":", lw=2, label="Rg=%.3g" % rg)
                ax.plot(x, ty, ":", lw=2, color="red", label="Model Total")
                xmin, xmax = ax.get_xlim()
                ymin, ymax = ax.get_ylim()
                ymax = max(np.max(y), np.max(ty))*1.2
                ax.set_ylim(ymin, ymax)

                x0 = rp_params[1]
                ax.plot([x0, x0], [ymin, ymax], color="yellow", label="x0")
                ax.legend()

                w = 0.05
                tx = xmin*(1-w) + xmax*w
                w = 0.9
                ty = ymin*(1-w) + ymax*w
                ax.text(tx, ty, "Pore Size=%.3g" % rp_params[0], fontsize=30, alpha=0.2)
                for name, v in zip(["rp", "x0", "nperm", "tperm"], rp_params):
                    print("%s = %.3g" % (name, v))
                print("fv=%g" % fv)
                print("elution_dev=%g" % elution_dev, "rg_dev=%g" % rg_dev, "penalty=%g" % penalty)
                axt = ax.twinx()
                axt.grid(False)
                if self.rg_curve is not None:
                    for k, (x_, y_, rg_) in enumerate(self.rg_curve.get_curve_segments()):
                        label = 'Rg (guinier)' if k == 0 else None
                        axt.plot(x_, rg_, color='gray', alpha=0.5, label=label)

                axt.plot(x, rg_estimation, ":", lw=2, color="gray", label="Rg (model fit)")
                axt.set_ylim(10, rg_params[0]*1.2)
                axt.legend(loc="center right")
                ret = plt.show()
                if not ret:
                    debug = False
        return fv

    def optimize(self, init_params, n_iters=100):
        ret = basinhopping(self.objective_function, init_params, niter=n_iters)
        return ret

def fit_single_pore_pdf(ecurve, rg_curve, init_params, n_species=3, n_iters=100, n_trials=0):
    import warnings
    warnings.filterwarnings("ignore")

    x = ecurve.x
    y = ecurve.y

    solver = SinglePoreSolver(x, y, n_species, rg_curve=rg_curve)

    ret = None
    best_fv = None
    for k in range(n_trials):
        print([k], "optimizing ...")
        ret = solver.optimize(init_params, n_iters=n_iters)
        if best_fv is None or ret.fun < best_fv:
            best_fv = ret.fun
            print([k], "best_fv=%g" % best_fv)
            init_params = ret.x

    solver.objective_function(init_params, debug=True)
