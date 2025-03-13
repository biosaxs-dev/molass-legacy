"""
    SecTheory.InitialGuess.py

    Copyright (c) 2022-2024, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import UnivariateSpline
from molass_legacy.KekLib.GeometryUtils import rotated_argminmax
import molass_legacy.KekLib.DebugPlot as plt
from .CumsumInverse import cumsum_inverse

class InitialGuess:
    def __init__(self, ecurve, rg_info, sec_info, localopt, lrf_src=None, debug=False):
        self.sec_info = sec_info

        if lrf_src is not None:
            secparams = lrf_src.guess_secparams()

        x = ecurve.x
        y = ecurve.y
        sy = ecurve.sy
        cum_y = np.cumsum(sy)
        cum_y /= cum_y[-1]
        cum_spline = UnivariateSpline(x, cum_y, s=0, ext=3)

        k = 12
        def sigmoid(z):
            return 1/(1 + np.exp(-k*(z - 0.5)))

        nc = localopt.nc
        d_init = sec_info.d_init
        t0_init = sec_info.t0_init
        self.zeros_w = np.zeros(nc)

        init_y = np.linspace(0, 1, nc+2)[1:-1]
        init_ys = sigmoid(init_y)
        print("init_ys=", init_ys)

        m1, m2 = rotated_argminmax(-np.pi/64, cum_y, debug=False)
        init_tr = cumsum_inverse(cum_spline, init_ys, *x[[m1, m2]])
        rho_ = min(1, rg_info.rgs[-1]/d_init)
        K = (init_tr[-1] - t0_init)/(1 - rho_*0.5)
        print("init_tr=", init_tr)
        self.init_tr =init_tr

        if debug:
            with plt.Dp():
                fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
                fig.suptitle("Initial Guess", fontsize=20)
                axt = ax2.twinx()
                axt.grid(False)

                ax1.set_title("Proportion Change Sigmoid (k=%d)" % k, fontsize=16)
                ax2.set_title("Integral Curve Mapped Points", fontsize=16)

                z = np.linspace(0, 1, 20)
                ax1.plot(z, sigmoid(z))
                ax1.plot(init_y, init_ys, "o", color="orange")

                ax2.plot(x, y)
                ax2.plot(x, ecurve.spline(x), ":")
                ymin, ymax = ax2.get_ylim()
                ax2.set_ylim(ymin, ymax)

                for px in [t0_init, K]:
                    ax2.plot([px, px], [ymin, ymax], color="green")

                for tr in init_tr:
                    ax2.plot([tr, tr], [ymin, ymax], color="yellow")

                axt.plot(x, cum_y, color="cyan")
                axt.plot(init_tr, cum_spline(init_tr), "o", color="pink")
                # axt.plot(init_tr, init_ys, "o", color="red")
                # ax2.plot(init_tr, ret.x, "o", color="orange")

                ymin, ymax = axt.get_ylim()
                ax1.set_ylim(ymin, ymax)
                axt.set_ylim(ymin, ymax)
                xmin, xmax = ax1.get_xlim()
                ax1.set_xlim(xmin, xmax)

                for y_, ys_ in zip(init_y, init_ys):
                    ax1.plot([y_, y_], [ymin, ys_], ":", color="pink")
                    ax1.plot([y_, xmax], [ys_, ys_], ":", color="pink")

                xmin, xmax = axt.get_xlim()
                axt.set_xlim(xmin, xmax)
                for ys_, tr_ in zip(init_ys, init_tr):
                    axt.plot([xmin, tr_], [ys_, ys_], ":", color="pink")

                fig.tight_layout()
                ret = plt.show()

            if not ret:
                raise RuntimeError("InitialGuess: Debug plot was canceled.")

        self.init_rg = d_init * (1 - np.power((init_tr - t0_init)/K, 1/2) )
        init_w_ = np.max([self.zeros_w, ecurve.spline(init_tr)], axis=0)
        self.init_seccol_params = sec_info.get_sec_params(K)

        def objective_scales(p, get_curves=False, debug=False):
            all_params = np.concatenate([self.init_seccol_params, self.init_rg, p])
            return localopt.objective_all(all_params, get_curves=get_curves, debug=debug, debug_title="scales")

        # adjust the scale parameters so that the areas take the same size.
        # this adjustment makes the choice of elution models safe or irrelevant.
        # i.e., both pdf and scaled pdf can be used interchangeably
        cy_list, ty = objective_scales(init_w_, get_curves=True)
        area_ratio = np.sum(y)/np.sum(ty)
        init_w_ *= area_ratio

        if debug:
            print("---------------- init_seccol_params=", self.init_seccol_params)
            print("---------------- init_rg=", self.init_rg)
            print("---------------- init_w_=", init_w_)
            init_fv = objective_scales(init_w_, debug=debug)
            print("---------------- init_fv=", init_fv)
            if localopt.abort:
                raise RuntimeError("InitialGuess: Debug plot was canceled before optimization.")

        # method="Nelder-Mead"
        self.ret = minimize(objective_scales, init_w_)

        if debug:
            ret = self.ret
            print("---------------- weights optimization: ret.success=", ret.success)
            print("---------------- weights optimization: ret.message=", ret.message)

            objective_scales(ret.x, debug=debug)
            if localopt.abort:
                raise RuntimeError("InitialGuess: Debug plot was canceled after optimization.")

    def get_params(self):
        init_w = np.max([self.zeros_w, self.ret.x], axis=0)
        return np.concatenate([self.init_seccol_params, self.init_rg, init_w])


class InitialGuessAdapter:
    def __init__(self, ret):
        pass