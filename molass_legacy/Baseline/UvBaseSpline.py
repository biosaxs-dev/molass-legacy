"""
    UV.UvBaseSpline.py

    Copyright (c) 2022-2024, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import UnivariateSpline
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.Trimming.Sigmoid import guess_bent_sigmoid, ex_sigmoid, adjust_ex_sigmoid
from molass_legacy.Peaks.EghSupples import egh, d_egh
from molass_legacy.KekLib.SciPyCookbook import smooth
from molass_legacy._MOLASS.SerialSettings import get_setting
from .Constants import SLOPE_SCALE

SMOOTHING = True
PENALTY_SCALE = 1e8
CERTAINTY_LIMIT = 0.5
BETTER_LOWER_NEEDED_RATIO = 0.5
TRY_ADJUST_EX_SIGMOID = True

class UvBaseSpline:
    def __init__(self, sd, curve1=None, spline=None):
        self.sd = sd
        self.curve1 = curve1
        if spline is None:
            spline = curve1.d1
        self.diff_spline = spline
        self.baseline_type = get_setting("unified_baseline_type")
        self.lower_xy = None
        self.base_xy = None

    def guess_params(self, curve2, pp, slice_, fc_point=None, debug=False):
        x = curve2.x
        y = curve2.y

        x_ = x[slice_]
        if SMOOTHING:
            y_ = smooth(y[slice_])
        else:
            y_ = y[slice_]

        certainty = curve2.sigmoid_certainty
        if certainty < CERTAINTY_LIMIT:
            sigm_params = None
            if debug:
                print("(1) certainty=", certainty)
            if certainty == 0:
                if debug:
                    from importlib import reload
                    import Trimming.SigmoidApplicability
                    reload(Trimming.SigmoidApplicability)
                from molass_legacy.Trimming.SigmoidApplicability import check_applicability
                ok, info = check_applicability(x_, y_, debug=debug)
                if not ok:
                    sigm_params = list(info.safe_params)
            if sigm_params is None:
                sigm_params = list(guess_bent_sigmoid(x_, y_, debug=debug))
        else:
            sigm_params = list(curve2.sigmoid_params)

        if debug:
            print("(2) certainty=", certainty, CERTAINTY_LIMIT)
            print("fc_point=", fc_point)
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.set_title("guess_params (1)")
                ax.plot(x, y, label="data")
                ax.plot(x_, y_, label="smoothed")
                ax.plot(x, ex_sigmoid(x, *sigm_params), label="ex_sigmoid")
                ax.legend()
                fig.tight_layout()
                plt.show()

        if fc_point is None:
            force_L0_x0 = True
            sigm_params[0:2] = (0,0)    # this modification should be considerd earlier
        else:
            force_L0_x0 = False         

        ty = None       # dummy arg which will not be used here

        def obj_func(p, debug=False):
            fy = self.__call__(x_, p, ty)
            fv = np.sum((fy - y_)**2)
            if debug:
                print("fv=", fv)
                with plt.Dp():
                    fig, ax = plt.subplots()
                    ax.plot(x_, y_)
                    ax.plot(x_, fy)
                    fig.tight_layout()
                    plt.show()
            if force_L0_x0:
                # or consider using bounds
                fv += PENALTY_SCALE*np.sum(p[0:2]**2)   # L, x0 penalty
            return fv

        opt_ret = None
        for scale in [-1, 1]:
            init_params = (*sigm_params, scale)
            ret = minimize(obj_func, init_params, method='Nelder-Mead')
            # ret = basinhopping(obj_func, init_params)
            print("ret.x=", ret.x)
            if opt_ret is None or ret.fun < opt_ret.fun:
                opt_ret = ret

        print("opt_ret.fun=", opt_ret.fun)

        x1 = self.curve1.x
        y1 = self.curve1.y
        # use lower part only
        lower = y1 < 0.1* self.curve1.max_y
        x1_ = x1[lower]
        y1_ = y1[lower]
        lower_ratio = len(x1_)/len(x1)
        if debug:
            print("lower_ratio=", lower_ratio)
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.set_title("lower part check")
                ax.plot(x1, y1, label="data")
                ax.plot(x1_, y1_, ":", label="lower part")
                ax.legend()
                fig.tight_layout()
                plt.show()

        if TRY_ADJUST_EX_SIGMOID:
            if lower_ratio < BETTER_LOWER_NEEDED_RATIO:
                # as in 20191118_4
                x1_, y1_ = self.make_better_lower_part(self.curve1)
            adjusted_params = adjust_ex_sigmoid(x1_, y1_, opt_ret.x)
        else:
            adjusted_params = opt_ret.x

        if debug:
            with plt.Dp():
                fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
                ax1.set_title("sliced data")
                ax1.plot(x, y, label="data")
                ax1.plot(x_, y_, label="sliced smooth data")
                ax1.legend()
                ax2.set_title("after adjust_ex_sigmoid")
                ax2.plot(x1, y1, label="data")
                ax2.plot(x1_, y1_, ":", label="sliced data")
                ax2.plot(x1, ex_sigmoid(x1, *opt_ret.x[:-1]), label="optimized params")
                ax2.plot(x1, ex_sigmoid(x1, *adjusted_params[:-1]), label="adjusted params")
                ax2.legend()
                fig.tight_layout()
                plt.show()

        self.lower_xy = x1_, y1_
        if fc_point is None:
            self.base_xy = self.lower_xy
        else:
            # as in 20171203
            extended_base = np.logical_or(lower, x <= fc_point)
            self.base_xy = x[extended_base], y[extended_base]

        if debug:
            from molass_legacy.UV.PlainCurveUtils import get_both_wavelengths

            sd = self.sd
            wavelengths = get_both_wavelengths(sd.lvector)
            with plt.Dp():
                from bisect import bisect_right
                from molass_legacy.SerialAnalyzer.DataUtils import get_in_folder
                from molass_legacy.Elution.CurveUtils import simple_plot
                from molass_legacy.DataStructure.MatrixData import simple_plot_3d
                fig = plt.figure(figsize=(18,10))
                fig.suptitle("guess_params (2) for %s" % get_in_folder(), fontsize=20)
                ax1 = fig.add_subplot(221)
                ax2 = fig.add_subplot(222)
                ax3 = fig.add_subplot(223, projection="3d")
                ax4 = fig.add_subplot(224)

                ax1.set_title(r"Elution at $\lambda=%.3g$" % wavelengths[0], fontsize=16)
                simple_plot(ax1, self.curve1, legend=False)
                by280 = self.__call__(x, adjusted_params, ty)
                by400 = self.__call__(x, opt_ret.x, ty)
                ax1.plot(x, by280, color="red", label="baseline at 280")
                ax1.plot(x, by400, ":", color="red", label="baseline at 400")
                ax1.legend(fontsize=16)
                ax1.set_xlabel("Eno")

                ax2.set_title(r"Elution at $\lambda=%.3g$" % wavelengths[1], fontsize=16)
                ax2.plot(x, y)
                if SMOOTHING:
                    ax2.plot(x_, y_)
                fy = self.__call__(x, opt_ret.x, ty)
                ax2.plot(x, fy)
                ax2.set_xlabel("Eno")

                ax3.set_title("3D View", fontsize=16)
                ax4.set_title("Complementary View", fontsize=16)
                ax4.set_xlabel(r"$\lambda$")
                wv = sd.lvector
                U = sd.conc_array
                simple_plot_3d(ax3, U, x=wv)
                ones = np.ones(len(wv))
                for j, color in (len(x)-1, "cyan"), (0, "pink"):
                    z = U[:,j]
                    ax3.plot(wv, ones*j, z, color=color)
                    ax4.plot(wv, z, color=color, label="j = %d" % j)

                ymin, ymax = ax4.get_ylim()
                ax4.set_ylim(ymin, ymax)
                ones = np.ones(len(x))
                jj = np.arange(len(x))

                for w in wavelengths:
                    i = bisect_right(wv, w)
                    ax3.plot(ones*w, jj, U[i,:], color="yellow")
                    ax4.plot([w, w], [ymin, ymax], label=r"$\lambda=%.3g$" % w, color="yellow")

                ax4.legend(fontsize=16)
                fig.tight_layout()
                plt.show()

        if self.baseline_type == 1:
            ret_params = adjusted_params
        elif self.baseline_type == 2:
            intreg_scale = 0
            ret_params = np.concatenate([adjusted_params, [intreg_scale]])
        elif self.baseline_type == 3:
            intreg_scale = 0
            ret_params = np.concatenate([adjusted_params, [intreg_scale]])
        else:
            assert False

        ret_params[4:6] *= SLOPE_SCALE
        return ret_params

    def guess_integral_scale(self, x, y, params, ty=None, debug=False):
        assert len(params) == 7
        if ty is None:
            by = self.__call__(x, params, y)    # y won't be used
            ty = y - by

        opt_params = np.concatenate([params, [0]])

        def objective(p):
            opt_params[-1] = p[0]
            integ_base = self.__call__(x, opt_params, ty)   # y will be used
            return (integ_base[-1] - y[-1])**2 + PENALTY_SCALE*min(0, opt_params[-1])**2

        ret = minimize(objective, (0,))
        ret.x[0] = max(0, ret.x[0])     # force it to be non-negative

        if debug:
            opt_params[-1] = ret.x[0]
            integ_base = self.__call__(x, opt_params, ty) 
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.set_title("guess_integral_scale")
                ax.plot(x, y)
                ax.plot(x, integ_base, color="red")
                fig.tight_layout()
                plt.show()

        return ret.x[0], ty

    def make_better_lower_part(self, curve, debug=False):
        from molass_legacy.Baseline.ScatteringBaseline import ScatteringBaseline
        x = curve.x
        y = curve.y
        sbl = ScatteringBaseline(y, x=x)
        A, B = sbl.solve()
        y_ = A*x+B
        if curve.is_roughly_centered() and abs(y_[-1] - y[-1])/curve.max_y > 0.05:
            # as in 20160628
            # this works, but more general solution is desirable
            # print("retrying make_better_lower_part")
            A, B = sbl.solve(p_final=50)
            y_ = A*x+B        
        if debug:
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.set_title("make_better_lower_part")
                ax.plot(x, y)
                ax.plot(x, y_, color="red")
                fig.tight_layout()
                plt.show()
        return x, y_

    def __call__(self, x, arg_params, ty, cy_list=None):
        params = arg_params.copy()
        params[4:6] /= SLOPE_SCALE
        k = 7
        y = compute_baseline_impl(x, params[0:k], self.diff_spline)
        if len(params) == 8:
            if cy_list is None:
                y += params[k]*np.cumsum(ty)
            else:
                for cy in cy_list:
                    y += params[k]*np.cumsum(cy)
        return y
           
    def get_shifted(self, n):
        x = self.curve1.x
        dy = self.diff_spline(x)
        shifted_spline = UnivariateSpline(x-n, dy, s=0, ext=3)
        return UvBaseSpline(self.sd, spline=shifted_spline)

def compute_baseline_impl(x, params, diff_spline):
    """
        requires 6 + 1 params
    """
    y = np.zeros(len(x))
    y += ex_sigmoid(x, *params[0:-1])
    y += params[-1]*diff_spline(x)
    return y