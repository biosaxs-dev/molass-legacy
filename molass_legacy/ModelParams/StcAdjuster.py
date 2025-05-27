"""
    StcAdjuster.py

    Copyright (c) 2022, SAXS Team, KEK-PF
"""
from scipy.optimize import minimize
import molass_legacy.KekLib.DebugPlot as plt

class StcAdjuster:
    def __init__(self):
        pass

    def fit(self, fullopt, fv_array, x_list, debug=False):
        params = x_list[0]
        print(fv_array.shape, len(x_list), params.shape)

        n_iterations = 10   # any size > 0

        n = fullopt.n_components
        nc = n - 1
        i = nc + 2 + nc + 1                 # b
        j = nc + 2 + nc + 2 + nc + 7 + 2    # t0
        fv = fv_array[0]

        fit_params = params.copy()

        def objective(p):
            fit_params[[i,j]] = p
            fullopt.prepare_for_optimization(fit_params)
            return (fullopt.objective_func(fit_params) - fv)**2

        xr_curve = fullopt.xr_curve
        b, t0 = params[[i,j]]
        ret = minimize(objective, (b, xr_curve.x[0] + t0))
        fit_params[[i,j]] = ret.x

        if debug:
            print("b : %g → %g" % (b, ret.x[0]))
            print("t0: %g → %g" % (t0, ret.x[1]))
            with plt.Dp():
                fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(12,10))
                for i, p in enumerate([params, fit_params]):
                    ax1 = axes[i,0]
                    ax2 = axes[i,1]
                    fullopt.prepare_for_optimization(p)
                    fullopt.objective_func(p, plot=True, axis_info=(fig, (ax1, ax2, None, None)))
                fig.tight_layout()
                plt.show()

        self.fit_ij = [i,j]
        self.fit_delta = ret.x - params[[i,j]]

    def convert(self, params):
        if type(params) is list:
            params = [self.convert(p) for p in params]
        else:
            params = params.copy()
            for k in range(2):
                params[self.fit_ij[k]] += self.fit_delta[k]
        return params
