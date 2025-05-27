"""
    Basseline.BoBaseline.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.Baseline.Constants import SLOPE_SCALE

USE_BAYESIAN_OPTIMIZION = True

def bo_baseline_trial_impl(caller):
    from importlib import reload
    import UV.UvPreRecog
    reload(UV.UvPreRecog)
    from molass_legacy.UV.UvPreRecog import UvPreRecog
    from molass_legacy.Trimming.PreliminaryRecognition import PreliminaryRecognition
    print('bo_baseline_trial_impl')

    sd = caller.serial_data
    pre_recog = PreliminaryRecognition(sd)

    upr = UvPreRecog(sd, pre_recog, debug=True)

    base_curve = upr.base_curve
    base_params = upr.init_params.copy()
    base_params[4:6] /= SLOPE_SCALE
    uv_curve = sd.get_uv_curve()
    x = uv_curve.x
    y = uv_curve.y
    ty = None
    by = base_curve(x, base_params, ty)

    print("baseline_type=", base_curve.baseline_type)
 
    def plot_baseline(title, plot_basepart=True):
        with plt.Dp():
            fig, ax = plt.subplots()
            ax.set_title(title)
            ax.plot(x, y)
            if plot_basepart:
                ax.plot(tbx, tby)
            ax.plot(x, by, color='red')
            fig.tight_layout()
            plt.show()

    tbx, tby = base_curve.base_xy
    plot_baseline("bo_baseline_trial_impl: before minimize")
    return

    def objective_func(params):
        y_ = base_curve(tbx, params, ty)
        return np.sum((y_ - tby)**2)

    if USE_BAYESIAN_OPTIMIZION:
        import logging
        from ultranest import ReactiveNestedSampler
        from molass_legacy.Optimizer.OptimizerUtils import OptimizerResult
        allow = 10
        upper = base_params - allow
        lower = base_params + allow

        def my_prior_transform(cube):
            params = cube * (upper - lower) + lower
            return params
        def my_likelihood(params):
            return -objective_func(params)

        num_params = len(base_params)
        param_names = ["p%02d" % i for i in range(num_params)]
        sampler = ReactiveNestedSampler(param_names, my_likelihood, my_prior_transform)
        sampler.logger.setLevel(logging.INFO)       # to suppress debug log
        result = sampler.run(min_num_live_points=100, max_ncalls=20000)
        opt_params = np.array(result['maximum_likelihood']['point'])
        res = OptimizerResult(x=opt_params)
    else:
        from scipy.optimize import basinhopping
        minimizer_kwargs = dict(method='Nelder-Mead')
        res = basinhopping(objective_func, base_params, minimizer_kwargs=minimizer_kwargs)

    by = base_curve(x, res.x, ty)
    plot_baseline("bo_baseline_trial_impl: after minimize")
