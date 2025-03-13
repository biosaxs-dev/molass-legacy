"""
    Optimizer.UvBaselineEstimator.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import logging
import numpy as np
from scipy.optimize import basinhopping
LOG_IDENTIFICATION = "FixedBaseline Optimizer"

class UvBaselineEstimator:
    def __init__(self, optimizer):
        self.logger = logging.getLogger(__name__)
        self.optimizer = optimizer
        self.logger.info("%s constructed", LOG_IDENTIFICATION)

    def prepare(self, init_params, real_bounds=None):
        self.optimizer.prepare_for_optimization(init_params, real_bounds=real_bounds)
        nc = self.optimizer.num_pure_components

        model_name = self.optimizer.get_model_name()

        if model_name == "EGH":
            xr_start = nc*5
        elif model_name == "SDM":
            xr_start = nc
        elif model_name == "EDM":
            xr_start = nc*7
        else:
            assert False

        xr_stop = xr_start + 2
        uv_start = xr_stop + nc
        uv_indeces = np.arange(uv_start, uv_start+7)
        self.indeces = uv_indeces
        self.logger.info("%s prepared with model=%s, indeces=%s", LOG_IDENTIFICATION, model_name, str(list(self.indeces)))

    def solve(self, init_params, real_bounds=None, niter=100, seed=None, open_mode="w", debug=False):
        self.logger.info("FixedBaseline solve started with init_params=%s", str(init_params))
        self.prepare(init_params, real_bounds=real_bounds)

        self.cb_fh = open("callback.txt", open_mode)
        self.optimizer.cb_fh = self.cb_fh
        self.optimizer.write_init_callback_txt(init_params)

        self.temp_params = self.optimizer.to_norm_params(init_params)

        bounds = [(0, 10)]*len(self.indeces)
        minimizer_kwargs = dict(method='Nelder-Mead', bounds=bounds)
        result = basinhopping(self.objective_func_wrapper, self.temp_params[self.indeces],
                              callback=self.minima_callback,
                              niter=niter,
                              seed=seed,
                              minimizer_kwargs=minimizer_kwargs)

        self.cb_fh.close()
        self.temp_params[self.indeces] = result.x
        result.x = self.temp_params
        return result
    
    def objective_func_wrapper(self, p):
        self.temp_params[self.indeces] = p
        real_params = self.optimizer.to_real_params(self.temp_params)
        return self.optimizer.objective_func(real_params)
    
    def minima_callback(self, x, f, accept):
        self.temp_params[self.indeces] = x
        return self.optimizer.minima_callback(self.temp_params, f, accept)

def test_estimate_uvbaseline_impl(optimizer, init_params):
    print("test_estimate_baseline_impl")
    optimizer.objective_func(init_params, plot=True)
    estimator = UvBaselineEstimator(optimizer)
    result = estimator.solve(init_params)
    optimizer.objective_func(result.x, plot=True)