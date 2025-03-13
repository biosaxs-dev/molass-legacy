"""
    Optimizer.StrategicOptimizer.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import logging
from scipy.optimize import basinhopping
LOG_IDENTIFICATION = "Strategic Optimizer"

class StrategicOptimizer:
    def __init__(self, optimizer, indeces):
        self.logger = logging.getLogger(__name__)
        self.optimizer = optimizer
        self.indeces = indeces
        self.logger.info("%s constructed with indeces=%s", LOG_IDENTIFICATION, str(indeces))

    def solve(self, init_params, real_bounds=None, niter=100, seed=None, open_mode="w", debug=False):
        self.logger.info("strategic solve started with init_params=%s", str(init_params))
        self.optimizer.prepare_for_optimization(init_params, real_bounds=real_bounds)

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
