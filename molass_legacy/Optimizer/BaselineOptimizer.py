"""
    Optimizer.BaselineOptimizer.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import logging
from scipy.optimize import basinhopping
LOG_IDENTIFICATION = "Baseline Optimizer"

class BaselineOptimizer:
    def __init__(self, optimizer):
        self.logger = logging.getLogger(__name__)
        self.optimizer = optimizer
        self.logger.info("%s constructed", LOG_IDENTIFICATION)

    def solve(self, init_params, baseline_indeces, real_bounds=None):
        self.indeces = baseline_indeces
        self.logger.info("strategic solve started with init_params=%s", str(init_params))
        self.optimizer.prepare_for_optimization(init_params, real_bounds=real_bounds)
        self.optimizer.cb_fh = self.cb_fh
        self.optimizer.write_init_callback_txt(init_params)

        self.temp_params = self.optimizer.to_norm_params(init_params)

        bounds = [(0, 10)]*len(self.indeces)
        minimizer_kwargs = dict(method='Nelder-Mead', bounds=bounds)
        result = basinhopping(self.baseline_objective, self.temp_params[self.indeces],
                              callback=self.minima_callback,
                              niter=20,
                              minimizer_kwargs=minimizer_kwargs)

        self.cb_fh.close()
        self.temp_params[self.indeces] = result.x
        result.x = self.temp_params
        return result
    
    def baseline_objective(self, p):
        pass
    
    def minima_callback(self, x, f, accept):
        self.temp_params[self.indeces] = x
        return self.optimizer.minima_callback(self.temp_params, f, accept)

def test_optimizer(caller):
    print("test_optimizer")
    optimizer = caller.get_optimizer()
    init_params = caller.get_init_params()
    optimizer.objective_func(init_params, plot=True)