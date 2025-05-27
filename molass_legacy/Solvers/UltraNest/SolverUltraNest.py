"""
    Solvers.UltrNest.SolverUltraNest.py

    Copyright (c) 2024, SAXS Team, KEK-PF    
"""
import logging
import numpy as np
from ultranest import ReactiveNestedSampler 
from ultranest.stepsampler import SliceSampler, generate_mixture_random_direction
from molass_legacy.Optimizer.OptimizerUtils import OptimizerResult
from molass_legacy.Optimizer.StateSequence import save_opt_params

NARROW_BIND_ALLOW = 1.0

def get_max_ncalls(niter):
    return niter*7000

class SolverUltraNest:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.shm = optimizer.shm
        self.num_pure_components = optimizer.num_pure_components
        self.cb_fh = optimizer.cb_fh
        self.callback_counter = 0
        self.logger = logging.getLogger(__name__)

    def minimize(self, objective, init_params, niter=100, seed=1234, bounds=None, callback=None, narrow_bounds=True):
        from importlib import reload
        import Solvers.UltraNest.SamplerCallback
        reload(Solvers.UltraNest.SamplerCallback)
        from Solvers.UltraNest.SamplerCallback import SamplerCallback

        num_params = len(init_params)
        self.objective = objective

        if narrow_bounds:
            lower = init_params - NARROW_BIND_ALLOW
            upper = init_params + NARROW_BIND_ALLOW
        else:
            lower = bounds[:,0]
            upper = bounds[:,1]
        def my_prior_transform(cube):
            # transform location parameter: uniform prior
            params = cube * (upper - lower) + lower
            return params

        def my_likelihood(params):
            # print("objective_func_wrapper: par=", par)
            fv = objective(params)
            return -fv

        # logging.basicConfig(level=logging.INFO)     # to suppress debug log

        param_names = ["p%02d" % i for i in range(num_params)]
        sampler = ReactiveNestedSampler(param_names, my_likelihood, my_prior_transform)
        sampler.logger.setLevel(logging.INFO)       # to suppress debug log
        sampler_callback = SamplerCallback(self, sampler)

        self.logger.info("running without any step sampler")
        result1 = sampler.run(min_num_live_points=400, max_ncalls=10000, viz_callback=sampler_callback)

        self.logger.info("running with a step sampler: SliceSampler")
        # add a step sampler: from the "Higher-dimensional fitting" tutorial
        nsteps = 2 * num_params
        # create step sampler:
        sampler.stepsampler = SliceSampler(
            nsteps=nsteps,
            generate_direction=generate_mixture_random_direction,
            # adaptive_nsteps=False,
            # max_nsteps=400
        )

        max_ncalls = get_max_ncalls(niter)
        result2 = sampler.run(min_num_live_points=400, max_ncalls=max_ncalls, viz_callback=sampler_callback)

        opt_params = result2['maximum_likelihood']['point']

        return OptimizerResult(x=opt_params, nit=niter, nfev=self.optimizer.eval_counter)

    def callback(self, norm_params, f, accept):
        fv = self.objective(norm_params)
        self.logger.info("callback: fv=%.3g", fv)
        real_params = self.optimizer.to_real_params(norm_params)
        save_opt_params(self.cb_fh, real_params, fv, accept, self.optimizer.eval_counter)
        self.callback_counter += 1
        if self.shm is not None:
            self.shm.array[0] = self.callback_counter
        return False