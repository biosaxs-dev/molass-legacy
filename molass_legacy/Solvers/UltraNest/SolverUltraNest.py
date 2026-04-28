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
from molass_legacy.Solvers.UltraNest.SamplerCallback import _running_in_jupyter_kernel

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

    def _get_mapping_norm_indeces(self):
        """Return the norm-space indices of mapping params (mp_a, mp_b).

        Mapping params (UV↔XR frame scale and offset) may have a wrong initial
        estimate from the LRF decomposition.  Using a narrow NS prior centred on
        a wrong estimate locks NS out of the true solution.  Callers use this
        method to identify which normalised-param indices should receive wide
        priors regardless of NARROW_BIND_ALLOW.

        Returns None if the optimizer has no params_type with a pos attribute
        (e.g. non-SDM models that don't have a mapping group).
        """
        opt = self.optimizer
        params_type = getattr(opt, 'params_type', None)
        if params_type is None or not hasattr(params_type, 'pos') or len(params_type.pos) < 4:
            return None
        real_start = params_type.pos[3]      # pos[3] = start of mapping group
        real_indeces = np.array([real_start, real_start + 1])
        if opt.xr_params_indeces is None:
            return real_indeces
        # When only a subset of params is optimised, find where the mapping
        # real indices appear in the xr_params_indeces subset array.
        norm_indeces = []
        for ri in real_indeces:
            positions = np.where(opt.xr_params_indeces == ri)[0]
            if len(positions) > 0:
                norm_indeces.append(int(positions[0]))
        return np.array(norm_indeces) if norm_indeces else None

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
            # Mapping params (mp_a, mp_b) may have a wrong initial estimate from
            # LRF decomposition.  Override narrow prior with wide bounds so NS can
            # reach the correct value even when the initial estimate is far off.
            # (molass-legacy #32)
            if bounds is not None:
                mapping_idx = self._get_mapping_norm_indeces()
                if mapping_idx is not None and len(mapping_idx) > 0:
                    lower[mapping_idx] = bounds[mapping_idx, 0]
                    upper[mapping_idx] = bounds[mapping_idx, 1]
                    self.logger.info("NS wide prior applied to mapping params at norm indices %s", mapping_idx)
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

        _show_status = not _running_in_jupyter_kernel()
        self.logger.info("running without any step sampler")
        result1 = sampler.run(min_num_live_points=400, max_ncalls=10000, viz_callback=sampler_callback, show_status=_show_status)

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
        result2 = sampler.run(min_num_live_points=400, max_ncalls=max_ncalls, viz_callback=sampler_callback, show_status=_show_status)

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