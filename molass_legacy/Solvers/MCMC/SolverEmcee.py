"""
    MCMC.SolverEmcee.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import logging
import numpy as np
import emcee
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.Optimizer.StateSequence import save_opt_params
from molass_legacy.Optimizer.OptimizerUtils import OptimizerResult

def walkers_independent(coords):
    if not np.all(np.isfinite(coords)):
        return False
    C = coords - np.mean(coords, axis=0)[None, :]
    C_colmax = np.amax(np.abs(C), axis=0)
    if np.any(C_colmax == 0):
        return False
    C /= C_colmax
    C_colsum = np.sqrt(np.sum(C**2, axis=0))
    C /= C_colsum
    return np.linalg.cond(C.astype(float)) <= 1e8

class SolverEmcee:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.cb_fh = optimizer.cb_fh
        self.nfev = 0
        self.logger = logging.getLogger(__name__)

    def log_likelihood(self, theta):
        self.nfev += 1
        return -self.objective(theta)

    def log_prior(self, theta):
        if np.min(theta - self.bounds[:,0]) >= 0 and np.max(theta - self.bounds[:,1]) <= 0:
            return 0.0
        return -np.inf

    def log_probability(self, theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta)

    def minimize(self, objective, init_params, niter=100, seed=1234, bounds=None, callback=None, debug=True):
        self.objective = objective
        self.bounds = bounds
        if callback is None:
            callback = self.callback
        ndim = len(init_params)
        nwalkers = ndim*4
        temp_params = init_params.copy()
        is_nan = np.isnan(temp_params)
        temp_params[is_nan] = np.mean(bounds[is_nan,:], axis=1)

        if debug:
            self.optimizer.debug_plot_params(temp_params)

        callback(temp_params, 0, False)
        p0 = [temp_params + 1e-7*np.random.randn(ndim) for i in range(nwalkers)]
        ret = walkers_independent(p0)
        print("ret=", ret)
        self.sampler = sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability, args=())

        self.logger.info("Running burn-in...")
        p0, _, _ = sampler.run_mcmc(p0, 100)

        self.logger.info("Running production...")
        for sample in sampler.sample(p0, iterations=200):
            # Only check convergence every 100 steps
            if sampler.iteration % 10:
                continue

            samples = sampler.flatchain
            theta_max = samples[np.argmax(sampler.flatlnprobability)]
            theta_max = sampler.flatchain[np.argmax(sampler.flatlnprobability)]
            callback(theta_max, 0, True)

        return OptimizerResult(x=theta_max, nit=niter, nfev=self.nfev)
    
    def callback(self, norm_params, f, accept):
        fv = self.objective(norm_params)
        self.logger.info("callback: fv=%.3g", fv)
        real_params = self.optimizer.to_real_params(norm_params)
        save_opt_params(self.cb_fh, real_params, fv, accept, self.nfev)
        return False