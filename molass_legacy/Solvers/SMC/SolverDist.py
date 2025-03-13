"""
    SMC.SolverDist.py

    Copyright (c) 2024, SAXS Team, KEK-PF    
"""
import numpy as np
import pymc as pm

def create_custom_dist(objective, lower, upper, arg_size, params):
    from Solvers.SMC.SolverLogLike import SolverLogLike

    loglike_op = SolverLogLike(objective)
    rng_ = np.random.default_rng()

    def custom_loglike(data, params):
        # data, or observed is always passed as the first input of CustomDist
        return loglike_op(params)

    def custom_random(params, rng=None, size=None):
        """
        from the original CustomDist class documentation
        https://github.com/pymc-devs/pymc/blob/main/pymc/distributions/custom.py

        see also
        Custom Distribution with pm.CustomDist
        https://discourse.pymc.io/t/custom-distribution-with-pm-customdist/11071/2

        Pickle a model containing a custom distribution
        https://discourse.pymc.io/t/pickle-a-model-containing-a-custom-distribution/9668

        """
        return rng_.uniform(lower, upper, size=arg_size)

    dummy = None
    likelihood = pm.CustomDist(
        "likelihood", params,
        observed=dummy,
        logp=custom_loglike,
        random=custom_random,
    )

    return likelihood
