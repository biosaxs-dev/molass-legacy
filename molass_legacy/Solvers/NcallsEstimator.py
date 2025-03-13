"""
    Solvers.NcallsEstimator.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
def bh_estimator(niter, ncallbacks, latest_ncalls):
    pass

def ultranest_estimator(niter, ncallbacks, latest_ncalls):
    pass

ESTIMATIOR_FUNC_DICT ={
    "bh"        : bh_estimator,
    "ultranest" : ultranest_estimator,
}

def estimate_ncalls(solver_name, niter, ncallbacks, latest_ncalls):
    func = ESTIMATIOR_FUNC_DICT[solver_name]
    return func(niter, ncallbacks, latest_ncalls)