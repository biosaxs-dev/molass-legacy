"""
optimizer_recipe.py

Subprocess entry point for Option E (recipe-based subprocess).
Delegates all pipeline logic to molass.Rigorous.RecipeRunner so the
same code can be tested directly in a notebook.
"""


def create_optimizer_from_recipe(work_folder, n_components, class_code):
    """Thin wrapper — delegates to the testable library implementation."""
    from molass.Rigorous.RecipeRunner import create_optimizer_from_recipe as _impl
    return _impl(work_folder, class_code).optimizer


def main_recipe():
    """Entry point for subprocess with recipe mode."""
    import sys
    import os
    import getopt
    import numpy as np
    from molass_legacy.KekLib.ChangeableLogger import Logger
    from molass_legacy._MOLASS.SerialSettings import set_setting
    from molass_legacy.Optimizer.OptimizerSettings import OptimizerSettings
    from molass_legacy.Optimizer.TheUtils import get_analysis_folder_from_work_folder
    
    # Accept all flags BackRunner passes; unused ones are silently ignored.
    optlist, args = getopt.getopt(sys.argv[1:], 'c:w:f:n:i:b:d:m:s:r:p:T:M:S:L:X:O:')
    optdict = dict(optlist)
    
    work_folder = optdict['-w']
    os.chdir(work_folder)
    work_folder = os.getcwd()
    
    log_file = "optimizer.log"
    logger = Logger(log_file)
    logger.info("=== Recipe-based subprocess optimizer started ===")
    logger.info("Work folder: %s", work_folder)
    
    # Standard setup
    analysis_folder = get_analysis_folder_from_work_folder(work_folder)
    set_setting("analysis_folder", analysis_folder)
    optimizer_folder = os.path.join(analysis_folder, "optimized")
    set_setting("optimizer_folder", optimizer_folder)
    
    # Load settings
    settings = OptimizerSettings()
    settings.load(optimizer_folder=optimizer_folder)
    
    # Parse arguments
    class_code = optdict['-c']
    n_components = int(optdict['-n'])
    init_params_txt = optdict['-i']
    init_params = np.loadtxt(init_params_txt)
    bounds_txt = optdict['-b']
    real_bounds = np.loadtxt(bounds_txt) if os.path.exists(bounds_txt) else None
    niter = int(optdict['-m'])
    seed = int(optdict['-s'])
    solver = optdict.get('-S')
    
    # Create optimizer from recipe
    optimizer = create_optimizer_from_recipe(work_folder, n_components, class_code)

    # Re-prepare with the parent's init_params so both processes start identically.
    optimizer.prepare_for_optimization(init_params)

    with open("pid.txt", "w") as fh:
        fh.write("pid=%d\n" % os.getpid())
    with open("seed.txt", "w") as fh:
        fh.write("seed=%d\n" % seed)

    ns_narrow_bounds = settings.get('ns_narrow_bounds')
    ns_adaptive_nsteps = settings.get('ns_adaptive_nsteps')
    ns_nsteps = settings.get('ns_nsteps')

    logger.info("Starting optimization with solver=%s, niter=%d, seed=%d", solver or 'BH', niter, seed)
    result = optimizer.solve(
        init_params, real_bounds=real_bounds,
        niter=niter, seed=seed, callback=True,
        method=solver,
        ns_narrow_bounds=ns_narrow_bounds,
        ns_adaptive_nsteps=ns_adaptive_nsteps,
        ns_nsteps=ns_nsteps,
        debug=False,
    )
    logger.info("Optimization complete: best_fv=%.6f", result.fun)
    # scipy solvers (BH, DE, ...) expose why they stopped (converged vs. maxiter
    # reached) via .message/.nit; without this, diagnosing premature convergence
    # requires reverse-engineering it from callback.txt generation counts.
    _stop_message = getattr(result, 'message', None)
    _stop_nit = getattr(result, 'nit', None)
    if _stop_message is not None or _stop_nit is not None:
        logger.info("Solver stop reason: message=%s, nit=%s", _stop_message, _stop_nit)


if __name__ == '__main__':
    main_recipe()
