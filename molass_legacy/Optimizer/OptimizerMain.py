"""
    Optimizer.OptimizerMain.py

    Copyright (c) 2021-2025, SAXS Team, KEK-PF
"""
import sys
import os
import getopt
import numpy as np
from molass_legacy.Baseline.BaselineUtils import create_xr_baseline_object
from .FuncImporter import import_objective_function

def main_driver():
    optlist, args = getopt.getopt(sys.argv[1:], 'c:w:f:n:i:b:d:m:s:r:t:p:T:M:S:L:X:O:')
    print(optlist, args)
    optdict = dict(optlist)
    if optdict.get('-r'):
        main_impl(optdict, optlist)
    else:
        main_tk(optdict, optlist)

def main_tk(optdict, optlist):
    from molass_legacy.KekLib.TkUtils import get_tk_root
    root = get_tk_root()
    def run_main():
        main_impl(optdict, optlist)
        root.quit()
    root.after(0, run_main)
    root.mainloop()
    root.destroy()

def main_impl(optdict, optlist):
    import sys
    import os
    import numpy as np
    from molass_legacy.KekLib.ChangeableLogger import Logger
    from molass_legacy import get_version
    from molass_legacy._MOLASS.SerialSettings import set_setting
    from molass_legacy.Optimizer.OptimizerSettings import OptimizerSettings
    from molass_legacy.Optimizer.OptimizerMain import optimizer_main
    from molass_legacy.Optimizer.TheUtils import get_analysis_folder_from_work_folder
    from molass_legacy.Optimizer.SettingsSerializer import unserialize_for_optimizer

    work_folder = optdict['-w']
    os.chdir(work_folder)
    work_folder = os.getcwd()   # to get absolute path

    nnn = int(work_folder[-3:])

    log_file = "optimizer.log"
    logger = Logger(log_file)

    try:
        logger.info(get_version(toml_only=True) + " (development version)")
    except Exception as e:
        logger.info(get_version())
    logger.info("Optimizer started in %s", work_folder)
    python_syspath = os.environ.get('MOLASS_PYTHONPATH')
    logger.info("MOLASS_PYTHONPATH=%s", str(python_syspath))

    analysis_folder = get_analysis_folder_from_work_folder(work_folder)
    logger.info("xxx sys.argv[1:]: %s", sys.argv[1:])
    logger.info("work_folder: %s", work_folder)
    logger.info("analysis_folder inferred as %s", analysis_folder)
    set_setting("analysis_folder", analysis_folder)
    optimizer_folder = os.path.join(analysis_folder, "optimized")
    set_setting("optimizer_folder", optimizer_folder)   # optimizer_folder will be referenced in DataTreatment.load()

    in_folder = optdict['-f']
    set_setting("in_folder", in_folder)     # required in the devel mode of SecTheory.ColumnTypes.py

    settings = OptimizerSettings()
    settings.load(optimizer_folder=optimizer_folder)    # this should restore required temporary settings
    logger.info("optimizer settings restored as %s", str(settings))

    try:
        class_code = optdict['-c']
        n_components = int(optdict['-n'])
        init_params_txt = optdict['-i']
        init_params = np.loadtxt(init_params_txt)
        bounds_txt = optdict['-b']
        if os.path.exists(bounds_txt):
            real_bounds = np.loadtxt(bounds_txt)
        else:
            real_bounds = None
        drift_type = optdict['-d']
        niter = int(optdict['-m'])
        seed = int(optdict['-s'])
        trimming_txt = optdict['-r']
        sleep_seconds = optdict.get('-t')
        legacy = optdict.get('-L') == 'legacy'

        unserialize_for_optimizer(optdict.get('-p'))    # "poresize_bounds", "t0_upper_bound"

        test_pattern = optdict.get('-T')
        if test_pattern != "None":
            set_setting("test_pattern", test_pattern)

        with open("in_data_info.txt", "w") as fh:
            fh.write("in_folder=%s\n" % in_folder)

        callback_txt = "callback.txt"
        with open(callback_txt, "w") as fh:
            pass

        with open("pid.txt", "w") as fh:
            fh.write("pid=%d\n" % os.getpid())

        with open("seed.txt", "w") as fh:
            fh.write("seed=%d\n" % seed)

        shm_name = optdict.get('-M')
        if shm_name is None or shm_name  == "None":
            shared_memory = None
        else:
            from molass_legacy.Optimizer.NpSharedMemory import get_shm_proxy
            shared_memory = get_shm_proxy(shm_name)

        solver = optdict.get('-S')
        xr_only = optdict.get('-X') == '1'
        MOLASS_OPTIMIZER_TEST = os.environ.get("MOLASS_OPTIMIZER_TEST")
        logger.info("MOLASS_OPTIMIZER_TEST=%s", str(MOLASS_OPTIMIZER_TEST))
        optimizer_test = (optdict.get('-O') == '1' or MOLASS_OPTIMIZER_TEST == '1')   # workaround for -O not passed issue

        if sleep_seconds is None:
            logger.info("optimizer started with class_code=%s, optlist=%s, shared_memory=%s, xr_only=%s, optimizer_test=%s",
                        class_code, str(optlist), shm_name, xr_only, optimizer_test)
            optimizer_main(in_folder,
                    trimming_txt=trimming_txt,
                    n_components=n_components,
                    solver=solver,
                    drift_type=drift_type,
                    init_params=init_params,
                    real_bounds=real_bounds,
                    niter=niter,
                    seed=seed,
                    class_code=class_code,
                    shared_memory=shared_memory,
                    nnn=nnn,
                    legacy=legacy,
                    xr_only=xr_only,
                    optimizer_test=optimizer_test,
                    debug=False,
                    )

        else:
            from time import sleep
            logger.info("dummy started with niter=%d, seed=%s", niter, str(seed))
            for k in range(int(sleep_seconds)):
                if k % 3 == 0:
                    with open(callback_txt, "a") as fh:
                        fh.write(str([k])+"\n")
                sleep(1)
        if shared_memory is not None:
            shared_memory.close()
            logger.info("shared_memory closed")
    except:
        from molass_legacy.KekLib.ExceptionTracebacker import log_exception
        log_exception(logger, "main_impl failed: ", n=10)
        exit(-1)

def create_optimizer_from_job(in_folder=None, n_components=None, class_code=None, trimming_txt=None, shared_memory=None,
                              need_settings=False):
    if need_settings:
        pass

    from molass.Bridge.OptimizerInput import OptimizerInput
    fullopt_input = OptimizerInput(in_folder=in_folder, trimming_txt=trimming_txt, legacy=True)
    dsets = fullopt_input.get_dsets()
    x_shifts_file = trimming_txt.replace('trimming.txt', 'x_shifts.txt')
    if os.path.exists(x_shifts_file):
        x_shifts = np.loadtxt(x_shifts_file, dtype=int)
        dsets.apply_x_shifts(x_shifts)
    fullopt_class = import_objective_function(class_code)
    uv_base_curve = fullopt_input.get_base_curve()      # uv_base_curve comes from FullOptInput.get_sd_from_folder()
    xr_base_curve = create_xr_baseline_object()
    qvector, wvector = fullopt_input.get_spectral_vectors()
    optimizer = fullopt_class(dsets, n_components,
                uv_base_curve=uv_base_curve,
                xr_base_curve=xr_base_curve,
                qvector=qvector,
                wvector=wvector,
                shared_memory=shared_memory)
    return optimizer

def optimizer_main(in_folder, trimming_txt=None, n_components=3,
                   solver=None,
                   init_params=None, real_bounds=None,
                   drift_type=None, niter=100, seed=None,
                   callback=True, class_code='F0000', shared_memory=None,
                   nnn=0,
                   legacy=True,
                   xr_only=False,
                   optimizer_test=False,
                   debug=True):

    optimizer = create_optimizer_from_job(in_folder=in_folder,
                                          n_components=n_components,
                                          class_code=class_code,
                                          trimming_txt=trimming_txt,
                                          shared_memory=shared_memory)
    optimizer.set_xr_only(xr_only)

    if optimizer_test:
        from molass_legacy.Optimizer.Compatibility import test_optimizer_compatibility
        test_optimizer_compatibility(optimizer, init_params)
        return

    if seed is None:
        seed = np.random.randint(100000, 999999)
    strategy = optimizer.get_strategy()
    if strategy.trust_initial_baseline():
        baseline_fixed = True
        from molass_legacy.Optimizer.FixedBaselineOptimizer import FixedBaselineOptimizer
        fb_optimizer = FixedBaselineOptimizer(optimizer)
        result = fb_optimizer.solve(init_params, real_bounds=real_bounds, niter=niter, seed=seed, debug=debug)
    else:
        baseline_fixed = False
        if strategy.baseline_first():
            if nnn == 0:
                from molass_legacy.Optimizer.BaselineOptimizer import BaselineOptimizer
                baseline_optimizer = BaselineOptimizer(optimizer)
                baseline_indeces = optimizer.get_baseline_indeces()
                result = baseline_optimizer.solve(init_params, baseline_indeces)
            else:
                baseline_fixed = True
                result = optimizer.solve(init_params, real_bounds=real_bounds, niter=niter, seed=seed, callback=callback, method=solver,
                                        baseline_fixed=baseline_fixed, debug=debug)
        else:
            if strategy.is_strategic(nnn):
                from molass_legacy.Optimizer.StrategicOptimizer import StrategicOptimizer
                temp_params = init_params
                indeces_list = strategy.get_indeces_list(nnn)
                assert len(indeces_list) == 1   # for now
                open_mode = "w"                 # "callback.txt" open mode
                for indeces in indeces_list:
                    strategic_optimizer = StrategicOptimizer(optimizer, indeces)
                    # task: add method option
                    result = strategic_optimizer.solve(temp_params, real_bounds=real_bounds, niter=niter, seed=seed, open_mode=open_mode, debug=debug)
                    temp_params = result.x
                    open_mode = "a"
            else:
                result = optimizer.solve(init_params, real_bounds=real_bounds, niter=niter, seed=seed, callback=callback, method=solver, debug=debug)
    fig_info = [in_folder, None, result]
    if debug:
        optimizer.objective_func(result.x, plot=True, fig_info=fig_info)
