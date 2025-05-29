"""
    optimizer.py

    Copyright (c) 2021-2025, SAXS Team, KEK-PF
"""

def main():
    import os
    import sys
    import getopt
    this_dir = os.path.dirname( os.path.abspath( __file__ ) )
    root_dir = os.path.dirname(os.path.dirname( this_dir ))
    sys.path.insert(0, root_dir)

    optlist, args = getopt.getopt(sys.argv[1:], 'c:w:f:n:i:b:d:m:s:r:t:p:T:M:S:')
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
    import os
    import numpy as np
    from molass_legacy.KekLib.ChangeableLogger import Logger
    from molass_legacy._MOLASS.Version import get_version_string
    from molass_legacy._MOLASS.SerialSettings import set_setting
    from molass_legacy.Optimizer.OptimizerSettings import OptimizerSettings
    from molass_legacy.Optimizer.OptimizerMain import optimizer_main
    from molass_legacy.Optimizer.TheUtils import get_analysis_folder_from_work_folder
    from molass_legacy.Optimizer.SettingsSerializer import unserialize_for_optimizer

    work_folder = optdict['-w']
    os.chdir(work_folder)

    nnn = int(work_folder[-3:])

    log_file = "optimizer.log"
    logger = Logger(log_file)

    logger.info(get_version_string())

    analysis_folder = get_analysis_folder_from_work_folder(work_folder)
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

        if sleep_seconds is None:
            logger.info("optimizer started with class_code=%s, optlist=%s, shared_memory=%s", class_code, str(optlist), shm_name)
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

if __name__ == '__main__':
    main()
