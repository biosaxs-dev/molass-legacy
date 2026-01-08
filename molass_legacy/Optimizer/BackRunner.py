"""
    Optimizer.BackRunner.py

    Copyright (c) 2021-2026, SAXS Team, KEK-PF
"""
import os
import sys
import numpy as np
import subprocess
from molass_legacy._MOLASS.SerialSettings import get_setting, set_setting
from molass_legacy.KekLib.BasicUtils import mkdirs_with_retry, is_empty_dir
from molass_legacy.Trimming import save_trimming_txt
from .TheUtils import get_optjob_folder_impl
from .NpSharedMemory import get_shm_singleton
from .SettingsSerializer import serialize_for_optimizer

MAX_NUM_JOBS = 1000

class BackRunner:
    def __init__(self, xr_only=False):
        self.optjob_folder = get_optjob_folder_impl()
        self.np_shm = get_shm_singleton()
        self.process = None
        self.solver = None
        self.xr_only = xr_only

    def get_optjob_folder(self):
        return self.optjob_folder

    def get_work_folder(self):
        optjob_folder = self.optjob_folder
        if not os.path.exists(optjob_folder):
            mkdirs_with_retry(optjob_folder)

        ok = False
        for k in range(MAX_NUM_JOBS):
            work_folder = os.path.join(optjob_folder, '%03d'%k)
            if os.path.exists(work_folder):
                if is_empty_dir(work_folder):
                    ok = True
                    break
            else:
                mkdirs_with_retry(work_folder)
                ok = True
                break

        assert ok

        return work_folder

    def set_work_folder(self, folder):
        # this method is currently used by tester
        self.working_folder = folder

    def run(self, optimizer, init_params, niter=100, seed=1234, work_folder=None, dummy=False, x_shifts=None, legacy=True,
            debug=False, devel=False):
        from .FullOptResult import FILES

        n_components = optimizer.n_components
        class_code = optimizer.__class__.__name__
        composite = optimizer.composite

        # note that, for next trials, which is distiguished from the first trial,
        # init_params is an optimized set of params different from optimizer.init_params

        if work_folder is None:
            folder = self.get_work_folder()
        else:
            # i.e., caller has prepared the folder (may be by get_work_folder())
            folder = work_folder
        if debug:
            print("BackRunner: work_folder =", folder)
        self.working_folder = folder
        set_setting("optjob_folder", folder)
        set_setting("optworking_folder", folder)    # unifiy these setting items
        init_params_txt = FILES[2]
        init_params_file = os.path.join(folder, init_params_txt)
        np.savetxt(init_params_file, init_params)

        bounds_txt = FILES[7]
        if optimizer.exports_bounds:
            np.savetxt(os.path.join(folder, bounds_txt), optimizer.real_bounds)

        this_folder = os.path.dirname(os.path.abspath( __file__ ))
        optimizer_py = os.path.join(this_folder, 'optimizer-dummy.py' if dummy else 'optimizer.py')

        in_folder = get_setting('in_folder')
        if in_folder is None:
            in_folder = 'IN_FOLDER_NOT_SET'

        trimming_txt = FILES[6]
        trimming_file = os.path.join(folder, trimming_txt)
        save_trimming_txt(trimming_file)

        if x_shifts is not None:
            x_shifts_txt = FILES[8]
            x_shifts_file = os.path.join(folder, x_shifts_txt)
            np.savetxt(x_shifts_file, x_shifts, fmt="%d")

        # test_pattern = str(get_setting("test_pattern"))
        test_pattern = "0"      # always set it to "0" to suppress execution-blocking messages

        serialized_str = serialize_for_optimizer()  # "poresize_bounds", "t0_upper_bound"

        np_shm_name = "None" if self.np_shm is None else self.np_shm.name

        from .OptimizerUtils import get_impl_method_name
        nnn = int(self.working_folder[-3:])
        self.solver = get_impl_method_name(nnn)
        python_syspath = os.getenv('PYTHONPATH', '') if devel else ''

        self.process = subprocess.Popen([sys.executable, optimizer_py,
                '-c', class_code,
                '-w', folder,
                '-f', in_folder,
                '-n', str(n_components),
                '-i', init_params_txt,
                '-b', bounds_txt,
                '-d', 'linear',
                '-m', str(niter),
                '-s', str(seed),
                '-r', trimming_txt,
                '-p', serialized_str,
                # '-t', '10',
                '-T', test_pattern,
                '-M', np_shm_name,
                '-S', self.solver,
                '-L', 'legacy' if legacy else 'library',
                '-P', python_syspath,
                '-X', '1' if self.xr_only else '0',
                ])

    def poll(self):
        return self.process.poll()

    def getpid(self):
        return self.process.pid

    def get_callback_txt_path(self):
        return os.path.join(self.working_folder, 'callback.txt')

    def terminate(self):
        self.process.terminate()

    def revive(self):
        # still working on this method as of 20210616
        nodes = os.listdir(self.optjob_folder)
        for i in range(1, 3):
            last_folder = os.path.join(self.optjob_folder, nodes[-i])
            try:
                self.process = PopenProxy(nodes[-1])
                break
            except:
                pass

class PopenProxy:
    def __init__(self, folder):
        # still working on this method as of 20210616
        pid_txt = os.path.join(folder, "pid.txt")
        # time check
        # pid
        # active
        callback_txt = os.path.join(folder, "callback.txt")
        # time check
