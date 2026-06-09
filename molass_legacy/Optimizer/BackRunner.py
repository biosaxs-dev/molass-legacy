"""
    Optimizer.BackRunner.py

    Copyright (c) 2021-2026, SAXS Team, KEK-PF
"""
import os
import sys
import logging
import numpy as np
import subprocess
from molass_legacy._MOLASS.SerialSettings import get_setting, set_setting
from molass_legacy.KekLib.BasicUtils import mkdirs_with_retry, is_empty_dir
from molass_legacy.Trimming import save_trimming_txt
from .TheUtils import get_optjob_folder_impl
from .SettingsSerializer import serialize_for_optimizer

MAX_NUM_JOBS = 1000

class BackRunner:
    def __init__(self, xr_only=False, shared_memory=True):
        self.logger = logging.getLogger(__name__)
        self.optjob_folder = get_optjob_folder_impl()
        if shared_memory:
            from .NpSharedMemory import get_shm_singleton
            self.np_shm = get_shm_singleton()
        else:
            self.np_shm = None
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
            optimizer_test=False, debug=False, devel=False):
        from .FullOptResult import FILES

        self.logger.info("Running optimizer: %s with optimizer_test=%s", optimizer.__class__.__name__, optimizer_test)

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
        opt_O = '1' if optimizer_test else '0'
        os.environ["MOLASS_OPTIMIZER_TEST"] = opt_O
        if debug:
            print("BackRunner: work_folder =", folder, "optimizer_test =", optimizer_test, "opt_O =", opt_O, "shared memory =", self.np_shm)
        if optimizer_test:
            pass
        else:
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

        # Save in_folder so DsetsDebug.reconstruct_subprocess_dsets() can
        # reconstruct the subprocess datasets without live SerialSettings.
        with open(os.path.join(folder, 'in_folder.txt'), 'w') as _fh:
            _fh.write(in_folder)

        trimming_txt = FILES[6]
        trimming_file = os.path.join(folder, trimming_txt)
        save_trimming_txt(trimming_file)

        if x_shifts is not None:
            x_shifts_txt = FILES[8]
            x_shifts_file = os.path.join(folder, x_shifts_txt)
            np.savetxt(x_shifts_file, x_shifts, fmt="%d")

        # Save frozen_components if set on the optimizer
        if hasattr(optimizer, 'frozen_components') and optimizer.frozen_components is not None:
            frozen_file = os.path.join(folder, 'frozen_components.txt')
            np.savetxt(frozen_file, optimizer.frozen_components, fmt="%d")

        # Save frozen_param_groups if set on the optimizer
        if getattr(optimizer, 'frozen_param_groups', None) is not None:
            fpg_file = os.path.join(folder, 'frozen_param_groups.txt')
            with open(fpg_file, 'w') as _fh:
                _fh.write('\n'.join(optimizer.frozen_param_groups) + '\n')

        # test_pattern = str(get_setting("test_pattern"))
        test_pattern = "0"      # always set it to "0" to suppress execution-blocking messages

        serialized_str = serialize_for_optimizer()  # "poresize_bounds", "t0_upper_bound"

        np_shm_name = "None" if self.np_shm is None else self.np_shm.name

        from .OptimizerUtils import get_impl_method_name
        nnn = int(self.working_folder[-3:])
        self.solver = get_impl_method_name(nnn)

        stderr_path = os.path.join(folder, 'optimizer_stderr.txt')
        self._stderr_file = open(stderr_path, 'w')

        # Export ip_*.npy override files so the subprocess reads the same
        # in-process-prepared data as the parent (molass-library#206).
        # These are the same 6 files that prepare_rigorous_folders() writes when
        # called via the notebook path (make_rigorous_decomposition_impl).
        # Without them, the subprocess re-derives all data from disk via the legacy
        # loader (get_sd_from_folder_impl), reproducing the pre-#38 divergence
        # (~5-6 SV gap for all solvers from the GUI).
        try:
            _opt_folder = os.path.dirname(self.optjob_folder)  # analysis_folder/optimized (parent of jobs/)
            _np = __import__('numpy')
            _np.save(os.path.join(_opt_folder, 'ip_xr_elcurve_y.npy'),  optimizer.xr_curve.y)
            _np.save(os.path.join(_opt_folder, 'ip_uv_elcurve_y.npy'),  optimizer.uv_curve.y)
            _np.save(os.path.join(_opt_folder, 'ip_xr_D.npy'),          optimizer.xrD)
            _np.save(os.path.join(_opt_folder, 'ip_uv_U.npy'),          optimizer.uvD)
            _np.save(os.path.join(_opt_folder, 'ip_xr_E.npy'),          optimizer.xrE)
            _np.save(os.path.join(_opt_folder, 'ip_xr_qvector.npy'),    optimizer.qvector)
            self.logger.info("BackRunner: exported ip_*.npy override files to %s", _opt_folder)
        except Exception as _e:
            self.logger.warning("BackRunner: ip_*.npy export failed (%s); subprocess will use legacy-derived data", _e)

        # Pass MOLASS_NS_SUBPROCESS so SamplerCallback skips CustomLivePointsWidget,
        # which spawns a tkinter GUI subprocess and blocks on Queue.put() when that
        # subprocess crashes (molass-legacy#67).
        _subprocess_env = os.environ.copy()
        _subprocess_env['MOLASS_NS_SUBPROCESS'] = '1'
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
                '-X', '1' if self.xr_only else '0',
                '-O', opt_O,
                ], stderr=self._stderr_file, env=_subprocess_env)

    def poll(self):
        return self.process.poll()

    def getpid(self):
        return self.process.pid

    def get_callback_txt_path(self):
        return os.path.join(self.working_folder, 'callback.txt')

    def terminate(self):
        self.process.terminate()
        if hasattr(self, '_stderr_file') and self._stderr_file:
            self._stderr_file.close()
            self._stderr_file = None

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
