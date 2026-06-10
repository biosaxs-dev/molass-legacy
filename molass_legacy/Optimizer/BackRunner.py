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

        # Phase 2 — Library-quality D and rg_curve for LEG-GUI path (molass-library#211).
        #
        # The root cause of the ~2 SV gap between LIB-IN and LEG-GUI is that the legacy
        # data loader retains 8 noisy low-q rows near the beamstop (q<0.0143 Å⁻¹) that
        # the library's load_xr_with_options strips during trimming.  Those rows cause
        # SimpleGuinier to fail on ~50% of frames → legacy rg_curve has only 121 valid
        # frames vs library's 239.
        #
        # Fix: load the raw data from in_folder, apply the library pipeline
        # (trimmed_copy → corrected_copy), then:
        #   1. Overwrite ip_xr_D.npy with the library-corrected, library-trimmed D
        #      (consistent replacement: both D and rg_curve come from the same pipeline)
        #   2. Compute rg_curve from that same library D
        #   3. Export to rg_curve_parent/
        #
        # The D+rg_curve pair is now LIB+LIB (consistent).
        # Total overhead: ~35 s (data load 0.6 s + trim 0.1 s + correct 0.5 s + rg_curve 30 s).
        # Wrapped in try/except: failure is non-fatal; subprocess falls back to legacy data.
        try:
            _in_folder = get_setting('in_folder')
            if _in_folder and os.path.exists(_in_folder):
                from molass.DataObjects.SecSaxsData import SecSaxsData as _SSD
                from molass.Global.Options import set_molass_options as _sqo
                _sqo(quiet=True)
                self.logger.info("BackRunner: building library SSD from in_folder=%s", _in_folder)
                _ssd_raw      = _SSD(_in_folder, xr_only=True)
                _ssd_trimmed  = _ssd_raw.trimmed_copy()
                _ssd_corrected = _ssd_trimmed.corrected_copy()
                _D_lib  = _ssd_corrected.xr.M
                _qv_lib = _ssd_corrected.xr.q_values
                _E_lib  = _ssd_corrected.xr.E
                _jv_lib = _np.array(_ssd_corrected.xr.jv, dtype=int)

                # Overwrite ip_xr_D.npy / ip_xr_E.npy / ip_xr_qvector.npy / ip_xr_jv.npy
                # with library-quality versions so subprocess sees consistent data.
                # IMPORTANT: also overwrite ip_xr_elcurve_y.npy — the elcurve y must have
                # the same frame count as D (241 frames here vs 242 for legacy).
                # Mismatch causes ValueError in BasicOptimizer.compute_LRF_matrices:
                # shapes (4,242) vs (1,241).
                _xr_icurve_y = _ssd_corrected.xr.get_icurve().y   # (n_frames,) matching D
                _np.save(os.path.join(_opt_folder, 'ip_xr_D.npy'),          _D_lib)
                _np.save(os.path.join(_opt_folder, 'ip_xr_E.npy'),          _E_lib)
                _np.save(os.path.join(_opt_folder, 'ip_xr_qvector.npy'),    _qv_lib)
                _np.save(os.path.join(_opt_folder, 'ip_xr_jv.npy'),         _jv_lib)
                _np.save(os.path.join(_opt_folder, 'ip_xr_elcurve_y.npy'),  _xr_icurve_y)
                self.logger.info("BackRunner: library D (%s), qvector (%s), elcurve (%s) written",
                                 _D_lib.shape, _qv_lib.shape, _xr_icurve_y.shape)
            else:
                # No in_folder available — fall through to legacy rg_curve path below
                _D_lib = _qv_lib = _E_lib = _jv_lib = None
                self.logger.warning("BackRunner: in_folder not found; using legacy D for rg_curve")
        except Exception as _e3:
            _D_lib = _qv_lib = _E_lib = _jv_lib = None
            self.logger.warning("BackRunner: library D computation failed (%s); using legacy D", _e3)

        # Compute rg_curve from library D (if Phase 2 succeeded) or legacy D (fallback),
        # then export to rg_curve_parent/ so subprocess reads library-quality Guinier data.
        try:
            from molass.Guinier.RgCurveUtils import compute_rg_curve_from_arrays
            from molass.Bridge.LegacyRgCurve import LegacyRgCurve
            import shutil as _shutil

            if _D_lib is not None:
                # Phase 2 succeeded: use library D (LIB+LIB consistent pair)
                _D_rg, _qv_rg, _E_rg, _jv_rg = _D_lib, _qv_lib, _E_lib, _jv_lib
                self.logger.info("BackRunner: using library D for rg_curve (Phase 2)")
            else:
                # Fallback: use legacy D (GUI+GUI consistent pair)
                _D_rg   = optimizer.xrD
                _qv_rg  = optimizer.qvector
                _E_rg   = optimizer.xrE
                _jv_rg  = _np.array(optimizer.xr_curve.x, dtype=int)
                self.logger.info("BackRunner: using legacy D for rg_curve (fallback)")

            _xr_curve = optimizer.xr_curve
            _lib_rgcurve = compute_rg_curve_from_arrays(_D_rg, _qv_rg, _E_rg, jv=_jv_rg)
            _legacy_rgcurve = LegacyRgCurve(_xr_curve, _lib_rgcurve)

            _parent_rg_folder = os.path.join(_opt_folder, 'rg_curve_parent')
            if os.path.exists(_parent_rg_folder):
                _shutil.rmtree(_parent_rg_folder)
            os.makedirs(_parent_rg_folder)
            _legacy_rgcurve.export(_parent_rg_folder)

            from molass_legacy._MOLASS.SerialSettings import set_setting as _ss
            _ss('trust_rg_curve_folder', True)
            self.logger.info("BackRunner: library rg_curve exported to %s", _parent_rg_folder)
        except Exception as _e2:
            self.logger.warning("BackRunner: library rg_curve export failed (%s); "
                                "subprocess will use legacy rg-curve/", _e2)

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
