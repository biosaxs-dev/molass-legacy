"""
Optimizer.MplMonitor.py

migration of FullOptDialog to Jupyter Notebook
"""
import sys
import io
import warnings
import os
import logging
import shutil
import time
import threading
import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
from molass_legacy.KekLib.IpyLabelUtils import inject_label_color_css
from molass_legacy._MOLASS.SerialSettings import get_setting, set_setting
from molass_legacy.KekLib.IpyLabelUtils import inject_label_color_css, set_label_color

# Global registry of active monitor instances
_ACTIVE_MONITORS = {}


def _safe_range(arr):
    """Return [min, max, len] of an array-like, or None on failure."""
    try:
        a = np.asarray(arr)
        if a.size == 0:
            return None
        return [float(np.nanmin(a)), float(np.nanmax(a)), int(a.size)]
    except Exception:
        return None


def _summarize_axes(fig):
    """Extract per-axis line summaries from a matplotlib Figure for the JSON sidecar.

    Issue #22: lets AI assistants see what was actually plotted (xlim, ylim, line
    colors, ranges) without having to OCR the PNG.
    """
    out = []
    for ax in fig.get_axes():
        try:
            lines = []
            for ln in ax.get_lines():
                xd = ln.get_xdata()
                yd = ln.get_ydata()
                lines.append({
                    "label": str(ln.get_label()),
                    "color": str(ln.get_color()),
                    "linestyle": str(ln.get_linestyle()),
                    "marker": str(ln.get_marker()),
                    "n_points": int(np.size(xd)),
                    "x_range": _safe_range(xd),
                    "y_range": _safe_range(yd),
                })
            out.append({
                "title": ax.get_title(),
                "xlim": [float(v) for v in ax.get_xlim()],
                "ylim": [float(v) for v in ax.get_ylim()],
                "n_lines": len(lines),
                "lines": lines,
            })
        except Exception as e:
            out.append({"error": str(e)})
    return out


def _coord_provenance(opt, label):
    """Snapshot the data-coordinate domains of an optimizer's curves.

    Issue #22: makes 'data references diverged' bugs (e.g. #21) instantly visible
    without spelunking through the optimizer object.
    """
    info = {"label": label}
    if opt is None:
        info["present"] = False
        return info
    info["present"] = True
    info["class"] = opt.__class__.__name__
    for attr in ("xr_curve", "uv_curve"):
        cur = getattr(opt, attr, None)
        if cur is None:
            info[attr] = None
            continue
        d = {"x_range": _safe_range(getattr(cur, "x", None))}
        spline = getattr(cur, "spline", None)
        # UnivariateSpline-like: get_knots() gives the data domain
        if spline is not None and hasattr(spline, "get_knots"):
            try:
                k = spline.get_knots()
                d["spline_domain"] = [float(k[0]), float(k[-1])]
            except Exception:
                pass
        info[attr] = d
    rg = getattr(opt, "rg_curve", None)
    if rg is not None:
        seg_x_ranges = []
        for s in getattr(rg, "segments", []) or []:
            try:
                seg_x_ranges.append(_safe_range(s[0]))
            except Exception:
                pass
        info["rg_curve"] = {
            "x_range": _safe_range(getattr(rg, "x", None)),
            "segments_x_ranges": seg_x_ranges,
            "n_segments": len(seg_x_ranges),
        }
    return info


def _build_monitor_snapshot_json(monitor, display_optimizer, params):
    """Build the JSON sidecar payload for one MplMonitor.update_plot() call. (Issue #22)"""
    snap = {
        "schema_version": 1,
        "timestamp": datetime.now().isoformat(),
        "trial": int(getattr(monitor, "num_trials", 0)),
        "curr_index": getattr(monitor, "curr_index", None),
    }
    # Optimization state from display_optimizer
    try:
        snap["eval_counter"] = int(getattr(display_optimizer, "eval_counter", -1))
    except Exception:
        pass
    try:
        fv = float(display_optimizer.objective_func(params))
        snap["fv"] = fv
        from molass_legacy.Optimizer.FvScoreConverter import convert_score
        snap["sv"] = float(convert_score(fv))
    except Exception as e:
        snap["fv_error"] = str(e)
    # Issue #AI-A: record best fv/sv seen across ALL update_plot() calls so that
    # an AI tool reading this file gets the global minimum, not just the current
    # live-point snapshot (which may be worse than the historical best).
    try:
        from molass_legacy.Optimizer.FvScoreConverter import convert_score as _cs
        js = getattr(monitor, "job_state", None)
        if js is not None and hasattr(js, "fv") and len(js.fv) > 0:
            best_fv_hist = float(np.min(js.fv[:, 1]))
            snap["best_fv"] = best_fv_hist
            snap["best_sv"] = float(_cs(best_fv_hist))
    except Exception:
        pass
    # Param breakdown
    try:
        sp = display_optimizer.split_params_simple(params)
        xr_p, xr_bp, rg_p, ab, uv_p, uv_bp, cd, sec_p = sp[0:8]
        snap["params"] = {
            "xr_h": np.asarray(xr_p)[:, 0].tolist(),
            "xr_m": np.asarray(xr_p)[:, 1].tolist(),
            "xr_s": np.asarray(xr_p)[:, 2].tolist(),
            "xr_t": np.asarray(xr_p)[:, 3].tolist(),
            "rg":   np.asarray(rg_p).tolist(),
            "ab_mapping": [float(ab[0]), float(ab[1])],
            "uv_h":  np.asarray(uv_p).tolist(),
            "xr_baseparams": np.asarray(xr_bp).tolist(),
            "uv_baseparams": np.asarray(uv_bp).tolist(),
            "cd": [float(cd[0]), float(cd[1])],
            "seccol_params": np.asarray(sec_p).tolist(),
        }
    except Exception as e:
        snap["params_error"] = str(e)
    # Per-axis summary
    try:
        snap["axes"] = _summarize_axes(monitor.fig)
    except Exception as e:
        snap["axes_error"] = str(e)
    # Coordinate provenance — the diagnostic gold for #21-class bugs
    snap["data_provenance"] = {
        "display": _coord_provenance(display_optimizer, "display_optimizer"),
        "parent":  _coord_provenance(getattr(monitor, "optimizer", None), "self.optimizer"),
    }
    return snap


class _SubprocessSource:
    """Wraps BackRunner to implement the ProgressSource protocol.

    Concentrates the subprocess-specific surface so that MplMonitor can be
    parameterised with alternative sources (e.g. _RunInfoSource for in-process
    runs) without duplicating its widget/watcher code.  Added in Phase 1 of the
    MplMonitor pluggable-source refactor (molass-library#139).
    """

    def __init__(self, runner):
        self._runner = runner

    def is_alive(self):
        """True while the subprocess is still running."""
        return self._runner.poll() is None

    def terminate(self):
        """Terminate the subprocess."""
        self._runner.terminate()

    @property
    def working_folder(self):
        """Path to the subprocess working folder."""
        return self._runner.working_folder


class _RunInfoSource:
    """Wraps RunInfo to implement the ProgressSource protocol for in-process runs.

    Used by ``MplMonitor.for_run_info()`` so the same widget/watcher machinery
    serves both the subprocess and in-process paths.  Added in Phase 2 of the
    MplMonitor pluggable-source refactor (molass-library#139).
    """

    def __init__(self, run_info):
        self._ri = run_info

    def is_alive(self):
        """True while the in-process optimizer thread is running."""
        return self._ri.is_alive

    def terminate(self):
        """No-op: cannot cleanly kill a Python thread.

        The Terminate button is hidden when this source is used; this method
        exists only so the protocol is complete.  A cooperative-flag mechanism
        is tracked as a future enhancement.
        """
        pass

    @property
    def working_folder(self):
        """Path to the in-process optimizer working folder."""
        return self._ri.work_folder


class MplMonitor:
    """Interactive Jupyter notebook monitor for optimization processes with subprocess management.
    
    This class provides a dashboard-based interface for running and monitoring optimization jobs
    in Jupyter notebooks. It manages background subprocess execution, provides real-time progress
    visualization, and implements robust recovery mechanisms to prevent losing control of running
    processes when notebook outputs are cleared.
    
    The monitor tracks active processes through both an in-memory registry and a persistent file-based
    registry, allowing recovery from accidental notebook state loss.
    
    Parameters
    ----------
    function_code : str, optional
        Function code identifier for logging purposes.
    clear_jobs : bool, default=True
        If True, clears existing job folders in the optimizer directory on initialization.
    debug : bool, default=True
        If True, enables debug mode with module reloading for development.
    
    Attributes
    ----------
    optimizer_folder : str
        Path to the folder containing optimization outputs and logs.
    logger : logging.Logger
        Logger instance for recording monitor activities.
    runner : BackRunner
        Background process runner managing the subprocess execution.
    dashboard : ipywidgets.VBox
        The main dashboard widget containing plots and controls.
    process_id : str
        String representation of the current subprocess PID.
    instance_id : int
        Unique identifier for this monitor instance.
    
    Examples
    --------
    Basic usage with automatic recovery::
    
        from molass_legacy.Optimizer.MplMonitor import MplMonitor
        
        # Create and configure monitor
        monitor = MplMonitor(clear_jobs=True)
        monitor.create_dashboard()
        
        # Run optimization
        monitor.run(optimizer, init_params, niter=20, max_trials=30)
        monitor.show()
        monitor.start_watching()
    
    Recovering a lost dashboard after clearing notebook outputs::
    
        # Retrieve the most recent active monitor
        monitor = MplMonitor.get_active_monitor()
        monitor.redisplay_dashboard()
    
    Checking all active monitors::
    
        # Display status of all running monitors
        MplMonitor.show_active_monitors()
        
        # Get all active instances
        monitors = MplMonitor.get_all_active_monitors()
    
    Cleaning up orphaned processes::
    
        # Interactive cleanup of orphaned processes
        MplMonitor.cleanup_orphaned_processes()
    
    Notes
    -----
    - The monitor maintains two registries: an in-memory registry for quick access to active
      instances, and a file-based registry (``active_processes.json``) for subprocess tracking
      that persists across notebook sessions.
    
    - When creating a new monitor while others are active, a warning is displayed with
      instructions for recovery.
    
    - The dashboard includes real-time plot updates, status indicators, and control buttons
      for terminating jobs and exporting data.
    
    - Background processes are automatically cleaned up when the monitor detects they are
      orphaned or when the monitor instance is destroyed.
    
    - For optimal use in Jupyter notebooks, use ``start_watching()`` to run progress monitoring
      in a background thread, keeping the notebook interactive.
    
    .. note::
       Process registry and dashboard recovery features implemented with assistance from
       GitHub Copilot (January 2026).
    
    See Also
    --------
    BackRunner : Manages subprocess execution for optimization jobs.
    JobState : Tracks and parses optimization job state from callback files.
    
    """
    def __init__(self, source=None, function_code=None, clear_jobs=True, xr_only=False, debug=True):
        """Initialise MplMonitor.

        Parameters
        ----------
        source : _SubprocessSource or _RunInfoSource, optional
            ProgressSource that determines who is alive/terminate/working_folder.
            When ``None`` (default) a ``_SubprocessSource`` backed by a new
            ``BackRunner`` is constructed — existing subprocess behaviour,
            unchanged.
        """
        if source is None:
            if debug:
                from importlib import reload
                import molass_legacy.Optimizer.BackRunner
                reload(molass_legacy.Optimizer.BackRunner)
            from molass_legacy.Optimizer.BackRunner import BackRunner
            source = _SubprocessSource(BackRunner(xr_only=xr_only, shared_memory=False))
        self.source = source
        analysis_folder = get_setting("analysis_folder")
        optimizer_folder = os.path.join(analysis_folder, "optimized")
        self.optimizer_folder = optimizer_folder
        if clear_jobs:
            self.clear_jobs()
        logpath = os.path.join(optimizer_folder, 'monitor.log')
        self.fileh = logging.FileHandler(logpath, 'w')
        format_csv_ = '%(asctime)s,%(levelname)s,%(name)s,%(message)s'
        datefmt_ = '%Y-%m-%d %H:%M:%S'
        self.formatter_csv_ = logging.Formatter(format_csv_, datefmt_)
        self.fileh.setFormatter(self.formatter_csv_)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(self.fileh)
        # Phase 5c (mirror of InProcessRunner): register atexit to close
        # self.fileh before logging.shutdown() iterates _handlerList.
        # The watch_thread (daemon) holds self.fileh.lock during log calls;
        # logging.shutdown() trying to acquire that lock → deadlock on restart.
        # atexit runs LIFO so this fires before logging.shutdown().
        import atexit as _atexit_monitor
        def _cleanup_monitor_fileh(_fh=self.fileh, _log=self.logger):
            try:
                _log.removeHandler(_fh)
                _fh.close()
            except Exception:
                pass
        _atexit_monitor.register(_cleanup_monitor_fileh)
        del _cleanup_monitor_fileh, _atexit_monitor
        self.logger.info("MplMonitor initialized.")
        if isinstance(source, _SubprocessSource):
            self.logger.info(f"Optimizer job folder: {self.source._runner.optjob_folder}")
        self.result_list = []
        self.suptitle = None
        self.func_code = function_code
        self.process_id = None  # Will be set when process starts
        self.instance_id = id(self)
        self.watch_thread = None  # Will be set when watching starts
        self.dsets = None  # Set via run() caller or export_data(); see issue #7
        # Optional subprocess-equivalent optimizer for on-screen objective
        # re-evaluation, so MplMonitor SV matches callback.txt SV (issue #118).
        # If None, the plot path falls back to self.optimizer.
        self.monitor_optimizer = None
        self.stop_watch_event = threading.Event()  # For graceful thread shutdown
        self.is_monitoring = False  # Flag to track active monitoring state
        
        # Check for existing active monitors and warn user
        if _ACTIVE_MONITORS:
            self.logger.info(f"Found {len(_ACTIVE_MONITORS)} existing active monitor(s)")
            print(f"⚠ Warning: {len(_ACTIVE_MONITORS)} monitor(s) already active.")
            print("  Use MplMonitor.show_active_monitors() to see them.")
            print("  Use MplMonitor.get_active_monitor() to retrieve the last one.")
        
        # Register this instance in global registry
        _ACTIVE_MONITORS[self.instance_id] = self
        self.logger.info(f"Registered monitor instance {self.instance_id}")
        
        # Clean up any orphaned processes from previous sessions
        if isinstance(self.source, _SubprocessSource):
            self._cleanup_orphaned_processes()

    @classmethod
    def for_subprocess(cls, *, xr_only=False, function_code=None, clear_jobs=True, debug=True):
        """Create a MplMonitor backed by a new BackRunner subprocess.

        This is the canonical way to create a subprocess-mode monitor.  It is
        equivalent to the pre-Phase-2 ``MplMonitor()`` constructor and preserves
        all existing behaviour byte-for-byte.

        Parameters
        ----------
        xr_only : bool, default=False
            Pass ``True`` for X-ray-only (no UV) optimization.
        function_code, clear_jobs, debug
            Forwarded to ``__init__``.
        """
        return cls(source=None, function_code=function_code,
                   clear_jobs=clear_jobs, xr_only=xr_only, debug=debug)

    @classmethod
    def for_run_info(cls, run_info, *, niter=20, function_code=None, clear_jobs=False):
        """Create a MplMonitor that watches an in-process RunInfo.

        Use this after ``optimize_rigorously(in_process=True, async_=True)``
        to get a live dashboard without a subprocess::

            run_info = decomp.optimize_rigorously(rgcurve, method='BH',
                                                   niter=20, async_=True)
            mon = MplMonitor.for_run_info(run_info, niter=20)
            mon.create_dashboard()
            mon.show()
            mon.start_watching()

        The Resume and Terminate buttons are hidden automatically because
        thread-based runs cannot be killed cleanly.  The Export button
        remains available once the run completes.

        Parameters
        ----------
        run_info : molass.Rigorous.RunInfo.RunInfo
            The RunInfo object returned by ``optimize_rigorously(async_=True)``.
        niter : int, default=20
            Number of optimizer iterations — must match the value passed to
            ``optimize_rigorously()``.  Used to scale the SV-history axis.
        function_code : str, optional
            Forwarded to ``__init__``.
        clear_jobs : bool, default=False
            Set ``True`` to clear existing job folders. Defaults to ``False``
            because an in-process run has already created its working folder.
        """
        mon = cls(source=_RunInfoSource(run_info),
                  function_code=function_code, clear_jobs=clear_jobs)
        # Set attributes that watch_progress() and update_plot() expect.
        # For the subprocess path these are set inside run() / run_impl();
        # for the in-process path we initialize them here.
        mon.niter = niter
        mon.seed = 1234             # unused for in-process (no run_impl call), but watch_progress references it
        mon.num_trials = 0
        mon.max_trials = 0          # in-process runs cannot be auto-resumed (0 means no resume allowed)
        mon.optimizer = run_info.optimizer
        mon.dsets = run_info.dsets
        mon.job_state = None        # lazily set in watch_progress once work_folder is known
        mon.curr_index = None
        mon.work_folder = None      # required by update_plot(); snapshot falls back to optimizer_folder
        return mon

    def clear_jobs(self):
        folder = self.optimizer_folder
        for sub in os.listdir(folder):
            subpath =  os.path.join(folder, sub)
            if os.path.isdir(subpath):
                shutil.rmtree(subpath)
                os.makedirs(subpath, exist_ok=True)

    @property
    def working_folder(self):
        """Path to the current optimizer working folder, or None."""
        if hasattr(self, 'source'):
            return self.source.working_folder
        return None

    def create_dashboard(self):
        self.plot_output = widgets.Output()

        in_process = isinstance(self.source, _RunInfoSource)

        self.status_label = widgets.Label(value="Status: Running")
        self.space_label1 = widgets.Label(value="　　　　")
        self.resume_button = widgets.Button(description="Resume Job", button_style='warning', disabled=True)
        self.resume_button.on_click(self.trigger_resume)
        self.space_label2 = widgets.Label(value="　　　　")
        if not hasattr(self, 'terminate_event'):
            self.terminate_event = threading.Event()
        self.terminate_button = widgets.Button(description="Terminate Job", button_style='danger')
        self.terminate_button.on_click(self.trigger_terminate)
        self.space_label3 = widgets.Label(value="　　　　")
        self.export_button = widgets.Button(description="Export Data", button_style='success', disabled=True)
        self.export_button.on_click(self.export_data)

        if in_process:
            # Resume and Terminate are not meaningful for thread-based runs:
            # niter is fixed and threads cannot be killed cleanly.
            controls_children = [self.status_label,
                                  self.space_label3,
                                  self.export_button]
        else:
            controls_children = [self.status_label,
                                  self.space_label1,
                                  self.resume_button,
                                  self.space_label2,
                                  self.terminate_button,
                                  self.space_label3,
                                  self.export_button]
        self.controls = widgets.HBox(controls_children)

        self.message_output = widgets.Output(layout=widgets.Layout(border='1px solid gray', background_color='gray', padding='10px'))

        # Fix cursor on disabled buttons (VS Code ipywidgets renderer doesn't enforce this)
        self._button_css = widgets.HTML(
            '<style>.widget-button:disabled { cursor: not-allowed !important; opacity: 0.5; }</style>'
        )

        self.dialog_output = widgets.Output()
        self.dashboard = widgets.VBox([self._button_css, self.plot_output, self.controls, self.message_output, self.dialog_output])
        self.dashboard_output = widgets.Output()

    def run(self, optimizer, init_params, niter=20, seed=1234, max_trials=30, work_folder=None, dummy=False, x_shifts=None, debug=False, devel=True):
        self.optimizer = optimizer
        self.init_params = init_params
        self.niter = niter
        self.seed = seed
        self.num_trials = 0
        self.max_trials = max_trials
        self.work_folder = work_folder
        self.x_shifts = x_shifts
        self.run_impl(optimizer, init_params, niter=niter, seed=seed, work_folder=work_folder, dummy=dummy, debug=debug, devel=devel)

    def run_impl(self, optimizer, init_params, niter=20, seed=1234, work_folder=None, dummy=False,
                 optimizer_test=False, debug=False, devel=False):
        from importlib import reload
        import molass_legacy.Optimizer.JobState
        reload(molass_legacy.Optimizer.JobState)
        from molass_legacy.Optimizer.JobState import JobState

        if optimizer_test:
            pass
        else:
            optimizer.prepare_for_optimization(init_params)

        self.source._runner.run(optimizer, init_params, niter=niter, seed=seed, work_folder=work_folder, dummy=dummy, x_shifts=self.x_shifts,
                        optimizer_test=optimizer_test, debug=debug, devel=devel)
        if optimizer_test:
            abs_working_folder = os.path.abspath(work_folder)
        else:
            abs_working_folder = os.path.abspath(self.source.working_folder)
            cb_file = os.path.join(abs_working_folder, 'callback.txt')
            self.job_state = JobState(cb_file, niter)
            # Register this process in the registry
            self._add_to_registry(abs_working_folder)
            self.curr_index = None
        self.logger.info("Starting optimization job in folder: %s with optimizer_test=%s", abs_working_folder, optimizer_test)
        
    def test_subprocess_optimizer(self):
        from importlib import reload
        import molass_legacy.Optimizer.Compatibility
        reload(molass_legacy.Optimizer.Compatibility)
        from molass_legacy.Optimizer.Compatibility import test_subprocess_optimizer_impl
        test_subprocess_optimizer_impl(self)

    def trigger_resume(self, b):
        """Resume optimization from the best parameters of the completed job.

        Called when the user clicks the 'Resume Job' button. Reads the best
        parameters from the latest callback.txt, launches a new subprocess,
        and restarts the watch thread.
        """
        if self.resume_button.disabled:
            return
        self.resume_button.disabled = True
        self.logger.info("Resume requested by user")

        try:
            # Get best params from the completed job
            best_params = self.get_best_params()
            self.init_params = best_params

            # Reset termination flag
            self.terminate_event.clear()

            # Launch a new job (preserving previous job folders)
            self.run_impl(self.optimizer, best_params, niter=self.niter,
                          seed=self.seed, work_folder=None, dummy=False, debug=False)

            self.status_label.value = "Status: Running"
            set_label_color(self.status_label, "green")
            self.terminate_button.disabled = False

            # Restart the watch thread
            self.start_watching()
            self.logger.info("Resumed optimization successfully")
        except Exception as e:
            self.logger.error(f"Resume failed: {e}")
            self.status_label.value = f"Status: Resume failed"
            set_label_color(self.status_label, "red")
            self.resume_button.disabled = False
            with self.message_output:
                clear_output(wait=True)
                print(f"Resume failed: {e}")

    def trigger_terminate(self, b):
        if self.terminate_button.disabled:
            return
        try:
            from molass_legacy.KekLib.IpyUtils import ask_user
        except (ModuleNotFoundError, ImportError):
            # Fallback: terminate immediately without confirmation dialog
            self.terminate_event.set()
            self.status_label.value = "Status: Terminating"
            set_label_color(self.status_label, "yellow")
            self.logger.info("Terminate job requested (no dialog). id(self)=%d", id(self))
            return

        def handle_response(answer):
            print("Callback received:", answer)
            if answer:
                self.terminate_event.set()
                self.status_label.value = "Status: Terminating"
                set_label_color(self.status_label, "yellow")
                self.logger.info("Terminate job requested. id(self)=%d", id(self))
        ask_user("Do you really want to terminate?", callback=handle_response, output_widget=self.dialog_output)

    def show(self, debug=False):
        self.update_plot()
        # with self.dashboard_output:
        display(self.dashboard)
        inject_label_color_css()
        set_label_color(self.status_label, "green")

    def update_plot(self):
        # No data yet — watcher will draw first frame once callback.txt appears.
        if not hasattr(self, 'job_state') or self.job_state is None:
            return

        from importlib import reload
        import molass_legacy.Optimizer.JobStatePlot
        reload(molass_legacy.Optimizer.JobStatePlot)
        from molass_legacy.Optimizer.JobStatePlot import plot_job_state

        # Get current plot info and best params
        plot_info = self.job_state.get_plot_info()
        params = self.get_best_params(plot_info=plot_info)

        # Capture warnings only — do NOT redirect sys.stdout/sys.stderr here.
        # This method runs on a background thread. Any redirect_stdout/sys.stdout
        # modification races with the main notebook thread (e.g. load_rigorous_result
        # also uses redirect_stdout), causing TOCTOU corruption of sys.stdout.
        # The ipywidgets.Output context (self.plot_output) captures any prints
        # from plot_job_state() naturally.
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            with self.plot_output:
                clear_output(wait=True)
                # Use monitor_optimizer (subprocess-equivalent) for objective
                # re-evaluation if available; fall back to self.optimizer.
                # See issue #118.
                display_optimizer = self.monitor_optimizer or self.optimizer
                # Issue #128: compute best accepted SV for widget title.
                _best_sv = None
                try:
                    from molass_legacy.Optimizer.FvScoreConverter import convert_score as _cs
                    _js = getattr(self, 'job_state', None)
                    if _js is not None and hasattr(_js, 'fv') and len(_js.fv) > 0:
                        _best_sv = float(_cs(float(np.min(_js.fv[:, 1]))))
                except Exception:
                    pass
                plot_job_state(self, params, plot_info=plot_info, niter=self.niter,
                               display_optimizer=display_optimizer, best_sv=_best_sv)
                # Close before display to remove from inline backend's auto-show list,
                # preventing a duplicate render. display() still works on a closed Figure.
                plt.close(self.fig)
                display(self.fig)

        # Issue #19 (AI-friendliness): opt-in disk snapshot of the dashboard.
        # The widget-rendered figure is not persisted in cell.outputs, so AI
        # tools and post-hoc reviewers cannot see the live dashboard. When
        # MOLASS_MONITOR_SNAPSHOT=1, write a PNG to <optimizer_folder>/figs/.
        # Issue #22: also write a structured JSON sidecar so AI assistants
        # can inspect what was actually plotted (axis ranges, line summaries,
        # data-coordinate provenance) without OCR'ing the PNG.
        # Note: self.work_folder is the run()-time arg (often None); use
        # self.optimizer_folder which is always set in __init__.
        snap_root = self.work_folder or getattr(self, "optimizer_folder", None)
        if os.environ.get("MOLASS_MONITOR_SNAPSHOT") == "1" and snap_root:
            try:
                snap_dir = os.path.join(snap_root, "figs")
                os.makedirs(snap_dir, exist_ok=True)
                self.fig.savefig(
                    os.path.join(snap_dir, "mplmonitor_latest.png"),
                    dpi=100, bbox_inches="tight",
                )
            except Exception as e:
                self.logger.warning("MplMonitor snapshot failed: %s", e)
            try:
                sidecar = _build_monitor_snapshot_json(self, display_optimizer, params)
                with open(os.path.join(snap_dir, "mplmonitor_latest.json"), "w") as fh:
                    json.dump(sidecar, fh, indent=2, default=str)
            except Exception as e:
                self.logger.warning("MplMonitor JSON sidecar failed: %s", e)

        # Collect warning messages
        messages = []
        messages_counts = {}
        for w in wlist:
            msg = str(w.message)
            if msg in messages_counts:
                messages_counts[msg] += 1
            else:
                messages_counts[msg] = 1
        for msg, count in messages_counts.items():
            if count > 1:
                messages.append(f"Warning: {msg} (x {count})")
            else:
                messages.append(f"Warning: {msg}")

        # Display warning messages in message_output
        if messages:
            with self.message_output:
                clear_output(wait=True)
                for msg in messages:
                    print(msg)

    def watch_progress(self, interval=1.0):
        """Main watching loop that monitors subprocess and updates dashboard.
        
        This runs in a background thread and can be stopped gracefully via stop_watch_event.
        """
        self.logger.info(f"Watch thread started for monitor {self.instance_id}")
        self.is_monitoring = True
        try:
            while True:
                # Check for graceful shutdown request
                if self.stop_watch_event.is_set():
                    self.logger.info("Watch thread shutdown requested")
                    break
                
                exit_loop = False
                has_ended = False
                
                # Check source status with error handling
                try:
                    if not self.source.is_alive():
                        exit_loop = True
                        has_ended = True
                except Exception as e:
                    self.logger.error(f"Error polling source: {e}")
                    # Assume source died if we can't poll it
                    exit_loop = True
                    has_ended = False
                
                # self.logger.info("self.terminate=%s, id(self)=%d", str(self.terminate_event.is_set()), id(self))
                if self.terminate_event.is_set():
                    self.logger.info("Terminating optimization job.")
                    try:
                        self.source.terminate()
                    except Exception as e:
                        self.logger.error(f"Error terminating source: {e}")
                    exit_loop = True

                resume_loop = False
                if exit_loop:
                    if has_ended:
                        self.logger.info("Optimization job ended normally.")
                        self.status_label.value = "Status: Completed"
                        set_label_color(self.status_label, "blue")
                        if self.num_trials < self.max_trials:
                            self.logger.info("Starting a new optimization trial (%d/%d).", self.num_trials, self.max_trials)
                            best_params = self.get_best_params()
                            self.run_impl(self.optimizer, best_params, niter=self.niter, seed=self.seed, work_folder=None, dummy=False, debug=False)
                            self.status_label.value = "Status: Running"
                            set_label_color(self.status_label, "green")
                            resume_loop = True
                        else:
                            self.status_label.value = "Status: Max Trials Reached"
                            set_label_color(self.status_label, "gray")
                            self.terminate_button.disabled = True
                    else:
                        self.logger.info("Optimization job terminated by user.")
                        self.status_label.value = "Status: Terminated"
                        set_label_color(self.status_label, "gray")
                        self.terminate_button.disabled = True

                    # Final redraw with the true best params so that the
                    # mplmonitor_latest.png/json snapshot reflects the actual
                    # best SV rather than the last mid-run update (timing lag).
                    # Force a fresh read of callback.txt by clearing last_mod_time
                    # (JobState.update() skips re-reading when mtime is unchanged,
                    # so without this the final update_plot() sees the same stale
                    # fv array as the last regular mid-run update).
                    if has_ended and hasattr(self, 'job_state') and self.job_state is not None:
                        try:
                            self.job_state.last_mod_time = None  # force fresh read
                            self.job_state.update()
                            self.update_plot()
                        except Exception as _fe:
                            self.logger.warning("Final update_plot failed: %s", _fe)

                    self.save_the_result_figure()

                    # Issue #AI-B: write run_complete.json so AI tools and
                    # post-hoc notebook cells can read best_fv/best_sv without
                    # parsing callback.txt or relying on the widget snapshot.
                    if has_ended and not resume_loop:
                        try:
                            self._write_run_complete_json()
                        except Exception as _rce:
                            self.logger.warning("run_complete.json write failed: %s", _rce)

                    self.num_trials += 1

                    if not resume_loop:
                        # Enable Resume and Export buttons for user interaction
                        self.resume_button.disabled = False
                        self.export_button.disabled = False
                        # Remove from registry when fully done
                        self._remove_from_registry()
                        # Stop monitoring to prevent further updates
                        self.is_monitoring = False
                        self.logger.info("Monitoring stopped - job fully completed")
                        break

                # Lazy job_state initialization for in-process (_RunInfoSource) path.
                # run_impl() is never called for in-process runs, so job_state is not
                # set at monitor creation time.  Once the optimizer thread has written
                # its work_folder, we can locate callback.txt and create JobState.
                if self.is_monitoring and isinstance(self.source, _RunInfoSource) \
                        and (not hasattr(self, 'job_state') or self.job_state is None):
                    wf = self.source._ri.work_folder
                    if wf is not None:
                        cb_file = os.path.join(wf, 'callback.txt')
                        from molass_legacy.Optimizer.JobState import JobState
                        self.job_state = JobState(cb_file, self.niter)
                        self._add_to_registry(os.path.abspath(wf))
                        self.curr_index = None

                # Only update if we have a valid job_state and monitoring is active
                if self.is_monitoring and hasattr(self, 'job_state') and self.job_state is not None:
                    try:
                        self.job_state.update()
                        if self.job_state.has_changed():
                            self.update_plot()
                            # Enable Export as soon as there's data to export
                            if self.export_button.disabled and self.dsets is not None:
                                self.export_button.disabled = False
                    except Exception as e:
                        self.logger.error(f"Error updating job state: {e}")
                        # If we can't update job state, assume job is dead
                        self.is_monitoring = False
                        self.logger.info("Stopping monitoring due to job state error")
                        break
                        
                time.sleep(interval)
        finally:
            self.is_monitoring = False
            self.watch_thread = None
            self.logger.info(f"Watch thread ended for monitor {self.instance_id}")

    def start_watching(self):
        """Start the background thread that monitors optimization progress.
        
        Only one watch thread can be active per monitor instance. If a thread is
        already running, this method will log a warning and return without starting
        a new thread.
        """
        # Check if thread is already running
        if self.watch_thread is not None and self.watch_thread.is_alive():
            self.logger.warning(f"Watch thread already running for monitor {self.instance_id}")
            print("⚠ Warning: Watch thread is already running for this monitor.")
            return
        
        # Clear stop event in case it was set previously
        self.stop_watch_event.clear()
        
        # Avoid Blocking the Main Thread:
        # Never run a long or infinite loop in the main thread in Jupyter if you want widget interactivity.
        self.watch_thread = threading.Thread(target=self.watch_progress, daemon=True)
        self.watch_thread.start()
        self.logger.info(f"Started watch thread for monitor {self.instance_id}")
    
    def stop_watching(self, timeout=5.0):
        """Stop the background watch thread gracefully.
        
        Args:
            timeout: Maximum time in seconds to wait for thread to stop.
        
        Returns:
            bool: True if thread stopped successfully, False if timeout occurred.
        """
        if self.watch_thread is None or not self.watch_thread.is_alive():
            self.logger.info("No active watch thread to stop")
            return True
        
        self.logger.info(f"Stopping watch thread for monitor {self.instance_id}")
        self.is_monitoring = False  # Stop monitoring immediately
        self.stop_watch_event.set()
        self.watch_thread.join(timeout=timeout)
        
        # After join: check if it actually stopped
        if self.watch_thread is not None and self.watch_thread.is_alive():
            self.logger.warning(f"Watch thread did not stop within {timeout}s")
            return False
        else:
            self.logger.info("Watch thread stopped successfully")
            self.watch_thread = None
            return True
    
    def is_watching(self):
        """Check if the watch thread is currently active.
        
        Returns:
            bool: True if watch thread is running, False otherwise.
        """
        return self.watch_thread is not None and self.watch_thread.is_alive()

    def terminate(self, timeout=5.0):
        """Terminate the optimization run and clean up.

        This is the recommended way to stop an optimization from code
        (e.g. in a notebook cell after ``get_current_decomposition``).

        It kills the subprocess directly, stops the watch thread, and
        removes the process from the registry.

        Args:
            timeout: Maximum seconds to wait for the watch thread to stop.

        Returns:
            bool: True if shutdown completed within the timeout.
        """
        self.terminate_event.set()
        # Kill the source directly — do not rely on the watch thread,
        # which may exit before reaching the terminate_event check.
        try:
            self.source.terminate()
        except Exception as e:
            self.logger.error(f"Error terminating source: {e}")
        result = self.stop_watching(timeout=timeout)
        self._remove_from_registry()
        return result

    def get_best_params(self, plot_info=None):
        if plot_info is None:
            plot_info = self.job_state.get_plot_info()

        x_array = plot_info[-1]

        if len(x_array) == 0:
            self.curr_index = 0
            return self.init_params

        fv = plot_info[0]
        k = np.argmin(fv[:,1])
        self.curr_index = k
        best_params = x_array[k]
        return best_params

    def get_progress_info(self):
        """Return current optimization progress as a dictionary.

        This method exposes the same timing and score information that is
        rendered visually in the matplotlib progress chart, making it
        accessible to AI agents and programmatic callers.

        Returns
        -------
        dict
            Keys:
            - ``status`` (str): Current status label text.
            - ``trial`` (int): Current trial number (0-based).
            - ``max_trials`` (int): Maximum number of trials.
            - ``num_evals`` (int): Number of function evaluations so far.
            - ``best_fv`` (float or None): Best objective function value.
            - ``starting_time`` (str): When the job started (HH:MM format).
            - ``time_elapsed`` (str): Time since start (H.MM format).
            - ``ending_time`` (str): Estimated completion time (HH:MM format).
            - ``time_ahead`` (str): Estimated remaining time (H.MM format).
            - ``is_running`` (bool): Whether the subprocess is still alive.
        """
        from molass_legacy.Optimizer.ProgressChart import (
            get_time_started, get_time_elapsed, guess_ending_time,
            get_remaining_time,
        )

        raw_status = getattr(self, 'status_label', None) and self.status_label.value or 'unknown'
        # Strip the "Status: " prefix used by the widget display
        clean_status = raw_status.replace('Status: ', '', 1) if isinstance(raw_status, str) else raw_status

        info = {
            'status': clean_status,
            'trial': getattr(self, 'num_trials', 0),
            'max_trials': getattr(self, 'max_trials', 0),
            'num_evals': 0,
            'best_fv': None,
            'starting_time': '',
            'time_elapsed': '',
            'ending_time': '',
            'time_ahead': '',
            'is_running': self.is_watching(),
        }

        if not hasattr(self, 'job_state'):
            return info

        plot_info = self.job_state.get_plot_info()
        fv = plot_info[0]

        if fv is None or len(fv) == 0:
            return info

        info['num_evals'] = int(fv[-1, 0])
        info['best_fv'] = float(np.min(fv[:, 1]))
        info['starting_time'] = get_time_started(fv)
        info['time_elapsed'] = get_time_elapsed(fv)
        ending_str, finish_time = guess_ending_time(fv, niter=self.niter)
        info['ending_time'] = ending_str
        info['time_ahead'] = get_remaining_time(fv, finish_time)

        return info

    def get_status_summary(self):
        """Print a concise one-line progress summary.

        Intended for AI agents and programmatic callers who cannot see the
        matplotlib widget.  Returns the summary string as well.

        Returns
        -------
        str
            A single-line summary of the current optimization state.
        """
        info = self.get_progress_info()
        parts = [f"Status: {info['status']}"]
        parts.append(f"Trial {info['trial']}/{info['max_trials']}")
        if info['best_fv'] is not None:
            parts.append(f"best_fv={info['best_fv']:.6f}")
        parts.append(f"evals={info['num_evals']}")
        if info['time_elapsed']:
            parts.append(f"elapsed={info['time_elapsed'].strip()}")
        if info['ending_time']:
            parts.append(f"ETA={info['ending_time'].strip()}")
        if info['time_ahead']:
            parts.append(f"ahead={info['time_ahead'].strip()}")
        summary = " | ".join(parts)
        print(summary)
        return summary

    def get_current_curves(self):
        """Return the data and model curves currently shown on the monitor.

        This is the **monitor readability** API (molass-legacy issue #31): it
        exposes as plain numpy arrays the same curves that are rendered in the
        matplotlib dashboard, so that an AI agent can reason from the same
        evidence as a human looking at the screen.

        Returns
        -------
        dict with keys:

        ``xr_frames``
            1-D array — XR frame indices (x-axis for all XR panels).
        ``xr_data``
            1-D array — XR data elution curve (observed total intensity per frame).
        ``xr_model``
            1-D array — XR model total (sum of all components including baseline).
        ``xr_components``
            2-D array (n_components × n_frames) — individual XR component curves.
        ``uv_frames``
            1-D array — UV frame indices (mapped to XR frame scale via linear mapping a·x+b).
        ``uv_data``
            1-D array — UV data elution curve.
        ``uv_model``
            1-D array — UV model total.
        ``uv_components``
            2-D array (n_components × n_frames) — individual UV component curves.
        ``sv_history``
            1-D array — SV values at each accepted optimization evaluation.
        ``best_sv``
            float — best SV seen so far (None if not available).
        ``params``
            1-D array — current best optimizer parameters.

        Returns None if the monitor has no data yet (job_state not initialised).

        Example
        -------
        >>> state = monitor.get_current_curves()
        >>> if state:
        ...     peak_data  = state['uv_frames'][np.argmax(state['uv_data'])]
        ...     peak_model = state['uv_frames'][np.argmax(state['uv_model'])]
        ...     print(f"UV data peak: {peak_data}, model peak: {peak_model}, shift: {peak_model - peak_data:+.1f}")
        """
        if not hasattr(self, 'job_state') or self.job_state is None:
            return None

        from molass_legacy.Optimizer.FvScoreConverter import convert_score

        plot_info = self.job_state.get_plot_info()
        params = self.get_best_params(plot_info=plot_info)

        # Re-evaluate the objective function to get the lrf_info data structure
        # that holds both data and model curves on the same scale.
        try:
            optimizer = self.monitor_optimizer or self.optimizer
            lrf_info = optimizer.objective_func(params, return_lrf_info=True)
        except Exception:
            return None

        # SV history from callback
        fv_array = plot_info[0]
        if fv_array is not None and len(fv_array) > 0:
            sv_history = np.array([convert_score(float(v)) for v in fv_array[:, 1]])
            best_sv = float(np.max(sv_history))
        else:
            sv_history = np.array([])
            best_sv = None

        return {
            'xr_frames':    lrf_info.x,
            'xr_data':      lrf_info.y,
            'xr_model':     lrf_info.xr_ty,
            'xr_components': lrf_info.scaled_xr_cy_array,
            'uv_frames':    lrf_info.uv_x,
            'uv_data':      lrf_info.uv_y,
            'uv_model':     lrf_info.uv_ty,
            'uv_components': lrf_info.scaled_uv_cy_array,
            'sv_history':   sv_history,
            'best_sv':      best_sv,
            'params':       params,
        }

    def save_the_result_figure(self, fig_file=None):
        if fig_file is None:
            figs_folder = os.path.join(self.optimizer_folder, "figs")
            if not os.path.exists(figs_folder):
                os.makedirs(figs_folder)
            fig_file = os.path.join(figs_folder, "fig-%03d.jpg" % self.num_trials)
        self.fig.savefig(fig_file)

    def _write_run_complete_json(self):
        """Write run_complete.json to <optimizer_folder>/figs/ on job completion.

        This file is the canonical, zero-parse-required answer to
        "what was the best result?" for AI tools and notebook cells in a new
        session.  It is always written on normal completion regardless of
        MOLASS_MONITOR_SNAPSHOT.  (Issue #AI-B)

        Keys
        ----
        schema_version : int
        completed_at   : ISO timestamp
        best_fv        : float  — global minimum fv from callback.txt
        best_sv        : float  — corresponding SV score
        n_evals        : int    — total evaluations logged in callback.txt
        n_accepted     : int    — accepted evaluations
        analysis_folder : str  — absolute path to the analysis folder
        """
        from molass_legacy.Optimizer.FvScoreConverter import convert_score
        payload = {
            "schema_version": 1,
            "completed_at": datetime.now().isoformat(),
            "best_fv": None,
            "best_sv": None,
            "n_evals": 0,
            "n_accepted": 0,
            "analysis_folder": getattr(self, "work_folder", None) or self.optimizer_folder,
        }
        js = getattr(self, "job_state", None)
        if js is not None and hasattr(js, "fv") and len(js.fv) > 0:
            fv_arr = js.fv  # shape (N, 2) — [eval_counter, fv]
            best_fv = float(np.min(fv_arr[:, 1]))
            payload["best_fv"] = best_fv
            payload["best_sv"] = float(convert_score(best_fv))
            payload["n_evals"] = int(len(fv_arr))
        figs_folder = os.path.join(self.optimizer_folder, "figs")
        os.makedirs(figs_folder, exist_ok=True)
        out_path = os.path.join(figs_folder, "run_complete.json")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        self.logger.info("run_complete.json written: %s", out_path)


    def export_data(self, b, debug=True):
        if self.export_button.disabled:
            return
        if debug:
            from importlib import reload
            import molass_legacy.Optimizer.LrfExporter
            reload(molass_legacy.Optimizer.LrfExporter)
        from .LrfExporter import LrfExporter

        params = self.get_best_params()
        # All output must go into self.message_output so it is visible in the
        # dashboard.  Bare print() in a button callback is routed to a kernel
        # stream that is not connected to any notebook cell output, making
        # success/error messages completely invisible to the user.
        with self.message_output:
            from IPython.display import clear_output
            clear_output(wait=True)
            try:
                if self.dsets is None:
                    print("Error: dsets not set. Cannot export.")
                    return
                # Derive the export folder from work_folder (in-process path)
                # so it lands next to the job folder rather than relying on
                # get_setting('optimizer_folder') which may be unset.
                export_folder = None
                if self.work_folder is not None:
                    export_folder = os.path.normpath(
                        os.path.join(self.work_folder, '../../exported'))
                exporter = LrfExporter(self.optimizer, params, self.dsets)
                folder = exporter.export(folder=export_folder)
                fig_file = os.path.join(folder, "result_fig.jpg")
                self.save_the_result_figure(fig_file=fig_file)
                print(f"Exported to folder: {folder}")
            except Exception as exc:
                from molass_legacy.KekLib.ExceptionTracebacker import log_exception
                log_exception(self.logger, "export: ")
                print(f"Failed to export due to: {exc}")

    # ===== Dashboard Recovery =====
    
    def redisplay_dashboard(self):
        """Redisplay the dashboard after it has been cleared.
        
        This method allows you to reconnect to a running monitor after
        accidentally clearing notebook outputs. Call this to get the
        dashboard back.
        
        Example:
            # After clearing outputs, retrieve and redisplay:
            monitor = MplMonitor.get_active_monitor()
            monitor.redisplay_dashboard()
        """
        if not hasattr(self, 'dashboard'):
            print("Dashboard not initialized. Call create_dashboard() first.")
            return
        
        # Update plot with current state
        if hasattr(self, 'job_state'):
            self.update_plot()
        
        # Redisplay the dashboard
        display(self.dashboard)
        inject_label_color_css()
        
        # Restore status label color based on current status
        status = self.status_label.value
        if "Running" in status:
            set_label_color(self.status_label, "green")
        elif "Completed" in status:
            set_label_color(self.status_label, "blue")
        elif "Terminated" in status or "Max Trials" in status:
            set_label_color(self.status_label, "gray")
        elif "Terminating" in status:
            set_label_color(self.status_label, "yellow")
        
        print(f"Dashboard redisplayed for monitor {self.instance_id}")
        self.logger.info(f"Dashboard redisplayed for instance {self.instance_id}")
    
    @classmethod
    def get_active_monitor(cls):
        """Get the most recently created active monitor instance.
        
        Returns the last MplMonitor instance that was created and is still active.
        Useful for recovering access to a monitor after clearing notebook outputs.
        
        Returns:
            MplMonitor: The most recent active monitor, or None if no monitors exist.
        
        Example:
            # After clearing outputs:
            monitor = MplMonitor.get_active_monitor()
            if monitor:
                monitor.redisplay_dashboard()
        """
        if not _ACTIVE_MONITORS:
            print("No active monitors found.")
            return None
        
        # Return the most recent (last inserted) monitor
        return list(_ACTIVE_MONITORS.values())[-1]
    
    @classmethod
    def get_all_active_monitors(cls):
        """Get all active monitor instances.
        
        Returns:
            list: List of all active MplMonitor instances.
        """
        return list(_ACTIVE_MONITORS.values())
    
    @classmethod
    def show_active_monitors(cls):
        """Display information about all active monitor instances.
        
        Shows a summary of all currently active monitors including their
        status, process ID, and working folder if available.
        
        Example:
            MplMonitor.show_active_monitors()
        """
        if not _ACTIVE_MONITORS:
            print("No active monitors found.")
            return
        
        print(f"Found {len(_ACTIVE_MONITORS)} active monitor(s):\n")
        
        for idx, (instance_id, monitor) in enumerate(_ACTIVE_MONITORS.items(), 1):
            print(f"Monitor #{idx} (ID: {instance_id})")
            
            # Status
            if hasattr(monitor, 'status_label'):
                print(f"  Status: {monitor.status_label.value}")
            else:
                print(f"  Status: Not started")
            
            # Process info
            if hasattr(monitor, 'process_id') and monitor.process_id:
                print(f"  Process ID: {monitor.process_id}")
            
            # Thread info
            if hasattr(monitor, 'watch_thread'):
                if monitor.watch_thread is not None and monitor.watch_thread.is_alive():
                    print(f"  Watch Thread: ACTIVE (ID: {monitor.watch_thread.ident})")
                else:
                    print(f"  Watch Thread: NOT RUNNING")
            
            # Working folder
            if hasattr(monitor, 'runner') and hasattr(monitor.runner, 'working_folder'):
                print(f"  Working folder: {monitor.runner.working_folder}")
            
            # Trial info
            if hasattr(monitor, 'num_trials') and hasattr(monitor, 'max_trials'):
                print(f"  Trials: {monitor.num_trials}/{monitor.max_trials}")
            
            print()
        
        print("To redisplay a dashboard:")
        print("  monitor = MplMonitor.get_active_monitor()")
        print("  monitor.redisplay_dashboard()")
    
    @classmethod
    def cleanup_orphaned_threads(cls):
        """Stop watch threads for monitors that are no longer needed.
        
        This method identifies and stops watch threads that are still running
        for monitors that may have lost their dashboard. Useful for cleaning up
        after accidentally clearing notebook outputs multiple times.
        
        Example:
            MplMonitor.cleanup_orphaned_threads()
        """
        if not _ACTIVE_MONITORS:
            print("No active monitors found.")
            return
        
        orphaned_count = 0
        stopped_count = 0
        
        for instance_id, monitor in _ACTIVE_MONITORS.items():
            if hasattr(monitor, 'watch_thread') and monitor.watch_thread is not None:
                if monitor.watch_thread.is_alive():
                    orphaned_count += 1
                    print(f"Monitor {instance_id}: Watch thread is running")
                    
                    # Check if we should stop it
                    response = input("  Stop this watch thread? (y/n): ").strip().lower()
                    if response == 'y':
                        print("  Stopping thread...")
                        success = monitor.stop_watching(timeout=5.0)
                        if success:
                            print("  Thread stopped successfully.")
                            stopped_count += 1
                        else:
                            print("  Warning: Thread did not stop cleanly.")
        
        if orphaned_count == 0:
            print("No active watch threads found.")
        else:
            print(f"\nStopped {stopped_count} of {orphaned_count} watch thread(s).")
    
    @classmethod
    def stop_all_threads(cls, force=False):
        """Stop all watch threads immediately without asking.
        
        Use this to quickly stop all monitoring threads, for example when
        experiencing periodic screen blackouts after killing processes.
        
        Args:
            force: If True, doesn't wait for graceful shutdown
        
        Example:
            # Quick fix for blackout issue
            MplMonitor.stop_all_threads()
        """
        if not _ACTIVE_MONITORS:
            print("No active monitors found.")
            return
        
        stopped_count = 0
        failed_count = 0
        
        for instance_id, monitor in list(_ACTIVE_MONITORS.items()):
            if hasattr(monitor, 'watch_thread') and monitor.watch_thread is not None:
                if monitor.watch_thread.is_alive():
                    print(f"Stopping watch thread for monitor {instance_id}...")
                    timeout = 0.5 if force else 5.0
                    success = monitor.stop_watching(timeout=timeout)
                    if success:
                        stopped_count += 1
                    else:
                        failed_count += 1
                        print(f"  Warning: Thread {instance_id} did not stop cleanly.")
        
        total = stopped_count + failed_count
        if total == 0:
            print("No active watch threads found.")
        else:
            print(f"\nStopped {stopped_count} of {total} watch thread(s).")
            if failed_count > 0:
                print(f"⚠ {failed_count} thread(s) did not stop cleanly. Consider restarting the kernel.")

    # ===== Process Registry Management =====
    
    def _get_registry_path(self):
        """Get the path to the process registry file."""
        return os.path.join(self.optimizer_folder, 'active_processes.json')
    
    def _load_registry(self):
        """Load the process registry from disk."""
        registry_path = self._get_registry_path()
        if not os.path.exists(registry_path):
            return {}
        try:
            with open(registry_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            self.logger.warning(f"Failed to load registry: {e}")
            return {}
    
    def _save_registry(self, registry):
        """Save the process registry to disk."""
        registry_path = self._get_registry_path()
        try:
            with open(registry_path, 'w') as f:
                json.dump(registry, f, indent=2)
        except IOError as e:
            self.logger.error(f"Failed to save registry: {e}")
    
    def _add_to_registry(self, working_folder):
        """Add current process to the registry."""
        if not hasattr(self.source, '_runner') or not hasattr(self.source._runner, 'process') or self.source._runner.process is None:
            self.logger.warning("No process to register")
            return
        
        pid = self.source._runner.process.pid
        self.process_id = str(pid)
        registry = self._load_registry()
        registry[self.process_id] = {
            'pid': pid,
            'working_folder': working_folder,
            'timestamp': datetime.now().isoformat(),
            'status': 'running'
        }
        self._save_registry(registry)
        self.logger.info(f"Registered process PID {pid} in registry")
    
    def _remove_from_registry(self):
        """Remove current process from the registry."""
        if self.process_id is None:
            return
        
        registry = self._load_registry()
        if self.process_id in registry:
            del registry[self.process_id]
            self._save_registry(registry)
            self.logger.info(f"Removed process {self.process_id} from registry")
        self.process_id = None
    
    def _is_process_alive(self, pid):
        """Check if a process with given PID is alive."""
        try:
            import psutil
            return psutil.pid_exists(pid)
        except ImportError:
            # Fallback to OS-specific method if psutil not available
            import signal
            if os.name == 'nt':  # Windows
                import subprocess
                try:
                    result = subprocess.run(
                        ['tasklist', '/FI', f'PID eq {pid}'],
                        capture_output=True, text=True, timeout=2
                    )
                    return str(pid) in result.stdout
                except Exception:
                    return False
            else:  # Unix-like
                try:
                    os.kill(pid, 0)
                    return True
                except OSError:
                    return False
    
    def _cleanup_orphaned_processes(self, auto_terminate=True):
        """Clean up orphaned processes from previous sessions.
        
        Args:
            auto_terminate: If True, automatically terminate orphaned processes.
                           If False, only report them.
        """
        registry = self._load_registry()
        if not registry:
            return
        
        orphaned = []
        cleaned = []
        
        for proc_id, info in list(registry.items()):
            pid = info.get('pid')
            if pid is None:
                cleaned.append(proc_id)
                continue
            
            # Check if process is still alive
            if not self._is_process_alive(pid):
                self.logger.info(f"Process {pid} is no longer running, removing from registry")
                cleaned.append(proc_id)
            else:
                orphaned.append(info)
                if auto_terminate:
                    self.logger.warning(f"Terminating orphaned process {pid} from {info.get('timestamp')}")
                    try:
                        import psutil
                        proc = psutil.Process(pid)
                        proc.terminate()
                        try:
                            proc.wait(timeout=5)
                        except psutil.TimeoutExpired:
                            proc.kill()
                        cleaned.append(proc_id)
                    except ImportError:
                        # Fallback without psutil
                        if os.name == 'nt':  # Windows
                            import subprocess
                            subprocess.run(['taskkill', '/F', '/PID', str(pid)], 
                                         capture_output=True)
                        else:  # Unix-like
                            import signal
                            os.kill(pid, signal.SIGTERM)
                        cleaned.append(proc_id)
                    except Exception as e:
                        self.logger.error(f"Failed to terminate process {pid}: {e}")
        
        # Clean up registry
        for proc_id in cleaned:
            if proc_id in registry:
                del registry[proc_id]
        
        if cleaned:
            self._save_registry(registry)
            self.logger.info(f"Cleaned up {len(cleaned)} entries from process registry")
        
        if orphaned and not auto_terminate:
            print(f"Warning: Found {len(orphaned)} orphaned processes:")
            for info in orphaned:
                print(f"  - PID {info['pid']} started at {info['timestamp']}")
            print("Call MplMonitor.cleanup_orphaned_processes() to terminate them.")
    
    @classmethod
    def cleanup_orphaned_processes(cls, optimizer_folder=None):
        """Class method to manually clean up orphaned processes.
        
        This can be called from a fresh notebook cell without an instance:
            MplMonitor.cleanup_orphaned_processes()
        
        Args:
            optimizer_folder: Path to optimizer folder. If None, uses default from settings.
        """
        if optimizer_folder is None:
            analysis_folder = get_setting("analysis_folder")
            optimizer_folder = os.path.join(analysis_folder, "optimized")
        
        registry_path = os.path.join(optimizer_folder, 'active_processes.json')
        
        if not os.path.exists(registry_path):
            print("No active process registry found.")
            return
        
        try:
            with open(registry_path, 'r') as f:
                registry = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Failed to load registry: {e}")
            return
        
        if not registry:
            print("No active processes in registry.")
            return
        
        print(f"Found {len(registry)} process(es) in registry:")
        
        for proc_id, info in list(registry.items()):
            pid = info.get('pid')
            timestamp = info.get('timestamp', 'unknown')
            working_folder = info.get('working_folder', 'unknown')
            
            # Check if process is alive
            try:
                import psutil
                is_alive = psutil.pid_exists(pid)
            except ImportError:
                # Fallback
                if os.name == 'nt':
                    import subprocess
                    try:
                        result = subprocess.run(
                            ['tasklist', '/FI', f'PID eq {pid}'],
                            capture_output=True, text=True, timeout=2
                        )
                        is_alive = str(pid) in result.stdout
                    except Exception:
                        is_alive = False
                else:
                    try:
                        os.kill(pid, 0)
                        is_alive = True
                    except OSError:
                        is_alive = False
            
            if is_alive:
                print(f"  - PID {pid}: RUNNING (started {timestamp})")
                print(f"    Folder: {working_folder}")
                response = input(f"    Terminate this process? (y/n): ").strip().lower()
                if response == 'y':
                    try:
                        import psutil
                        proc = psutil.Process(pid)
                        proc.terminate()
                        try:
                            proc.wait(timeout=5)
                            print(f"    Terminated PID {pid}")
                        except psutil.TimeoutExpired:
                            proc.kill()
                            print(f"    Killed PID {pid} (did not respond to terminate)")
                        del registry[proc_id]
                    except ImportError:
                        if os.name == 'nt':
                            import subprocess
                            subprocess.run(['taskkill', '/F', '/PID', str(pid)])
                        else:
                            import signal
                            os.kill(pid, signal.SIGTERM)
                        print(f"    Terminated PID {pid}")
                        del registry[proc_id]
                    except Exception as e:
                        print(f"    Failed to terminate: {e}")
            else:
                print(f"  - PID {pid}: NOT RUNNING (removing from registry)")
                del registry[proc_id]
        
        # Save updated registry
        try:
            with open(registry_path, 'w') as f:
                json.dump(registry, f, indent=2)
            print("\nRegistry updated.")
        except IOError as e:
            print(f"Failed to save registry: {e}")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        # Stop watch thread gracefully
        try:
            if hasattr(self, 'watch_thread') and self.watch_thread is not None:
                if self.watch_thread.is_alive():
                    if hasattr(self, 'logger'):
                        self.logger.info(f"Stopping watch thread in __del__ for {self.instance_id}")
                    self.stop_watch_event.set()
                    self.watch_thread.join(timeout=2.0)
        except Exception:
            pass  # Ignore errors during cleanup
        
        try:
            self._remove_from_registry()
        except Exception:
            pass  # Ignore errors during cleanup
        
        # Remove from global instance registry
        try:
            if hasattr(self, 'instance_id') and self.instance_id in _ACTIVE_MONITORS:
                del _ACTIVE_MONITORS[self.instance_id]
                if hasattr(self, 'logger'):
                    self.logger.info(f"Unregistered monitor instance {self.instance_id}")
        except Exception:
            pass  # Ignore errors during cleanup


