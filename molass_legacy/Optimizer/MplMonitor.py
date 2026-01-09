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
    def __init__(self, function_code=None, clear_jobs=True, xr_only=False, debug=True):
        if debug:
            from importlib import reload
            import molass_legacy.Optimizer.BackRunner
            reload(molass_legacy.Optimizer.BackRunner)
        from molass_legacy.Optimizer.BackRunner import BackRunner
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
        self.runner = BackRunner(xr_only=xr_only)
        self.logger.info("MplMonitor initialized.")
        self.logger.info(f"Optimizer job folder: {self.runner.optjob_folder}")
        self.result_list = []
        self.suptitle = None
        self.func_code = function_code
        self.process_id = None  # Will be set when process starts
        self.instance_id = id(self)
        self.watch_thread = None  # Will be set when watching starts
        self.stop_watch_event = threading.Event()  # For graceful thread shutdown
        
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
        self._cleanup_orphaned_processes()

    def clear_jobs(self):
        folder = self.optimizer_folder
        for sub in os.listdir(folder):
            subpath =  os.path.join(folder, sub)
            if os.path.isdir(subpath):
                shutil.rmtree(subpath)
                os.makedirs(subpath, exist_ok=True)

    def create_dashboard(self):
        self.plot_output = widgets.Output()

        self.status_label = widgets.Label(value="Status: Running")
        self.space_label1 = widgets.Label(value="　　　　")
        self.skip_button = widgets.Button(description="Skip Job", button_style='warning', disabled=True)
        self.space_label2 = widgets.Label(value="　　　　")
        if not hasattr(self, 'terminate_event'):
            self.terminate_event = threading.Event()
        self.terminate_button = widgets.Button(description="Terminate Job", button_style='danger')
        self.terminate_button.on_click(self.trigger_terminate)
        self.space_label3 = widgets.Label(value="　　　　")
        self.export_button = widgets.Button(description="Export Data", button_style='success', disabled=True)
        self.export_button.on_click(self.export_data)
        self.controls = widgets.HBox([self.status_label,
                                      self.space_label1,
                                      self.skip_button,
                                      self.space_label2,
                                      self.terminate_button,
                                      self.space_label3,
                                      self.export_button])

        self.message_output = widgets.Output(layout=widgets.Layout(border='1px solid gray', background_color='gray', padding='10px'))

        self.dashboard = widgets.VBox([self.plot_output, self.controls, self.message_output])
        self.dashboard_output = widgets.Output()
        self.dialog_output = widgets.Output()

    def run(self, optimizer, init_params, niter=20, seed=1234, max_trials=30, work_folder=None, dummy=False, x_shifts=None, debug=False, devel=True):
        self.optimizer = optimizer
        self.init_params = init_params
        self.nitrer = niter
        self.seed = seed
        self.num_trials = 0
        self.max_trials = max_trials
        self.x_shifts = x_shifts
        self.run_impl(optimizer, init_params, niter=niter, seed=seed, work_folder=work_folder, dummy=dummy, debug=debug, devel=devel)

    def run_impl(self, optimizer, init_params, niter=20, seed=1234, work_folder=None, dummy=False, debug=False, devel=False):
        from importlib import reload
        import molass_legacy.Optimizer.JobState
        reload(molass_legacy.Optimizer.JobState)
        from molass_legacy.Optimizer.JobState import JobState

        optimizer.prepare_for_optimization(init_params)

        self.runner.run(optimizer, init_params, niter=niter, seed=seed, work_folder=work_folder, dummy=dummy, x_shifts=self.x_shifts, debug=debug, devel=devel)
        abs_working_folder = os.path.abspath(self.runner.working_folder)
        cb_file = os.path.join(abs_working_folder, 'callback.txt')
        self.job_state = JobState(cb_file, niter)
        self.logger.info("Starting optimization job in folder: %s", abs_working_folder)
        self.curr_index = None
        # Register this process in the registry
        self._add_to_registry(abs_working_folder)

    def trigger_terminate(self, b):
        from molass_legacy.KekLib.IpyUtils import ask_user

        def handle_response(answer):
            print("Callback received:", answer)
            if answer:
                self.terminate_event.set()
                self.status_label.value = "Status: Terminating"
                set_label_color(self.status_label, "yellow")
                self.logger.info("Terminate job requested. id(self)=%d", id(self))
        display(self.dialog_output)
        ask_user("Do you really want to terminate?", callback=handle_response, output_widget=self.dialog_output)

    def show(self, debug=False):
        self.update_plot()
        # with self.dashboard_output:
        display(self.dashboard)
        inject_label_color_css()
        set_label_color(self.status_label, "green")

    def update_plot(self):
        from importlib import reload
        import molass_legacy.Optimizer.JobStatePlot
        reload(molass_legacy.Optimizer.JobStatePlot)
        from molass_legacy.Optimizer.JobStatePlot import plot_job_state

        # Get current plot info and best params
        plot_info = self.job_state.get_plot_info()
        params = self.get_best_params(plot_info=plot_info)

        # Prepare to capture warnings and prints
        buf_out = io.StringIO()
        buf_err = io.StringIO()
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout = buf_out
            sys.stderr = buf_err
            try:
                with self.plot_output:
                    clear_output(wait=True)
                    plot_job_state(self, params, plot_info=plot_info, niter=self.nitrer)
                    display(self.fig)
                    plt.close(self.fig)
            finally:
                # Restore stdout/stderr
                sys.stdout = old_stdout
                sys.stderr = old_stderr

        # Unique warning messages and counts
        messages_counts = {}
        for w in wlist:
            msg = str(w.message)
            if msg in messages_counts:
                messages_counts[msg] += 1
            else:
                messages_counts[msg] = 1

        # Collect all messages
        messages = []
        # Warnings
        for msg, count in messages_counts.items():
            if count > 1:
                messages.append(f"Warning: {msg} (x {count})")
            else:
                messages.append(f"Warning: {msg}")
        # Print output and errors
        out_str = buf_out.getvalue()
        err_str = buf_err.getvalue()
        if out_str.strip():
            messages.append(out_str.strip())
        if err_str.strip():
            messages.append(err_str.strip())

        # Display all messages in message_output
        with self.message_output:
            clear_output(wait=True)
            for msg in messages:
                print(msg)

    def watch_progress(self, interval=1.0):
        """Main watching loop that monitors subprocess and updates dashboard.
        
        This runs in a background thread and can be stopped gracefully via stop_watch_event.
        """
        self.logger.info(f"Watch thread started for monitor {self.instance_id}")
        try:
            while True:
                # Check for graceful shutdown request
                if self.stop_watch_event.is_set():
                    self.logger.info("Watch thread shutdown requested")
                    break
                
                exit_loop = False
                has_ended = False
                ret = self.runner.poll()

                if ret is not None:
                    exit_loop = True
                    has_ended = True
                # self.logger.info("self.terminate=%s, id(self)=%d", str(self.terminate_event.is_set()), id(self))
                if self.terminate_event.is_set():
                    self.logger.info("Terminating optimization job.")
                    self.runner.terminate()
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
                            self.run_impl(self.optimizer, best_params, niter=self.nitrer, seed=self.seed, work_folder=None, dummy=False, debug=False)
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

                    self.save_the_result_figure()
                    self.num_trials += 1

                    with self.plot_output:
                        clear_output(wait=True)  # Remove any possibly remaining plot
                    if not resume_loop:
                        # Remove from registry when fully done
                        self._remove_from_registry()
                        break

                self.job_state.update()
                if self.job_state.has_changed():
                    self.update_plot()
                    # clear_output(wait=True)
                    # display(self.dashboard)
                time.sleep(interval)
        finally:
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
        self.stop_watch_event.set()
        self.watch_thread.join(timeout=timeout)
        
        if self.watch_thread.is_alive():
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

    def save_the_result_figure(self, fig_file=None):
        if fig_file is None:
            figs_folder = os.path.join(self.optimizer_folder, "figs")
            if not os.path.exists(figs_folder):
                os.makedirs(figs_folder)
            fig_file = os.path.join(figs_folder, "fig-%03d.jpg" % self.num_trials)
        self.fig.savefig(fig_file)

    def export_data(self, b, debug=True):
        if debug:
            from importlib import reload
            import Optimizer.LrfExporter
            reload(Optimizer.LrfExporter)
        from .LrfExporter import LrfExporter

        params = self.optimizer.init_params
        try:
            exporter = LrfExporter(self.optimizer, params, self.dsets)
            folder = exporter.export()
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
        if not hasattr(self.runner, 'process') or self.runner.process is None:
            self.logger.warning("No process to register")
            return
        
        pid = self.runner.process.pid
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


