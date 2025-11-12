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
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
from molass_legacy._MOLASS.SerialSettings import get_setting

class MplMonitor:
    def __init__(self, debug=True):
        if debug:
            from importlib import reload
            import molass_legacy.Optimizer.BackRunner
            reload(molass_legacy.Optimizer.BackRunner)
        from molass_legacy.Optimizer.BackRunner import BackRunner
        analysis_folder = get_setting("analysis_folder")
        optimizer_folder = os.path.join(analysis_folder, "optimized")
        logpath = os.path.join(optimizer_folder, 'monitor.log')
        self.fileh = logging.FileHandler(logpath, 'w')
        format_csv_ = '%(asctime)s,%(levelname)s,%(name)s,%(message)s'
        datefmt_ = '%Y-%m-%d %H:%M:%S'
        self.formatter_csv_ = logging.Formatter(format_csv_, datefmt_)
        self.fileh.setFormatter(self.formatter_csv_)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(self.fileh)
        self.runner = BackRunner()
        self.logger.info("MplMonitor initialized.")
        self.logger.info(f"Optimizer job folder: {self.runner.optjob_folder}")
        self.result_list = []

    def clear_jobs(self):
        folder = self.runner.optjob_folder
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)

    def run(self, optimizer, init_params, niter=100, seed=1234, work_folder=None, dummy=False, debug=False):
        self.optimizer = optimizer
        self.init_params = init_params
        self.runner.run(optimizer, init_params, niter=niter, seed=seed, work_folder=work_folder, dummy=dummy, debug=debug)
        abs_working_folder = os.path.abspath(self.runner.working_folder)
        self.cb_file = os.path.join(abs_working_folder, 'callback.txt')
        self.logger.info("Starting optimization job in folder: %s", abs_working_folder)
        self.curr_index = None

    def create_dashboard(self):
        self.terminate_event = threading.Event()
        self.terminate_button = widgets.Button(description="Terminate Job", button_style='danger')
        self.terminate_button.on_click(self.terminate_job)
        self.plot_output = widgets.Output()
        # self.message_output = widgets.Output()
        self.message_output = widgets.Output(layout=widgets.Layout(border='1px solid gray', background_color='gray', padding='10px'))
        self.controls = widgets.HBox([self.terminate_button])
        self.dashboard = widgets.VBox([self.plot_output, self.controls, self.message_output])

    def terminate_job(self, b):
        self.terminate_event.set()
        self.logger.info("Terminate job requested. id(self)=%d", id(self))

    def show(self, debug=False):
        self.update_plot(params=self.init_params)
        display(self.dashboard)

    def update_plot(self, params=None, job_state=None):
        from molass_legacy.Optimizer.JobStatePlot import plot_job_state
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
                    plot_job_state(self, params=params, job_state=job_state)
                    display(self.fig)
                    plt.close(self.fig)
            finally:
                # Restore stdout/stderr
                sys.stdout = old_stdout
                sys.stderr = old_stderr

        # Collect all messages
        messages = []
        # Warnings
        for w in wlist:
            messages.append(f"Warning: {w.message}")
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
        while True:
            ret = self.runner.poll()
            if ret is not None:
                break
            self.logger.info("self.terminate=%s, id(self)=%d", str(self.terminate_event.is_set()), id(self))
            if self.terminate_event.is_set():
                self.logger.info("Terminating optimization job.")
                self.runner.terminate()
                with self.plot_output:
                    clear_output(wait=True)  # Remove any possibly remaining plot
                break
            job_state = self.update_job_state()
            if job_state is not None:
                self.update_plot(job_state=job_state)
            time.sleep(interval)

    def start_watching(self):
        # Never run a long or infinite loop in the main thread in Jupyter if you want widget interactivity.
        threading.Thread(target=self.watch_progress, daemon=True).start()

    def update_job_state(self, debug=True):
        if not os.path.exists(self.cb_file):
            self.logger.warning("callback.txt file not found: %s", self.cb_file)
            return
        fv_list, x_list = self.read_callback_txt(self.cb_file)
        if debug:
                self.logger.info("updating information from %s, len(x_list)=%d", self.cb_file, len(x_list))

        fv, xmax = self.get_fv_array(fv_list)
        return fv, xmax, np.array(x_list)

    def get_running_solver_info(self):
        return self.runner.solver, 100

    def get_fv_array(self, fv_list):
        fv = np.array(fv_list)
        solver_name, niter = self.get_running_solver_info()
        if solver_name == "ultranest":
            from molass_legacy.Solvers.UltraNest.SolverUltraNest import get_max_ncalls
            # task: unify this estimation
            xmax = get_max_ncalls(niter)
        else:
            xmax = 500000   # default large number
        return fv, xmax

    def read_callback_txt(self, cb_file):
        from .StateSequence import read_callback_txt_impl

        fv_list, x_list = read_callback_txt_impl(cb_file)
        if len(fv_list) == 0:
            # note that the first record is always included in these lists
            self.logger.info("making up the first record in callback.txt from the caller.")

        return fv_list, x_list
    
    def append_result(self, last_info):
        from .OptJobResultInfo import OptJobResultInfo
        fv_vector = last_info[0][:,1]
        k = np.argmin(fv_vector)
        fv = fv_vector[k]
        if self.known_best_fv is None or fv < self.known_best_fv:
            self.known_best_fv = fv
            self.known_best_index = len(self.result_list)
            self.logger.info("updated known_best_index to %d with known_best_fv=%g", self.known_best_index, fv)

        x_array = last_info[1]
        result_info = OptJobResultInfo(fv=fv, params=x_array[k])
        self.result_list.append(result_info)
        self.logger.info("appended to result_list[%d]: fv=%g", len(self.result_list)-1, fv)
