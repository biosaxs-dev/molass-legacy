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
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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
        self.fileh = logging.FileHandler(logpath, 'a')
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

    def run(self, optimizer, init_params, niter=100, seed=1234, work_folder=None, debug=False):
        self.optimizer = optimizer
        self.init_params = init_params
        self.runner.run(optimizer, init_params, niter=niter, seed=seed, work_folder=work_folder)
        abs_working_folder = os.path.abspath(self.runner.working_folder)
        self.cb_file = os.path.join(abs_working_folder, 'callback.txt')
        self.logger.info("Starting optimization job in folder: %s", abs_working_folder)

    def terminate_job(self, b):
        self.logger.info("Terminating optimization job...")
        self.runner.terminate()

    def create_dashboard(self):
        self.terminate_button = widgets.Button(description="Terminate Job", button_style='danger')
        self.terminate_button.on_click(self.terminate_job)
        self.plot_output = widgets.Output()
        # self.message_output = widgets.Output()
        self.message_output = widgets.Output(layout=widgets.Layout(border='1px solid gray', background_color='gray', padding='10px'))
        self.controls = widgets.HBox([self.terminate_button])
        self.dashboard = widgets.VBox([self.plot_output, self.controls, self.message_output])
    
    def show(self, debug=False):
        self.update_plot()
        display(self.dashboard)

    def update_plot(self, change=None):
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
                    self.fig = fig = plt.figure(figsize=(18, 9))
                    gs = GridSpec(33, 15)
                    axes = []
                    for j in range(3):
                        j_ = j*5
                        ax = fig.add_subplot(gs[0:16,j_:j_+5])
                        axes.append(ax)

                    axt = axes[1].twinx()
                    axes.append(axt)
                    self.axes = axes
                    self.prog_ax = fig.add_subplot(gs[17:21,2:])
                    peak_ax = fig.add_subplot(gs[21:25,2:])
                    rg_ax = fig.add_subplot(gs[25:29,2:])
                    map_ax = fig.add_subplot(gs[29:33,2:])
                    self.prog_axes = [self.prog_ax, peak_ax, rg_ax, map_ax]
                    for ax in self.prog_axes[0:3]:
                        ax.set_xticklabels([])
                    self.prog_title_axes = [fig.add_subplot(gs[17+i*4:21+i*4,0:2]) for i in range(0,4)]
                    prog_titles = ["Function SV", "Peak Top Positions", "Rg Values", "Mapped Range"]
                    for ax, title in zip(self.prog_title_axes, prog_titles):
                        ax.set_axis_off()
                        ax.text(-0.3, 0.5, title, fontsize=16)

                    axis_info = (fig, (*axes,))
                    self.optimizer.objective_func(self.init_params, plot=True, axis_info=axis_info)
                    display(fig)
                    plt.close(fig)
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
        last_state_info = None
        while True:
            ret = self.runner.poll()
            if ret is not None:
                break
            self.update_job_state()
            time.sleep(interval)
    
    def update_job_state(self, debug=True):
        if not os.path.exists(self.cb_file):
            self.logger.warning("callback.txt file not found: %s", self.cb_file)
            return
        fv_list, x_list = self.read_callback_txt(self.cb_file)
        if debug:
                self.logger.info("updating information from %s, len(x_list)=%d", self.cb_file, len(x_list))
  
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
