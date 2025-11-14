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
from molass_legacy.KekLib.IpyLabelUtils import inject_label_color_css
from molass_legacy._MOLASS.SerialSettings import get_setting
from molass_legacy.KekLib.IpyLabelUtils import inject_label_color_css, set_label_color
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
        self.suptitle = None
        self.func_code = "G0346"

    def clear_jobs(self):
        folder = self.runner.optjob_folder
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)

    def get_running_solver_info(self):
        return self.runner.solver, 100

    def run(self, optimizer, init_params, niter=100, seed=1234, work_folder=None, dummy=False, debug=False):
        from importlib import reload
        import molass_legacy.Optimizer.JobState
        reload(molass_legacy.Optimizer.JobState)
        from molass_legacy.Optimizer.JobState import JobState
        solver_name, niter = self.get_running_solver_info()
        self.optimizer = optimizer
        self.init_params = init_params
        self.runner.run(optimizer, init_params, niter=niter, seed=seed, work_folder=work_folder, dummy=dummy, debug=debug)
        abs_working_folder = os.path.abspath(self.runner.working_folder)
        cb_file = os.path.join(abs_working_folder, 'callback.txt')
        self.job_state = JobState(cb_file, solver_name, niter)
        self.logger.info("Starting optimization job in folder: %s", abs_working_folder)
        self.curr_index = None

    def create_dashboard(self):
        self.plot_output = widgets.Output()

        self.status_label = widgets.Label(value="Status: Running")
        self.space_label1 = widgets.Label(value="　　　　")
        self.skip_button = widgets.Button(description="Skip Job", button_style='warning', disabled=True)
        self.space_label2 = widgets.Label(value="　　　　")
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
        self.update_plot(params=self.init_params)
        # with self.dashboard_output:
        display(self.dashboard)
        inject_label_color_css()
        set_label_color(self.status_label, "green")

    def update_plot(self, params=None, plot_info=None):
        from importlib import reload
        import molass_legacy.Optimizer.JobStatePlot
        reload(molass_legacy.Optimizer.JobStatePlot)
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
                    plot_job_state(self, params=params, plot_info=plot_info)
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
            exit_loop = False
            ret = self.runner.poll()
            if ret is not None:
                exit_loop = True
            # self.logger.info("self.terminate=%s, id(self)=%d", str(self.terminate_event.is_set()), id(self))
            if self.terminate_event.is_set():
                self.logger.info("Terminating optimization job.")
                self.runner.terminate()
                exit_loop = True
            if exit_loop:
                self.status_label.value = "Status: Terminated"
                set_label_color(self.status_label, "gray")
                self.terminate_button.disabled = True
                with self.plot_output:
                    clear_output(wait=True)  # Remove any possibly remaining plot
                break

            self.job_state.update()
            if self.job_state.has_changed():
                plot_info = self.job_state.get_plot_info()
                self.update_plot(plot_info=plot_info)
                # clear_output(wait=True)
                # display(self.dashboard)
            time.sleep(interval)

    def start_watching(self):
        # Avoid Blocking the Main Thread:
        # Never run a long or infinite loop in the main thread in Jupyter if you want widget interactivity.
        threading.Thread(target=self.watch_progress, daemon=True).start()
    
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


