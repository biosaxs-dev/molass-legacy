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
from molass_legacy._MOLASS.SerialSettings import get_setting, set_setting
from molass_legacy.KekLib.IpyLabelUtils import inject_label_color_css, set_label_color

class MplMonitor:
    def __init__(self, function_code=None, clear_jobs=True, debug=True):
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
        self.runner = BackRunner()
        self.logger.info("MplMonitor initialized.")
        self.logger.info(f"Optimizer job folder: {self.runner.optjob_folder}")
        self.result_list = []
        self.suptitle = None
        self.func_code = function_code

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

    def run(self, optimizer, init_params, niter=20, seed=1234, max_trials=30, work_folder=None, dummy=False, x_shifts=None, debug=False):
        self.optimizer = optimizer
        self.init_params = init_params
        self.nitrer = niter
        self.seed = seed
        self.num_trials = 0
        self.max_trials = max_trials
        self.x_shifts = x_shifts
        self.run_impl(optimizer, init_params, niter=niter, seed=seed, work_folder=work_folder, dummy=dummy, debug=debug)

    def run_impl(self, optimizer, init_params, niter=20, seed=1234, work_folder=None, dummy=False, debug=False):
        from importlib import reload
        import molass_legacy.Optimizer.JobState
        reload(molass_legacy.Optimizer.JobState)
        from molass_legacy.Optimizer.JobState import JobState

        optimizer.prepare_for_optimization(init_params)

        self.runner.run(optimizer, init_params, niter=niter, seed=seed, work_folder=work_folder, dummy=dummy, x_shifts=self.x_shifts, debug=debug)
        abs_working_folder = os.path.abspath(self.runner.working_folder)
        cb_file = os.path.join(abs_working_folder, 'callback.txt')
        self.job_state = JobState(cb_file, niter)
        self.logger.info("Starting optimization job in folder: %s", abs_working_folder)
        self.curr_index = None

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
        while True:
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
                    break

            self.job_state.update()
            if self.job_state.has_changed():
                self.update_plot()
                # clear_output(wait=True)
                # display(self.dashboard)
            time.sleep(interval)

    def start_watching(self):
        # Avoid Blocking the Main Thread:
        # Never run a long or infinite loop in the main thread in Jupyter if you want widget interactivity.
        threading.Thread(target=self.watch_progress, daemon=True).start()
    
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


