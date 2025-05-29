"""
    Optimizer.FullOptDialog.py

    Copyright (c) 2021-2024, SAXS Team, KEK-PF
"""
import sys
import os
import numpy as np
import scipy.stats as stats
import re
import logging
from importlib import reload
from molass_legacy.KekLib.BasicUtils import ordinal_str
from molass_legacy.KekLib.OurTkinter import Tk, Dialog, ttk
from molass_legacy.KekLib.TkCustomWidgets import FolderEntry
from molass_legacy._MOLASS.Version import get_version_string, is_developing_version
import molass_legacy.KekLib.CustomMessageBox as MessageBox
from molass_legacy.KekLib.ExceptionTracebacker import log_exception
from molass_legacy._MOLASS.SerialSettings import get_setting
from .OptConstants import MIN_NUM_COMPONENTS, MAX_NUM_COMPONENTS
from .BackRunner import BackRunner
from .JobStateInfo import JobStateInfo
from .OptJobInfo import OptJobInfo, JOB_STATES
from .OptJobResultInfo import OptJobResultInfo
from molass_legacy.Peaks.PeakEditor import DRIFT_TYPES
from .OptimizerSettings import get_advanced_settings_text
from .ElutionComposer import COMPOSER_CB_TEXT

CANVAS_DEBUG = False
POLL_INTERVAL = 1000
# DEVELOP_MODE = sys.executable.find("pythonw") < 0
DEVELOP_MODE = is_developing_version()

class FullOptDialog(Dialog):
    def __init__(self, parent, parent_dialog, optinit_info, **kwargs):
        self.logger = logging.getLogger(__name__)
        self.parent = parent
        self.parent_dialog = parent_dialog

        self.fullopt = None
        self.runner = BackRunner()
        self.optinit_info = optinit_info
        self.class_code = optinit_info.class_code
        self.param_init_type = optinit_info.param_init_type
        self.bounds_type = optinit_info.bounds_type     # not used yet
        # self.job_list = kwargs.pop('job_list', [])
        self.job_list = optinit_info.job_list
        self.demo_index = kwargs.pop('demo_index', 0)
        self.inspecting = kwargs.pop('inspecting', False)   # True for Result View
        self.kwargs = kwargs
        self.terminating = False
        self.attending_current = True
        self.is_busy_ = not self.inspecting
        self.result_list = []
        self.known_best_fv = None
        self.known_best_index = None
        self.state_info = JobStateInfo(optinit_info, self.demo_index)
        self.fullopt = self.state_info.fullopt
        self.init_info = self.state_info.get_init_info()
        self.elution_model = get_setting("elution_model")
        self.recompute_rg_curve = get_setting("recompute_rg_curve")
        pid = os.getpid()
        version = get_version_string()
        self.poll_using_filesize = False     # temporary fix for the shared memory bug
        Dialog.__init__(self, parent, "Optimization Monitor - %s pid=%d" % (version, pid), visible=False)

    def initialize(self):
        pass

    def get_skipbutton_name(self):
        return "Skip" if self.is_busy_ else "Resume"

    def is_busy(self):
        return self.is_busy_

    def show(self):
        if self.inspecting:
            init_method = self.set_inspection_mode
        else:
            init_method = self.run
        self.is_focused = True
        self.after(100, init_method)
        self._show()

    def body(self, body_frame):
        if CANVAS_DEBUG:
            import molass_legacy.Optimizer.JobStateCanvas
            reload(molass_legacy.Optimizer.JobStateCanvas)
        from .JobStateCanvas import JobStateCanvas
        self.canvas = JobStateCanvas(body_frame, self)
        self.canvas.pack()
        self.num_params = 22 if self.init_info is None else self.init_info[1].shape[1]
        self.logger.info("num_params=%d", self.num_params)

        self.num_jobs = Tk.IntVar()
        self.folder_box = None

        if len(self.job_list) == 0:
            self.folders = []           # used in add_new_job
            self.add_new_job(init=True)
            self.curr_job = 0
        else:
            self.folders = [job_info.name for job_info in self.job_list]
            job_name = self.kwargs.pop('job_name')
            self.curr_job = int(job_name)

        num_jobs = len(self.job_list)
        self.num_jobs.set(num_jobs)

        self.canvas.draw_main()

        pframe = Tk.Frame(body_frame)
        pframe.pack(padx=20)
        self.build_panel(pframe)

    def get_job_info(self):
        return self.job_list[self.curr_job]

    def build_panel(self, pframe):
        curr_job_info = self.job_list[self.curr_job]
        print("curr_job_info=", curr_job_info)

        space_height = 5

        grid_row = 0
        space =Tk.Frame(pframe, height=space_height)
        space.grid(row=grid_row, column=0)

        grid_row += 1
        label = Tk.Label(pframe, text="Job Root Folder: ")
        label.grid(row=grid_row, column=0, sticky=Tk.E)
        self.optjob_folder = Tk.StringVar()
        self.optjob_folder.set(self.runner.get_optjob_folder())

        self.root_folder_entry = FolderEntry(pframe, textvariable=self.optjob_folder, width=100,
            on_entry_cb=self.on_entry_root_folder)
        self.root_folder_entry.grid(row=grid_row, column=1, columnspan=11, sticky=Tk.W)
        self.root_folder_entry.config(state=Tk.DISABLED)    # 
 
        grid_row += 1
        space =Tk.Frame(pframe, height=space_height)
        space.grid(row=grid_row, column=0)

        grid_row += 1
        label = Tk.Label(pframe, text="Max Number of Trials: ")
        label.grid(row=grid_row, column=0, sticky=Tk.E)

        self.maxnum_trials = Tk.IntVar()
        self.maxnum_trials.set(self.optinit_info.maxnum_trials)
        num_jobs = self.num_jobs.get()
        self.mn_sbox = Tk.Spinbox(pframe, textvariable=self.maxnum_trials,
                          from_=num_jobs, to=999, increment=1,
                          justify=Tk.CENTER, width=6)
        self.mn_sbox.grid(row=grid_row, column=1, sticky=Tk.W)
        self.maxnum_trials.trace("w", self.maxnum_trials_tracer)

        space_width = 20
        grid_row += 1
        space =Tk.Frame(pframe, height=space_height)
        space.grid(row=grid_row, column=0)

        grid_row += 1
        grid_col = 0
        label = Tk.Label(pframe, text="Number of Jobs: ")
        label.grid(row=grid_row, column=grid_col, sticky=Tk.E)

        grid_col += 1
        label = Tk.Label(pframe, textvariable=self.num_jobs, bg='white', width=3)
        label.grid(row=grid_row, column=grid_col, sticky=Tk.W)

        grid_col += 1
        space = Tk.Frame(pframe, width=space_width)
        space.grid(row=grid_row, column=grid_col)

        grid_col += 1
        label = Tk.Label(pframe, text="Attending Job: ")
        label.grid(row=grid_row, column=grid_col)

        grid_col += 1
        self.curr_folder = Tk.StringVar()
        self.curr_folder.set(curr_job_info[0])
        self.folder_box = ttk.Combobox(master=pframe, values=self.folders, textvariable=self.curr_folder,
                                width=5, justify=Tk.CENTER, state=Tk.DISABLED)
        self.folder_box.grid(row=grid_row, column=grid_col)
        self.curr_folder.trace("w", self.curr_folder_tracer)

        grid_col += 1
        space = Tk.Frame(pframe, width=space_width)
        space.grid(row=grid_row, column=grid_col)

        grid_col += 1
        label = Tk.Label(pframe, text="Job State: ")
        label.grid(row=grid_row, column=grid_col)

        grid_col += 1
        self.job_state = Tk.StringVar()
        self.job_state.set(curr_job_info[1])
        label = Tk.Label(pframe, textvariable=self.job_state, bg='white')
        label.grid(row=grid_row, column=grid_col)

        grid_col += 1
        space = Tk.Frame(pframe, width=5)
        space.grid(row=grid_row, column=grid_col)

        grid_col += 1
        label = Tk.Label(pframe, text="PID: ")
        label.grid(row=grid_row, column=grid_col)

        grid_col += 1
        self.job_pid = Tk.StringVar()
        label = Tk.Label(pframe, textvariable=self.job_pid, bg='white')
        label.grid(row=grid_row, column=grid_col)

        grid_col += 1
        space = Tk.Frame(pframe, width=space_width)
        space.grid(row=grid_row, column=grid_col)

        grid_col += 1
        label = Tk.Label(pframe, text="Number of Components: ")
        label.grid(row=grid_row, column=grid_col)

        grid_col += 1
        self.n_components = Tk.IntVar()
        # self.n_components.set(curr_job_info[2])
        self.n_components.set(self.fullopt.n_components - 1)    # just for display to the user
        self.nc_sbox = Tk.Spinbox(pframe, textvariable=self.n_components,
                          from_=MIN_NUM_COMPONENTS, to=MAX_NUM_COMPONENTS, increment=1,
                          justify=Tk.CENTER, width=6, state=Tk.DISABLED)
        self.nc_sbox.grid(row=grid_row, column=grid_col)

        grid_col += 1
        space = Tk.Frame(pframe, width=space_width)
        space.grid(row=grid_row, column=grid_col)

        grid_col += 1
        label = Tk.Label(pframe, text="Baseline: ")
        label.grid(row=grid_row, column=grid_col)

        label_text = get_advanced_settings_text()
        if label_text > "":
            label = Tk.Label(pframe, text=label_text)
            label.grid(row=grid_row-2, column=grid_col-2, columnspan=6)

        grid_col += 1
        unified_baseline_type = get_setting("unified_baseline_type")
        self.drift_type = Tk.StringVar()
        self.drift_type.set(DRIFT_TYPES[unified_baseline_type - 1])
        self.drift_type_box = ttk.Combobox(master=pframe,
                    values=DRIFT_TYPES, textvariable=self.drift_type, width=15, state=Tk.DISABLED)
        self.drift_type_box.grid(row=grid_row, column=grid_col)

        grid_col += 1
        space = Tk.Frame(pframe, width=space_width)
        space.grid(row=grid_row, column=grid_col)

        grid_col += 1
        label = Tk.Label(pframe, text="Number of Iterations: ")
        label.grid(row=grid_row, column=grid_col)

        grid_col += 1
        self.num_iter = Tk.IntVar()
        self.num_iter.set(curr_job_info[4])
        self.num_iter_entry = Tk.Entry(pframe, textvariable=self.num_iter, width=5, justify=Tk.CENTER)
        self.num_iter_entry.grid(row=grid_row, column=grid_col)

        grid_col += 1
        space = Tk.Frame(pframe, width=space_width)
        space.grid(row=grid_row, column=grid_col)

        grid_col += 1
        label = Tk.Label(pframe, text="Seed: ")
        label.grid(row=grid_row, column=grid_col)

        grid_col += 1
        self.seed = Tk.IntVar()
        self.seed.set(curr_job_info[5])
        self.seed_entry = Tk.Entry(pframe, textvariable=self.seed, width=10, justify=Tk.CENTER)
        self.seed_entry.grid(row=grid_row, column=grid_col)

        if is_developing_version():
            button = Tk.Button(pframe, text="Devel Test", command=self.devel_test)
            # note that 4 rows here above include 2 space rows.
            button.grid(row=grid_row-4, rowspan=4, column=grid_col, sticky=Tk.E)
 
        grid_row += 1
        space =Tk.Frame(pframe, height=space_height)
        space.grid(row=grid_row, column=0)

        grid_row += 1
        grid_col = 1
        self.try_model_composing = Tk.IntVar()
        self.try_model_composing.set(get_setting('try_model_composing'))
        cb = Tk.Checkbutton(pframe, text=COMPOSER_CB_TEXT, variable=self.try_model_composing)
        cb.grid(row=grid_row, column=grid_col, columnspan=5, sticky=Tk.W)
        self.try_model_composing_cb = cb

        grid_row += 1
        space =Tk.Frame(pframe, height=space_height)
        space.grid(row=grid_row, column=0)

    def buttonbox(self):
        box = Tk.Frame(self)
        box.pack()

        show_modify_button = DEVELOP_MODE and self.elution_model == 0
        padx = 90 if show_modify_button else 120

        w = Tk.Button(box, text="◀ Close", width=10, command=self.cancel)
        w.pack(side=Tk.LEFT, padx=padx, pady=10)
        self.cancel_button = w

        if show_modify_button:
            in_folder = get_setting("in_folder")
            state = Tk.NORMAL if in_folder.find("HasA") >= 0 else Tk.DISABLED
            w = Tk.Button(box, text="▽ Modify Restart", width=20, command=self.modify_restart, state=state)
            w.pack(side=Tk.LEFT, padx=padx, pady=10)
            self.modify_button = w

        name = self.get_skipbutton_name()
        w = Tk.Button(box, text=name, width=10, command=self.skip_to_next_trial)
        w.pack(side=Tk.LEFT, padx=padx, pady=10)
        self.skip_button = w
        self.remaining_dependents = [self.skip_button]

        w = Tk.Button(box, text="Terminate", width=12, command=self.terminate,
                            state=Tk.DISABLED)
        w.pack(side=Tk.LEFT, padx=padx, pady=10)
        self.stop_button = w
        self.stop_button_configs = [w.cget(a) for a in ['fg', 'bg']]

        w = Tk.Button(box, text="▶ Export", width=12, command=self.export)
        w.pack(side=Tk.LEFT, padx=padx, pady=10)

        self.job_state_dependents = [self.nc_sbox, self.seed_entry, self.num_iter_entry]
        self.bind("<Escape>", self.cancel)

        # do this later because it takes a while for those states getting ready
        self.after(500, self.update_try_model_composing_cb)

    def update_skipbutton_name(self):
        button_name = self.get_skipbutton_name()
        self.skip_button.config(text=button_name)

    def cancel(self):
        print("FullOptDialog.cancel")
        if self.is_busy_:
            MessageBox.showinfo("Can't Close Notification",
                "There seems to remain a still running job.\n"
                'You have to wait until the finish or "Terminate" before close.',
                parent=self)
            return

        self.canvas.close()
        Dialog.cancel(self)

    def run(self, restart_params=None, real_bounds=None, debug=True):
        from .FullOptResult import FILES, OPTIONAL_FILES

        assert self.job_state.get() == JOB_STATES[0]
        self.is_busy_ = True
        self.update_skipbutton_name()
        self.update_count = 0

        for w in self.job_state_dependents:
            w.config(state=Tk.DISABLED)

        if restart_params is None:
            fv_array, x_array = self.state_info.get_init_info()[0:2]
            init_params = x_array[0]
            if self.param_init_type == 0:
                # foreground plot has been initialized in the JobStateCanvas' initialization
                self.logger.info("got init_params from the initialization: num_params=", len(init_params))
            else:
                init_params = self.get_known_best_params(init_params)
        else:
            self.logger.info("resuming with restart_params=%s", restart_params)
            init_params = restart_params

        # try model composing
        # note that this mus be done before prepare_init_state
        work_folder = self.runner.get_work_folder()
        composite_info_txt = os.path.join(work_folder, OPTIONAL_FILES[0])   # composite_info.txt

        optimizer = self.fullopt
        try_model_composing = self.try_model_composing.get()
        composite = optimizer.composite
        if composite.really_composite:
            composite.save(composite_info_txt)
        else:
            if try_model_composing and len(self.job_list) > 1:
            # if try_model_composing:
                if debug:
                    from importlib import reload
                    import Optimizer.ElutionComposer
                    reload(Optimizer.ElutionComposer)
                from .ElutionComposer import ElutionComposer
                composer = ElutionComposer(optimizer, init_params, self.optinit_info.sd)
                if composer.reducible():
                    new_composite = composer.make_composite()
                    new_composite.save(composite_info_txt)
                    optimizer.set_composite(new_composite)

        # prepare for initial state
        if self.param_init_type == 0:
            pass
        else:
            # foreground plot must be synchronized with the background job
            self.state_info.prepare_init_state(init_params=init_params, real_bounds=real_bounds)
            self.canvas.update_init_state(self.state_info)

        niter = self.num_iter.get()
        seed = self.seed.get()

        # note that self.try_model_composing may have been changed when resuming from the "Result View"
        try_model_composing = self.try_model_composing.get()

        self.runner.run(optimizer, init_params, niter=niter, seed=seed, work_folder=work_folder)
        self.job_state.set(JOB_STATES[1])
        self.job_list[self.curr_job].update_state(JOB_STATES[1])
        self.stop_button.config(fg='white', bg='red', state=Tk.NORMAL)

        self.np_shm = self.runner.np_shm
        if self.np_shm is None:
            self.poll_using_filesize = True
        if self.poll_using_filesize:
            self.last_filesize = None
        else:
            self.last_callback_counter = None

        self.callback_txt_path = self.runner.get_callback_txt_path()
        self.after(POLL_INTERVAL, self.poll)

    def add_new_job(self, init=False):
        num_jobs = len(self.job_list)
        job_name = "%03d" % num_jobs
        job_state = JOB_STATES[0]

        if init:
            drift_type = DRIFT_TYPES[0]
            try:
                niter = self.optinit_info.n_iterations
                seed = self.optinit_info.seed
            except:
                niter = 100
                seed = np.random.randint(100000, 999999)
            n_components = int((self.num_params - 4)/6)
        else:
            n_components = self.n_components.get()
            drift_type = self.drift_type.get()
            niter = self.num_iter.get()
            seed = np.random.randint(100000, 999999)

        self.job_list.append(OptJobInfo(job_name, job_state, n_components, drift_type, niter, seed))
        self.folders.append(self.job_list[-1][0])

        num_jobs = len(self.job_list)
        self.num_jobs.set(num_jobs)

        if not init:
            self.folder_box.config(values=self.folders)
            self.mn_sbox.config(from_=num_jobs)

        if not init:
            num_remaining = self.maxnum_trials.get() - self.num_jobs.get()
            if num_remaining == 0:
                for w in self.remaining_dependents:
                    w.config(state=Tk.DISABLED)

    def poll(self):
        if self.is_busy_:
            ret = self.runner.poll()
            if ret is None:
                self.after(POLL_INTERVAL, self.poll)
                if self.job_pid.get() == "":
                    self.job_pid.set(str(self.runner.getpid()))
            else:
                self.logger.info("job %s finished with returncode=%d after %d state updates.", self.job_list[-1][0], ret, self.update_count)
                self.inspecting = True      # in order to enable "Resume", 
                # consider whether self.set_inspection_mode() is required
                self.job_pid.set("")
                job_state = JOB_STATES[2] if ret == 0 else JOB_STATES[3]
                self.job_state.set(job_state)
                self.job_list[self.curr_job].update_state(job_state)
                self.save_the_result_figure()
                self.try_update_information(append=True)    # append the result info before adding a new job
                num_remaining = self.maxnum_trials.get() - self.num_jobs.get()
                if num_remaining > 0 and not self.terminating:
                    self.start_next_trial()
                else:
                    self.curr_folder_tracer()
                    self.is_busy_ = False
                    self.update_skipbutton_name()
                    if num_remaining == 0:
                        for w in self.remaining_dependents:
                            w.config(state=Tk.DISABLED)
            self.try_update_information()

    def save_the_result_figure(self, fig_file=None):
        if fig_file is None:
            from .TheUtils import get_optimizer_folder
            optimizer_folder = get_optimizer_folder()
            figs_folder = os.path.join(optimizer_folder, "figs")
            if not os.path.exists(figs_folder):
                os.makedirs(figs_folder)
            fig_file = os.path.join(figs_folder, "fig-%s.jpg" % self.job_list[-1][0])

        self.canvas.save_the_figure(fig_file)

    def start_next_trial(self, restart_params=None, real_bounds=None):
        num_jobs = len(self.job_list)
        job_name = "%03d" % num_jobs
        if self.recompute_rg_curve:
            import RgProcess.RgCurveComputer
            reload(RgProcess.RgCurveComputer)
            from RgProcess.RgCurveComputer import RgCurveComputerDialog
            dialog = RgCurveComputerDialog(self, self, job_name)
            dialog.show()
            if not dialog.applied:
                return

        self.add_new_job()
        self.curr_folder.set(job_name)
        self.update()   # this will end up to call curr_folder_tracer
        self.run(restart_params=restart_params, real_bounds=real_bounds)

    def try_update_information(self, append=False, debug=False):
        if self.poll_using_filesize:
            try:
                filesize = os.path.getsize(self.callback_txt_path)
            except:
                # probably not yet exist
                timestamp = None
                filesize = 0

            updated = ( self.last_filesize is None
                        or filesize != self.last_filesize
                        )
            if debug:
                self.logger.info("try_update_information: filesize=%s, updated=%s;  ",      # ";  " has been added to avoid log overwrite confusion
                              filesize, updated)
        else:
            callback_counter = self.np_shm.array[0]
            updated = ( self.last_callback_counter is None
                        or callback_counter != self.last_callback_counter
                        )
            if debug:
                self.logger.info("try_update_information: callback_counter=%d, updated=%s;  ",      # ";  "  has been added to avoid log overwrite confusion
                             callback_counter, updated) 

        if (append      # make sure to pass the best result to the next trial
            or updated):
            try:
                self.update_information(append=append)
            except:
                log_exception(self.logger, "update_information: ")

        if self.poll_using_filesize:
            self.last_filesize = filesize
        else:
            self.last_callback_counter = callback_counter

    def update_information(self, work_folder=None, append=False, debug=True):
        if work_folder is None:
            work_folder = self.get_curr_work_folder()
        cb_file = os.path.join(work_folder, 'callback.txt')
        fv_list, x_list = self.read_callback_txt(cb_file)
        if debug:
            self.logger.info("updating information from %s, len(x_list)=%d", cb_file, len(x_list))
        xmax = self.estimate_xmax(fv_list)
        seed = self.seed.get()
        demo_info = np.array(fv_list), np.array(x_list), xmax, seed
        self.canvas.set_demo_info(demo_info)
        if append:
            self.append_result(demo_info)
        self.update_count += 1

    def get_curr_work_folder(self):
        optjob_folder = self.runner.get_optjob_folder()
        curr_folder = self.curr_folder.get()
        return os.path.join(optjob_folder, curr_folder)

    def estimate_xmax(self, fv_list):
        niter = self.num_iter.get()
        counter = fv_list[-1][0]
        if counter == 0:
            # init state
            xmax = 500000*niter//100
        else:
            xmax = int(counter * (niter+1)/len(fv_list))
        return xmax

    def get_progress(self):
        niter = self.num_iter.get()
        fv_array = self.canvas.demo_info[0]
        return fv_array.shape[0]/niter

    def read_callback_txt(self, cb_file):
        from .StateSequence import read_callback_txt_impl

        fv_list, x_list = read_callback_txt_impl(cb_file)
        if len(fv_list) == 0:
            # note that the first record is always included in these lists
            self.logger.info("making up the first record in callback.txt from the caller.")
            fv_array, x_array = self.state_info.get_init_info()[0:2]
            fv_list.append(fv_array[0])
            x_list.append(x_array[0])

        return fv_list, x_list

    def curr_folder_tracer(self, *args):
        job_name = self.curr_folder.get()
        i = self.folders.index(job_name)
        print("job_name=", job_name, i)
        self.curr_job = i
        job_state = self.job_list[i][1]
        self.job_state.set(job_state)
        state = Tk.NORMAL if job_state == JOB_STATES[0] else Tk.DISABLED
        for w in self.job_state_dependents:
            w.config(state=state)
        self.seed.set(self.job_list[i][5])

        w = self.stop_button
        if job_state == JOB_STATES[1]:
            w.config(fg='white', bg='red', state=Tk.NORMAL)
        else:
            configs = self.stop_button_configs
            w.config(fg=configs[0], bg=configs[1], state=Tk.DISABLED)

        self.folder_box.selection_clear()

    def maxnum_trials_tracer(self, *args):
        try:
            num_remaining = self.maxnum_trials.get() - self.num_jobs.get()
            state = Tk.NORMAL if num_remaining > 0 else Tk.DISABLED
            for w in self.remaining_dependents:
                w.config(state=state)
        except:
            # may be the spinbox is space
            pass

        self.update_try_model_composing_cb()

    def update_try_model_composing_cb(self):
        name = self.skip_button.cget('text')
        state = self.skip_button.cget('state')
        print("update_try_model_composing_cb: ", name, state)
        if name == 'Resume' and state == Tk.NORMAL:
            state = Tk.NORMAL
        else:
            state = Tk.DISABLED
        self.try_model_composing_cb.config(state=state)

    def terminate(self, immediately=False):
        if immediately:
            reply = True
        else:
            remaining_phrase = "" if self.maxnum_trials.get() == self.num_jobs.get() else " and stop the trials"
            reply = MessageBox.askokcancel("Terminate Confirmation",
                        "Are you sure to terminate the running job%s?" % remaining_phrase,
                        parent=self,
                        )
        if reply:
            self.runner.terminate()
            self.terminating = True

    def skip_to_next_trial(self, restart_params=None):
        num_remaining = self.maxnum_trials.get() - self.num_jobs.get()
        assert num_remaining > 0

        abort_phrase = "" if (self.terminating or self.inspecting) else " abort the running job and"
        skip_phrase = "resume from the next" if self.inspecting else "skip to the next"
        reply = MessageBox.askokcancel("Skip Confirmation",
                    "Are you sure to%s %s trial?" % (abort_phrase, skip_phrase),
                    parent=self,
                    )
        if reply:
            if self.terminating or self.inspecting:
                self.terminating = False
                self.start_next_trial(restart_params=restart_params)
            else:
                self.runner.terminate()

    def on_entry_root_folder(self):
        print("on_entry_root_folder")
        work_folder = os.path.join(self.optjob_folder.get(), '000')
        if os.path.exists(work_folder):
            self.canvas.get_ready_from_folder(work_folder)
            self.update_information(work_folder=work_folder)

    def set_inspection_mode(self):
        self.job_state.set(JOB_STATES[2])
        for w in self.remaining_dependents:
            w.config(state=Tk.DISABLED)

    def append_result(self, last_info):
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

    def get_known_best_params(self, init_params):
        if len(self.result_list) == 0:
            ret_params = init_params
            self.logger.info("got best params from the initialization: num_params=%d", len(ret_params))
        else:
            best_result = self.result_list[self.known_best_index]
            ret_params = best_result.params
            self.logger.info("got best params from the %s result: num_params=%d", ordinal_str(self.known_best_index), len(ret_params))
        return ret_params

    def export(self, debug=True):
        if debug:
            from importlib import reload
            import Optimizer.LrfExporter
            reload(Optimizer.LrfExporter)
        from .LrfExporter import LrfExporter
        import molass_legacy.KekLib.CustomMessageBox as MessageBox

        optinit_info = self.optinit_info
        sd = optinit_info.sd
        dsets = optinit_info.dsets
        fullopt = self.fullopt
        canvas = self.canvas
        params = canvas.demo_info[1][canvas.curr_index]
        try:
            exporter = LrfExporter(self.fullopt, params, dsets)
            folder = exporter.export()
            fig_file = os.path.join(folder, "result_fig.jpg")
            self.save_the_result_figure(fig_file=fig_file)
            MessageBox.showinfo(
                    'Success Notification',
                    'Successfully exported to "%s"' % folder,
                    parent=self,
                    )
        except Exception as exc:
            log_exception(self.logger, "export: ")
            MessageBox.showerror(
                    'Error Notification',
                    'Failed to export due to: %s' % str(exc),
                    parent=self,
                    )

    def modify_restart(self, debug=True):
        if debug:
            from importlib import reload
            import Optimizer.ModifyStateDialog
            reload(Optimizer.ModifyStateDialog)
        from .ModifyStateDialog import ModifyStateDialog

        # self.grab_set()  # temporary fix to the grab_release problem

        try:
            mdialog = ModifyStateDialog(self.parent, self)
            mdialog.show()
            if mdialog.applied:
                pe_proxy = mdialog.get_pe_proxy()
                pdialog = self.parent_dialog
                v2_menu = pdialog.get_v2_menu()
                pdialog.after(500, lambda: v2_menu.show_peak_editor(pe_proxy=pe_proxy))
                self.cancel()
        except:
            from molass_legacy.KekLib.ExceptionTracebacker import log_exception
            log_exception(self.logger, "modify_restart: ", n=10)

        # self.grab_set()  # temporary fix to the grab_release problem

    def devel_test_back(self):
        from importlib import reload
        import Theory.PdbCrysolRoute
        reload(Theory.PdbCrysolRoute)
        from Theory.PdbCrysolRoute import demo
        demo(self)

    def devel_test(self):
        print("devel_test")
        from importlib import reload
        import Tools.EmbedCushion
        reload(Tools.EmbedCushion)
        from molass_legacy.Tools.EmbedCushion import embed_cushion

        embed_cushion(self)
