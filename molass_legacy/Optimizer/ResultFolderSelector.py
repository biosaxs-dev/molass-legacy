"""
    Optimizer.ResultFolderSelector.py

    Copyright (c) 2021-2024, SAXS Team, KEK-PF
"""
import os
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.TkCustomWidgets import FolderEntry
from molass_legacy._MOLASS.SerialSettings import set_setting
from molass_legacy._MOLASS.Version import is_developing_version
from .FullOptResult import FILES
from .FullOptUtils import in_folder_consistency_ok

def job_folder_checker(self):
    optjob_folder = self.folder.get()
    files_ok = True
    for file in FILES[:-1]: # except "bounds.txt", which is optional
        path = os.path.join(optjob_folder, file)
        if not os.path.exists(path):
            files_ok = False
    entry = self.folder_entry
    if files_ok:
        path = os.path.join(optjob_folder, FILES[1])    # "in_data_info.txt"
        if in_folder_consistency_ok(self.in_folder, path, self):
            entry.config(fg="black")
            self.proceed_btn.config(state=Tk.NORMAL)
        else:
            entry.config(fg="red")
    else:
        entry.config(fg="red")
        self.update()
        import molass_legacy.KekLib.CustomMessageBox as MessageBox
        MessageBox.showerror("Invalid Folder",
            'This folder does not include required files.\n'
            'Select a folder named like "*/jobs/nnn".',
            parent=self)

def job_folder_applier(self):
    folder = self.folder.get()
    set_setting("optjob_folder", folder)
    optimizer_folder = os.path.dirname(os.path.dirname(folder))
    set_setting("optimizer_folder", optimizer_folder)

def joblist_folder_checker(self):
    # TODO: check if the folder is a job list folder
    folderpath = self.folder.get()
    print("joblist_folder_checker: folder=", folderpath)
    foldername = folderpath.split("/")[-1]
    if not foldername == "jobs":
        self.folder_entry.config(fg="red")
        self.update()
        import molass_legacy.KekLib.CustomMessageBox as MessageBox
        MessageBox.showerror("Invalid Folder",
            'This folder does not end with "jobs".\n'
            'Select a folder named like "*/jobs".',
            parent=self)
    else:
        self.folder_entry.config(fg="black")
        self.proceed_btn.config(state=Tk.NORMAL)

def joblist_folder_applier(self):
    folder = self.folder.get()
    optimizer_folder = os.path.dirname(folder)
    set_setting("optimizer_folder", optimizer_folder)

TARGET_FOLDER_DICT = {
    "Result Viewer" : ("Job Folder", job_folder_checker, job_folder_applier),
    "Result Animation" : ("Job List Folder", joblist_folder_checker, joblist_folder_applier),
}

if is_developing_version():
    TARGET_FOLDER_DICT["Parameter Transition"] = ("Job List Folder", joblist_folder_checker, joblist_folder_applier)

class ResultFolderSelector(Dialog):
    def __init__(self, parent, in_folder, proc_name):
        self.parent = parent
        self.in_folder = in_folder
        self.proc_name = proc_name
        self.applied = False
        Dialog.__init__(self, parent, proc_name, visible=False, location='lower center')

    def show(self):
        self._show()

    def body(self, body_frame):

        frame = Tk.Frame(body_frame)
        frame.pack(padx=20, pady=10)

        prompt_name, checker, applier = TARGET_FOLDER_DICT[self.proc_name]
        self.applier = applier

        label = Tk.Label(frame, text=prompt_name + ":")
        label.pack(side=Tk.LEFT)

        self.folder = Tk.StringVar()
        self.folder_entry = FolderEntry(frame, textvariable=self.folder, width=80,
            on_entry_cb=lambda: checker(self))
        self.folder_entry.pack(side=Tk.LEFT)

    def buttonbox(self):
        box = Tk.Frame(self)
        box.pack()

        w = Tk.Button(box, text="Cancel", width=10, command=self.cancel)
        w.pack(side=Tk.LEFT, padx=20, pady=5)
        w = Tk.Button(box, text="Proceed", width=10, command=self.ok, state=Tk.DISABLED)
        w.pack(side=Tk.LEFT, padx=20, pady=5)
        self.proceed_btn = w

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

    def apply(self):
        from molass_legacy._MOLASS.SerialSettings import set_setting
        self.applier(self)
        self.applied = True

    def get_logger(self):
        from molass_legacy.KekLib.ChangeableLogger import Logger
        logger = Logger(os.path.join(self.folder.get(), "opt_tracer.log"))
        return logger

    def get_result(self, folder=None, debug=False):
        if debug:
            import Optimizer.FullOptResult
            from importlib import reload
            reload(Optimizer.FullOptResult)
        from .FullOptResult import FullOptResult
        if folder is None:
            folder = self.folder.get()
        result = FullOptResult(self.parent.serial_data, self.parent.pre_recog, folder)
        return result

def result_viewer_impl(selector_dialog, parent_dialog, debug=False):
    if debug:
        from importlib import reload
        import Optimizer.FullOptDialog
        reload(Optimizer.FullOptDialog)
    from .FullOptDialog import FullOptDialog
    from molass_legacy.Trimming import get_trimming_info_list, set_trimming_info_list

    save_list = get_trimming_info_list()

    optjob_folder = selector_dialog.folder.get()
    set_setting('optworking_folder', optjob_folder)
    logger = selector_dialog.get_logger()
    logger.info("preparing FullOptDialog for view from %s.", optjob_folder)

    result = selector_dialog.get_result()
    optinit_info = result.get_init_info()   # this should restore required temporary settings by settings.load()

    demo_index = result.get_demo_index()
    job_name = result.get_job_name()
    dialog = FullOptDialog(parent_dialog.parent, parent_dialog, optinit_info, demo_index=demo_index, inspecting=True, job_name=job_name)
    parent_dialog.fullopt_dialog = dialog
    dialog.show()

    set_trimming_info_list(save_list, logger)

def result_animation_bridge(selector_dialog, debug=False):
    if debug:
        from importlib import reload
        import Optimizer.ResultAnimation
        reload(Optimizer.ResultAnimation)
    from molass_legacy.Optimizer.ResultAnimation import show_result_animation_impl
    show_result_animation_impl(selector_dialog)

def parameter_transition_brigde(selector_dialog, debug=False):
    if debug:
        from importlib import reload
        import Optimizer.ParameterTransition
        reload(Optimizer.ParameterTransition)
    from molass_legacy.Optimizer.ParameterTransition import show_parameter_transition_impl
    try:
        show_parameter_transition_impl(selector_dialog)
    except:
        import traceback
        traceback.print_exc()

def show_result_folder_selector_impl(parent_dialog, proc_name, ready_cb=None, debug=False):
    """
    task: consider unifying with Optimizer.OptimizerUtils.show_peak_editor_impl
    """

    in_folder = parent_dialog.in_folder.get()
    selector_dialog = ResultFolderSelector(parent_dialog, in_folder, proc_name)
    if ready_cb is not None:
        ready_cb(selector_dialog)

    parent_dialog.fullopt_dialog = None
    selector_dialog.show()

    IMPL_CLOSURE_DICT = {
        "Result Viewer": lambda: result_viewer_impl(selector_dialog, parent_dialog, debug=debug),
        "Result Animation" : lambda: result_animation_bridge(selector_dialog, debug=debug),
    }

    if is_developing_version():
        IMPL_CLOSURE_DICT["Parameter Transition"] = lambda: parameter_transition_brigde(selector_dialog, debug=True)

    if selector_dialog.applied:
        impl_closure = IMPL_CLOSURE_DICT[proc_name]
        impl_closure()