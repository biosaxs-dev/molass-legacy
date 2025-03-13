"""
    Simulative.SimulationBridge.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import os
from threading import Thread
from molass_legacy.KekLib.ChangeableLogger import Logger
from molass_legacy.KekLib.OurTkinter import Tk, ttk, Dialog
from molass_legacy._MOLASS.SerialSettings import load_settings, get_setting

MAXNUM_STEPS = 10

class PreparationProgress(Dialog):
    def __init__(self, parent, lrf_src):
        load_settings(lrf_src.run_setting_file)
        analysis_folder = get_setting('analysis_folder')
        log_file = os.path.join(analysis_folder, "simulation.log")
        self.logger = Logger(log_file)  # The logging module is thread-safe
        self.parent = parent
        self.lrf_src = lrf_src
        self.canceled = False
        self.ready = False
        Dialog.__init__(self, parent, "Preparation Progress", visible=False)

    def show(self):
        self._show()

    def body(self, body_frame):
        label = Tk.Label(body_frame, text="Preparation Progress")
        label.pack(padx=10, pady=10)
        self.pbar = ttk.Progressbar(body_frame, orient ="horizontal", length=400, mode="determinate")
        self.pbar["maximum"] = MAXNUM_STEPS
        self.pbar["value"] = 0
        self.pbar.pack(padx=10, pady=10)

        self.parent.after(100, self.start_preparation)

    def buttonbox( self ):
        box = Tk.Frame(self)
        box.pack()
        w = Tk.Button(box, text="Cancel", width=10, command=lambda: self.user_cancel(ask=True) )
        w.pack(side=Tk.LEFT, padx=5, pady=5)

    def apply(self):
        self.applied = True

    def user_cancel(self, ask=False):
        if ask:
            if not self.ask_cancel():
                return
        self.prepare_thread.terminate()
        self.cancel()

    def ask_cancel(self):
        from tkinter import messagebox
        ret = messagebox.askokcancel("Cancel", "Are you sure you want to cancel?")
        if ret:
            self.canceled = True

    def update_progress(self):
        if not self.ready and not self.canceled:
            self.parent.after(200, self.update_progress)
        else:
            if self.canceled:
                pass
            else:
                self.prepare_thread.join()
                self.ok()

    def start_preparation(self):
        self.prepare_thread = Thread(target=self.prepare)
        self.prepare_thread.start()
        self.update_progress()

    def prepare(self, use_study=True):
        if use_study:
            from importlib import reload
            import Models.Stochastic.MomentsStudy
            reload(Models.Stochastic.MomentsStudy)
            from molass_legacy.Models.Stochastic.MomentsStudy import moments_study_impl
            self.guess_info = moments_study_impl(self.lrf_src, return_rgs=True, progress_cb=self.progress_cb)
        else:
            self.guess_info = self.lrf_src.guess_lnpore_params(return_rgs=True, progress_cb=self.progress_cb)
        self.ready = True

    def progress_cb(self, progress):
        self.pbar["value"] = progress

def simulation_bridge_impl(parent, lrf_src):
    prep = PreparationProgress(parent, lrf_src)
    prep.show()
    if not prep.canceled:
        parent.quit()

    from importlib import reload
    import Simulative.SimulationCushion
    reload(Simulative.SimulationCushion)
    from Simulative.SimulationCushion import demo_cushion

    demo_cushion(lrf_src, prep.guess_info, parent=parent)

def simulation_bridge(lrf_src):
    from molass_legacy.KekLib.TkUtils import get_tk_root
    print('simulation_main')
    root = get_tk_root()
    root.after(1000, lambda: simulation_bridge_impl(root, lrf_src))
    root.mainloop()