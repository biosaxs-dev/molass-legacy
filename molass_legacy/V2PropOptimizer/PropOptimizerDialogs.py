"""
    V2PropOptimizer.PropOptimizerDialogs.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import queue
from molass_legacy.KekLib.OurTkinter import Tk, Dialog, ttk
from molass_legacy.KekLib.ReadOnlyText import ReadOnlyText

class PropOptimizerProgress(Dialog):
    def __init__(self, parent, job_args):
        self.done = False
        self.job_args = job_args

        Dialog.__init__(self, parent, "Proportion Optimizer Progress", visible=False)

    def show(self):
        self._show()

    def body(self, body_frame):
        self.progress_bar = ttk.Progressbar(body_frame, orient ="horizontal", length=500, mode="determinate")
        self.progress_bar.pack(padx=20, pady=10)

        self.progress_text = ReadOnlyText(body_frame, width=80, height=5)
        self.progress_text.pack(padx=20, pady=10)
        self.progress_text.update()

        self.after(200, self.start_optimizer)

    def buttonbox(self):
        # task: pack the close button so that it won't hide
        box = Tk.Frame(self)
        box.pack()

        w = Tk.Button(box, text="Cancel", width=10, command=self.cancel)
        w.pack(side=Tk.LEFT, pady=10)
        self.okcancel_btn = w

    def cancel(self):
        Dialog.cancel(self)

    def start_optimizer(self):
        from molass_legacy.KekLib.KillableThread import Thread
        self.progress_queue = queue.Queue()
        self.thread = Thread(
                        target=self.optimizer_thread,
                        name='V2PropOptimizer',
                        args=[]
                        )
        self.thread.start()
        self.counter = 0
        self.progress_bar['maximum'] = 10
        self.last_index = self.progress_text.index(Tk.INSERT)
        self.after(200, self.update_progress_text)

    def update_progress_text(self):
        progress = self.progress_queue.get()
        print("progress=", progress)
        if progress is not None:
            i = progress[0]
            if i < 0:
                self.progress_bar['value'] = self.progress_bar['maximum']
                self.done = True
                self.thread.join()
                self.after(200, self.cancel)
                return
            else:
                self.progress_bar['value'] = i
                self.update()
        self.after(200, self.update_progress_text)

    def optimizer_thread(self, devel=True):
        if devel:
            from importlib import reload
            import V2PropOptimizer.PropOptimizer
            reload(V2PropOptimizer.PropOptimizer)
        from V2PropOptimizer.PropOptimizerUtils import compute_optimal_proportion
        compute_optimal_proportion(self.progress_queue, self.job_args)

class PropOptimizerResult(Dialog):
    def __init__(self, parent, result):
         Dialog.__init__(self, parent, "Proportion Optimizer Progress", visible=False)

    def show(self):
        self._show()

    def body(self, body_frame):
        text = Tk.Label(body_frame, text="In Preparation")
        text.pack()

class JobArgs:
    def __init__(self, **entries): 
        self.__dict__.update(entries)
    def __str__(self):
        return str(self.__dict__)

def show_optimizer_dialog(parent, modelname, init_prop, v2_optimizer):
    job_args = JobArgs(modelname=modelname, init_prop=init_prop, v2_optimizer=v2_optimizer)
    dialog = PropOptimizerProgress(parent, job_args)
    dialog.show()

    if not dialog.done:
        return

    result = None
    dialog = PropOptimizerResult(parent, result)
    dialog.show()
