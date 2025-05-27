"""
    KekLib.OnTheFlyUtils.py

    Copyright (c) 2024, Masatsuyo Takahashi, KEK-PF
"""
import threading
from molass_legacy.KekLib.OurTkinter import Tk, Dialog, ttk
from molass_legacy.KekLib.DebugPlot import push, pop, get_parent

class ProgressDialog(Dialog):
    def __init__(self, parent, progress_queue, title="Progress", length=400, num_steps=10, interval=500):
        self.progress_queue = progress_queue
        self.length = length
        self.num_steps = num_steps
        self.interval = interval
        Dialog.__init__(self, parent, title, visible=False)

    def show(self):
        self._show()

    def body(self, body_frame): 
        self.mpb = ttk.Progressbar(body_frame,orient ="horizontal", length=self.length, mode="determinate")
        self.mpb.pack(padx=5, pady=5)
        self.mpb["maximum"] = self.num_steps
        self.update_progress()

    def update_progress(self):
        step = self.get_progress()
        if step is None:
            self.after(self.interval, self.update_progress)
            return

        if step < 0:
            self.cancel()

        if step < self.num_steps:
            self.mpb["value"] = step
            self.after(self.interval, self.update_progress)
            self.update()
        else:
            self.mpb["value"] = self.num_steps
            self.update()
            self.cancel()            

    def get_progress(self):
        return self.progress_queue.get()

def show_progress(func, progress_queue, num_steps, parent=None):
    thread = threading.Thread(target=func)
    thread.start()
    push()
    if parent is None:
        parent = get_parent()
    progress_queue.put(0)
    dialog = ProgressDialog(parent, progress_queue, num_steps=num_steps)
    dialog.show()
    pop()
    thread.join()