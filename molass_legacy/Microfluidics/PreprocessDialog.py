# coding: utf-8
"""
    PreprocessDialog.py

    Copyright (c) 2019, SAXS Team, KEK-PF
"""
import os
import numpy as np
from molass_legacy.KekLib.OurTkinter import Tk, Dialog, ttk
from molass_legacy.KekLib.TkSupplements import tk_set_icon_portable
from MethodAdjusterFrame import MethodAdjusterFrame
from SubtractorFrame import SubtractorFrame

class PreprocessDialog(Dialog):
    def __init__(self, parent):
        Dialog.__init__(self, parent, "Microfluidic Preprocess", visible=False)

    def body(self, body_frame):
        tk_set_icon_portable(self)

        self.notebook = ttk.Notebook(body_frame)
        self.notebook.pack()

        mtd_frame = MethodAdjusterFrame(body_frame)
        self.notebook.add(mtd_frame, text='Method File Ajuster')

        subt_frame = SubtractorFrame(body_frame)
        self.notebook.add(subt_frame, text='Intensity Subtractor')

        current_tab = self.notebook.tabs()[-1]
        self.notebook.select(current_tab)

    def buttonbox(self):
        box = Tk.Frame(self)
        box.pack(fill=Tk.X, expand=1, padx=5)
        bottom_space = Tk.Frame(self, height=10)
        bottom_space.pack()

        w = Tk.Button(box, text="▼ Exit", width=10, command=self.exit)
        w.pack(side=Tk.LEFT, padx=10, pady=5)

        w = Tk.Button(box, text="▶ Main", width=10, command=self.ok)
        w.pack(side=Tk.RIGHT, padx=10, pady=5)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

    def show(self):
        self._show(wait=False)
        self.waited_widget = Tk.Frame(self)
        self.wait_window(self.waited_widget)

    def ok(self):
        self.apply()
        self.close()

    def exit(self):
        self.close()

    def close(self):
        self.withdraw()
        self.update_idletasks()
        self.grab_release()
        self.waited_widget.destroy()

    def apply(self):
        self.applied = True
