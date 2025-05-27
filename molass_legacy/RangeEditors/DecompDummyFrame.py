"""
    DecompDummyFrame.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""

from molass_legacy.KekLib.OurTkinter import Tk

class DecompDummyFrame(Tk.Frame):
    def __init__(self, parent, model):
        self.model = model
        Tk.Frame.__init__(self, parent)

    def is_delayed(self):
        return self.model.is_delayed()

    def get_decomp_score(self):
        # return a very large value to avoid being selected
        return 1e+10

    def get_fit_error(self):
        # return a very large value to avoid contradiction
        return 1e+10

    def close_figs(self):
        # nothing to do
        pass
