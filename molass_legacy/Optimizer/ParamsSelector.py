"""
    Optimizer.ParamsSelector.py

    Copyright (c) 2021-2022, SAXS Team, KEK-PF
"""
from molass_legacy.KekLib.OurTkinter import Tk, Dialog

class ParamsSelector(Dialog):
    def __init__(self, parent, optimizer, params):
        self.optimizer = optimizer
        self.params = params
        self.selection = None
        Dialog.__init__(self, parent, "Parameter Selection", visible=False)

    def show(self):
        self._show()

    def body(self, body_frame):
        label = Tk.Label(body_frame, text="Select exactly two parameters for analysis.", bg="white")
        label.pack(fill=Tk.X, pady=10)
        self.psheet = self.optimizer.params_type.get_params_sheet(body_frame, self.params, None, self.optimizer)
        self.psheet.enable_selection()
        self.psheet.pack()

    def set_selection(self, sel_list):
        self.psheet.set_selection(sel_list)

    def get_selection(self):
        return self.selection

    def ok(self):
        try:
            self.selection = self.psheet.get_selection()    # get it before destruction
        except ValueError:
            return
        Dialog.ok(self)
