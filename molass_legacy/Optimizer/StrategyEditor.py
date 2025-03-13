"""
    Optimizer.StrategyEditor.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import os
from tksheet import Sheet
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.ImportUtils import import_module_from_path

class StrategyEditor(Dialog):
    def __init__(self, parent):
        self.parent = parent
        Dialog.__init__(self, parent, "Strategy Editor", visible=False)

    def show(self):
        self._show()
    
    def body(self, body_frame):
        num_components = self.parent.get_num_peaks()
        model_name = self.parent.get_model_name()
        module_name = model_name.title() + "Sheet"
        module_path = os.path.join(os.path.join(os.path.dirname(__file__), "Strategies"), module_name + ".py")
        module = import_module_from_path(module_name, module_path)
        data_list = module.get_data_list(num_components)
        self.sheet = Sheet(body_frame, width=800, height=600, data=data_list)
        self.sheet.headers(["parameter name", "first", "middle", "last"])
        self.sheet.pack()