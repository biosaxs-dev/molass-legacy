"""
    FunctionChanger.py

    Copyright (c) 2022, SAXS Team, KEK-PF
"""
import re
import logging
from molass_legacy.KekLib.OurTkinter import Tk, Dialog, ttk
from .FuncImporter import get_objective_function_info

func_code_re = re.compile(r"(F\d{4})")

class FunctionChanger(Dialog):
    def __init__(self, parent, js_canvas):
        self.logger = logging.getLogger(__name__)
        self.parent = parent
        self.js_canvas = js_canvas
        Dialog.__init__(self, parent, "Function Changer", visible=False)

    def show(self):
        self._show()

    def body(self, body_frame):
        class_code = self.js_canvas.dialog.class_code

        func_info = get_objective_function_info(logger=self.logger, default_func_code=class_code)
        self.func_dict = func_info.func_dict
        self.key_list = []
        for key in func_info.key_list:
            m = func_code_re.search(key)
            if m:
                func_code = m.group(1)
                if func_code[0:3] == class_code[0:3]:
                    self.key_list.append(key)

        self.selected_function = Tk.StringVar()
        self.selected_function.set(self.key_list[func_info.default_index])
        self.function_box = ttk.Combobox(master=body_frame, values=self.key_list, textvariable=self.selected_function,
                                width=80, justify=Tk.CENTER, state=Tk.NORMAL)
        self.function_box.pack(side=Tk.LEFT)

    def apply(self):
        func_key = self.selected_function.get()
        m = func_code_re.search(func_key)
        class_code = m.group(1)

        js_canvas = self.js_canvas
        js_canvas.dialog.class_code = class_code
        js_canvas.draw_suptitle()
        js_canvas.mpl_canvas.draw()

        func_class = self.func_dict[func_key]
        state_info = js_canvas.dialog.state_info
        state_info.optinit_info.class_code = class_code
        state_info.optinit_info.fullopt_class = func_class
